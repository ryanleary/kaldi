// decoder/simple-decoder.h

// Copyright 2009-2013  Microsoft Corporation;  Lukas Burget;
//                      Saarland University (author: Arnab Ghoshal);
//                      Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_DECODER_SIMPLE_DECODER_H_
#define KALDI_DECODER_SIMPLE_DECODER_H_

#ifdef __CUDACC__
  #define HOST __host__
  #define DEVICE __device__

#else
  #define HOST
  #define DEVICE
#endif

#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"
#include "omp.h"
#include <cuda_runtime_api.h>

namespace kaldi {
  
/** 
 * Simple Cuda Decoder
 */
class CudaDecoder;

class CudaFst {
  public:
    typedef fst::StdArc StdArc;
    typedef StdArc::Weight StdWeight;
    typedef StdArc::Label Label;
    typedef StdArc::StateId StateId;
    
    CudaFst() {};
    void initialize(const fst::Fst<StdArc> &fst);
    void finalize();

    inline uint32_t NumStates() const {  return numStates; }
    inline StateId Start() const { return start; }    
    HOST DEVICE inline float Final(StateId state) const;
    size_t getCudaMallocBytes() const { return bytes_cudaMalloc; }
  private:
    friend class CudaDecoder;
  
    unsigned int numStates;               //total number of states
    StateId  start;

    unsigned int max_ilabel;              //the largest ilabel
    unsigned int e_count, ne_count, arc_count;       //number of emitting and non-emitting states
  
    //This data structure is similar to a CSR matrix format 
    //where I have 2 matrices (one emitting one non-emitting).
 
    //Offset arrays are numStates+1 in size. 
    //Arc values for state i are stored in the range of [i,i+1)
    //size numStates+1
    unsigned int *e_offsets_h,*e_offsets_d;               //Emitting offset arrays 
    unsigned int *ne_offsets_h, *ne_offsets_d;            //Non-emitting offset arrays
 
    //These are the values for each arc. Arcs belonging to state i are found in the range of [offsets[i], offsets[i+1]) 
    //non-zeros (Size arc_count+1)
    BaseFloat *arc_weights_h, *arc_weights_d;
    StateId *arc_nextstates_h, *arc_nextstates_d;
    int32 *arc_ilabels_h, *arc_ilabels_d;
    int32 *arc_olabels_h;

    //final costs
    float *final_h;
    //allocation size
    size_t bytes_cudaMalloc;
};

template<typename T>
class CudaVector {
    public:
      HOST DEVICE inline T& operator[](uint32_t idx); 
      HOST DEVICE inline const T& operator[](uint32_t idx) const; 
      inline void allocate(uint32_t max_size);
      inline void free();
      HOST DEVICE inline uint32_t size() const; 
      HOST DEVICE inline void push_back(const T &val); 
      inline void clear(cudaStream_t stream=0); 
      inline bool empty() const;
      inline void swap(CudaVector<T> &v); 
      inline void copy_all_to_host(cudaStream_t stream=0);
      inline void copy_all_to_device(cudaStream_t stream=0);
      inline void copy_size_to_host(cudaStream_t stream=0);
      inline void copy_size_to_device(cudaStream_t stream=0);
      inline void copy_data_to_host(cudaStream_t stream=0);
      inline void copy_data_to_device(cudaStream_t stream=0);

      inline size_t getCudaMallocBytes(); 
    private:
      
      uint32_t *count_d, *count_h;
      uint32_t max_size;
      T* mem_d, *mem_h;
};

struct CudaDecoderConfig {
  BaseFloat beam;
  uint32_t max_tokens;
  uint32_t max_tokens_per_frame;


  CudaDecoderConfig(): beam(16.0),
                       max_tokens(300000000),
                       max_tokens_per_frame(1000000) {}
  
  void Register(OptionsItf *opts) {
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("max-tokens-pre-allocated", &max_tokens, "Total number of tokens pre-allocated (equivalent to reserve in a std vector).  If actual usaged exceeds this performance will be degraded");
    opts->Register("max-tokens-per-frame", &max_tokens_per_frame, "Number of tokens allocated per frame. If actual usaged exceeds this the results are undefined.");
   }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && max_tokens > 0 && max_tokens_per_frame > 0);
  }
};

// is mostly read in coalesced accesses
struct InfoToken { // we needed to take StateId out
    int prev_token;
    int arc_idx;
};

class CudaDecoder {

 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Weight StdWeight;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef float CostType;

  CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config);  
  ~CudaDecoder();

  inline size_t getCudaMallocBytes() const { return bytes_cudaMalloc; } 
  inline size_t getCudaMallocManagedBytes() const { return bytes_cudaMallocManaged;  }

  /// Decode this utterance.
  /// Returns true if any tokens reached the end of the file (regardless of
  /// whether they are in a final state); query ReachedFinal() after Decode()
  /// to see whether we reached a final state.
  bool Decode(DecodableInterface *decodable);

  bool ReachedFinal() const;

  // GetBestPath gets the decoding traceback. If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into account final-probs.
  // fst_out will be empty (Start() == kNoStateId) if nothing was available due to
  // search error.
  // If Decode() returned true, it is safe to assume GetBestPath will return true.
  // It returns true if the output lattice was nonempty (i.e. had states in it);
  // using the return value is deprecated.
  bool GetBestPath(Lattice *fst_out, bool use_final_probs = true) const;
  
  /// *** The next functions are from the "new interface". ***
  
  /// FinalRelativeCost() serves the same function as ReachedFinal(), but gives
  /// more information.  It returns the difference between the best (final-cost plus
  /// cost) of any token on the final frame, and the best cost of any token
  /// on the final frame.  If it is infinity it means no final-states were present
  /// on the final frame.  It will usually be nonnegative.
  BaseFloat FinalRelativeCost() const;

  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need
  /// to call this.  You can call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance. 
  void InitDecoding();  

  struct EndAndNarcs{
    int end;
    int narcs;
  };

  union QEndAndNarcs {
      EndAndNarcs split;
      unsigned long long both;
  };

  struct PreprocessParams {
      StateId *d_main_q_state; 
      CostType *d_main_q_cost;
      InfoToken *d_main_q_info; 

      int *d_main_q_local_offset; 
      int *h_main_q_local_offset; 
      int *d_main_q_end; 
      QEndAndNarcs *d_main_q_end_and_narcs_i2; 
      int *d_main_q_narcs; 
      int *h_main_q_end;
      int *h_main_q_narcs; 

      int *h_q_overflow; 
      int q_capacity;

      StateId *d_aux_q_state; 
      CostType *d_aux_q_cost;
      InfoToken *d_aux_q_info; 
      int *d_aux_q_end; 
      int *h_aux_q_end;
      
      int *d_degrees_scan; 
      unsigned int *d_arc_offsets; 
      int *d_main_q_arc_offsets; // offsets, relative to the queue

      int *d_state_cost; 
      BaseFloat *d_cutoff; 

      int *d_degrees_block_scan; 
      int *d_n_CTA_done;
  };


  struct ExpandArcParams {
      StateId *d_main_q_state; 
      CostType *d_main_q_cost;
      InfoToken *d_main_q_info; 
      int *d_degrees_scan; 

      int *d_main_q_narcs; 
      int *h_main_q_narcs; 

      int *d_main_q_local_offset;
      int *h_main_q_local_offset;
      int main_q_global_offset;
      int *d_main_q_end;

      int *h_main_q_end;

      StateId *d_aux_q_state; 
      CostType *d_aux_q_cost;
      InfoToken *d_aux_q_info; 
      int *d_aux_q_end;
      int *h_aux_q_end; 

      int *h_q_overflow; 
      int q_capacity;

      int *d_q_arc_offsets; 
      int *arc_ilabels; 

      BaseFloat *arc_weights; 
      StateId *arc_nextstates; 
      BaseFloat *d_cutoff;
      BaseFloat *d_loglikelihoods;
      BaseFloat beam; 

      int *d_lookup;
      bool is_emitting;
      int *d_n_CTA_done;
  };

  class TokenVector {
      size_t capacity, size;
      cudaStream_t copy_st;
      InfoToken *h_data;
      public:
      TokenVector();
      void Reset();
      void SetCudaStream(cudaStream_t st);
      void CopyFromDevice(size_t offset, InfoToken *d_ptr, size_t count);    
      void Reserve(size_t min_capacity);    
      InfoToken *GetRawPointer() const;
      virtual ~TokenVector();
  };


  void ExpandArcs(int nthreads, const ExpandArcParams &params);

  void ContractAndPreprocess(PreprocessParams &params);
  void PreprocessInPlace(PreprocessParams &params);
  void FinalizePreprocessInPlace();

  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.  If it returns false, then no tokens are alive,
  /// which is a kind of error state.
  void AdvanceDecoding(DecodableInterface *decodable,
                         int32 max_num_frames = -1);
  
  /// Returns the number of frames already decoded.  
  int32 NumFramesDecoded() const { return num_frames_decoded_; }

  StateId *d_main_q_state, *d_aux_q_state; 
  CostType *d_main_q_cost, *d_aux_q_cost;
  InfoToken *d_main_q_info, *d_aux_q_info;

  // Local offset (in d_q_from_*)
  int *d_main_q_local_offset;
  int *h_main_q_local_offset; // TODO not needed 

  // Global offset (in h_all_*)
  // Used to set the "prev_token" in new tokens
  int main_q_global_offset;

  // Pointer to end index in from (equal to size + offset)
  int *d_main_q_end;
  int *h_main_q_end;

  // total number of arcs contained in main q [off, end[
  // ie total # of arcs from tok.next_state, where tok is in [off,end[
  // (actually one "valid arcs" are counted, cf Preprocess)
  int *d_main_q_narcs;
  int *h_main_q_narcs; // pinned

  // Contains both q_end and narcs
  QEndAndNarcs *d_main_q_end_and_narcs_i2; 

  // Pointer to end index in to (equal to size + 0) (no offset)
  int *d_aux_q_end;
  int *h_aux_q_end;

  int *h_q_overflow;

  TokenVector h_all_tokens_info; // on host

  // Those are filled only if necessary
  StateId *h_main_q_state; // on host
  CostType *h_main_q_cost; // on host

  // Used to detect last CTA alive in some kernels
  int *d_n_CTA_done;

  // Scan of the outgoing arc degrees of tokens in [from,to[
  int *d_degrees_scan;
  // Scan of the total per block
  int *d_degrees_block_scan;

  // Cf Compute degrees
  int *d_main_q_arc_offsets;


  // Lookup table of all the costs
  // d_state_cost[state] -> best cost for that state
  // Resetted between frames
  // Costs is stored as an ordered int representing a float
  int *d_state_cost;

  // Current cutoff for current frame
  BaseFloat *d_cutoff;

  BaseFloat *loglikelihoods_d;

  cudaStream_t compute_st, copy_st;
  cudaEvent_t q_token_from_narcs_evt, can_write_to_main_q;

  //pre-computes log likelihoods for the current frame
  void ComputeLogLikelihoods(DecodableInterface *decodable);
 
  // ProcessEmitting decodes the frame num_frames_decoded_ of the
  // decodable object, then increments num_frames_decoded_.
  //void ProcessEmitting(DecodableInterface *decodable);

  // Descriptions in .cu file

  void InitLookup();
  void ResetLookup();
  void NonEmittingLongTail(unsigned int *d_arc_offsets, const ExpandArcParams &params);

  void GetBestCost(BaseFloat *min, int *arg, bool isfinal) const;
  void ProcessEmitting();
  void ProcessNonemitting();
  void PrintOverflowWarning();
 
  bool ProcessToken(unsigned int *d_arc_offsets, bool is_emitting);

  
  const CudaFst fst_;

  BaseFloat beam_;
  int max_tokens_, max_tokens_per_frame_;
  

  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;

  BaseFloat *cutoff;

  size_t bytes_cudaMalloc, bytes_cudaMallocManaged;

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaDecoder);
};


} // end namespace kaldi.


#endif
