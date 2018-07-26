// decoder/cuda-decoder.h

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

#ifndef KALDI_DECODER_CUDA_DECODER_H_
#define KALDI_DECODER_CUDA_DECODER_H_

#include "util/stl-utils.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"
#include "omp.h"
#include <cuda_runtime_api.h>

#include "decoder/cuda-decoder-utils.h"

namespace kaldi {

    /** 
     * Simple Cuda Decoder
     */
    class CudaDecoder;

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

            // Decode this utterance.
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
