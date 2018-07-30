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
            // otherwise it gets the most likely token not taking int32o account final-probs.
            // fst_out will be empty (Start() == kNoStateId) if nothing was available due to
            // search error.
            // If Decode() returned true, it is safe to assume GetBestPath will return true.
            // It returns true if the output lattice was nonempty (i.e. had states in it);
            // using the return value is deprecated.
            bool GetBestPath(Lattice *fst_out, bool use_final_probs = true) const;

            /// *** The next functions are from the "new int32erface". ***

            /// FinalRelativeCost() serves the same function as ReachedFinal(), but gives
            /// more information.  It returns the difference between the best (final-cost plus
            /// cost) of any token on the final frame, and the best cost of any token
            /// on the final frame.  If it is infinity it means no final-states were present
            /// on the final frame.  It will usually be nonnegative.
            BaseFloat FinalRelativeCost() const;

            /// InitDecoding initializes the decoding, and should only be used if you
            /// int32end to call AdvanceDecoding().  If you call Decode(), you don't need
            /// to call this.  You can call InitDecoding if you have already decoded an
            /// utterance and want to start with a new utterance. 
            void InitDecoding();  


            // Count of tokens and arcs in a queue
            // narcs = sum(number of arcs going out of token i next state) for each token in the queue
            // We use this struct to keep the two int32s adjacent in memory
            // we need this in order to update both using an atomic64 operation
            struct TokenAndArcCount {
                int32 ntokens;
                int32 narcs;
            };

            // Union structure of the TokenAndArcCount
            // We use split to access the int32s
            // We use both to update both using an atomic64
            union TokenAndArcCountUnion {
                TokenAndArcCount split;
                unsigned long long both;
            };

            // Parameters used by the Preprocess kernels
            // We store them in a struct to reduce the number of arguments passed to 
            // the kernel
            // The cuda kernel launch latency time is linear with the number of args 
            // Here we will pass only one arg, which is this struct

            struct PreprocessParams {
                StateId *d_main_q_state; 
                CostType *d_main_q_cost;
                InfoToken *d_main_q_info; 

                int32 *d_main_q_local_offset; 
                int32 *h_main_q_local_offset; 
                int32 *d_main_q_end; 
                TokenAndArcCountUnion *d_main_q_end_and_narcs_i2; 
                int32 *d_main_q_narcs; 
                int32 *h_main_q_end;
                int32 *h_main_q_narcs; 

                int32 *h_q_overflow; 
                int32 q_capacity;

                StateId *d_aux_q_state; 
                CostType *d_aux_q_cost;
                InfoToken *d_aux_q_info; 
                int32 *d_aux_q_end; 
                int32 *h_aux_q_end;

                int32 *d_degrees_scan; 
                uint32_t *d_arc_offsets; 
                int32 *d_main_q_arc_offsets; // offsets, relative to the queue

                int32 *d_state_cost; 
                BaseFloat *d_cutoff; 

                int32 *d_degrees_block_scan; 
                int32 *d_n_CTA_done;
            };

            // Parameters used by the Expand kernel
            // We store them in a struct to reduce the number of arguments passed to 
            // the kernel
            // The cuda kernel launch latency time is linear with the number of args 
            // Here we will pass only one arg, which is this struct


            struct ExpandArcParams {
                StateId *d_main_q_state; 
                CostType *d_main_q_cost;
                InfoToken *d_main_q_info; 
                int32 *d_degrees_scan; 

                int32 *d_main_q_narcs; 
                int32 *h_main_q_narcs; 

                int32 *d_main_q_local_offset;
                int32 *h_main_q_local_offset;
                int32 main_q_global_offset;
                int32 *d_main_q_end;

                int32 *h_main_q_end;

                StateId *d_aux_q_state; 
                CostType *d_aux_q_cost;
                InfoToken *d_aux_q_info; 
                int32 *d_aux_q_end;
                int32 *h_aux_q_end; 

                int32 *h_q_overflow; 
                int32 q_capacity;

                int32 *d_q_arc_offsets; 
                int32 *arc_ilabels; 

                BaseFloat *arc_weights; 
                StateId *arc_nextstates; 
                BaseFloat *d_cutoff;
                BaseFloat *d_loglikelihoods;
                BaseFloat beam; 

                int32 *d_lookup;
                bool is_emitting;
                int32 *d_n_CTA_done;
            };

private:
            //
            // Kernel wrappers
            // The following functions are wrappers for cuda kernels
            //

            //
            // ExpandArcs kernel
            // This kernel reads token from the main_q and uses the FST graph
            // to compute new token queue in aux_q
            // To do this, for each token in the main_q, we traverse each arcs going 
            // out of that token's next state. If that arc's next state is a valid candidate,
            // we create a new token and add it to aux_q.
            // For more information on the condition for the creation of a new token,
            // please refer to http://kaldi-asr.org/doc/decoders.html
            //

            void ExpandArcs(int32 nthreads, const ExpandArcParams &params);

            //
            // PreprocessAndContract kernel
            // Input  : aux_q, FST, d_state_costs, d_cutoff
            // Output : main_q, d_degrees_scan
            //
            // The Preprocess* kernels are used before executing Expand 
            // Computing data members needed by Expand (preprocess) and prune tokens on the fly (contract) 
            //
            // Pseudo code of the PreprocessAndContract kernel : 
            // - For each token in the aux_q, 
            //          compute bool is_best = (token.cost < cutoff) 
            //                                && (token.cost == d_state_costs[token.nextstate])
            // If is_best, then :
            //         1) append this token to the main_q
            //         2) compute out_degree = (# of outgoing arcs from token.nextstate) 
            //         3) compute the prefix sum of those out_degrees for all tokens appended in the main_q
            //         4) save that prefix sum in d_degrees_scan
            // Else 
            //    Do nothing (this token is pruned)
            //
            // After executing PreprocessAndContract, 
            // - aux_q is considered empty. d_aux_q_end was resetted to 0
            // - all tokens generated buy PreprocessAndContract are in the main_q,
            //   in the index range [d_main_q_local_offset, d_main_q_end[
            //
            // Note : Using a trick, we can compute everything (including the prefix sum)
            // using only one kernel (without global syncs)
            // We don't need to call FinalizePreprocessInPlace() after PreprocessAndContract
            //

            void PreprocessAndContract(PreprocessParams &params);

            //
            // PreprocessInPlace kernel
            // Input  : main_q, main_q_local_offset, FST, d_state_costs, d_cutoff
            // Output : main_q, d_degrees_scan
            //
            // The Preprocess* kernels are used before executing Expand 
            // Computing data members needed by Expand (preprocess) in place, without modifying queues
            // 
            // This is used when the input tokens were already used to generate children tokens
            // In practice, it happens when we are in a ProcessEmitting stage
            // The input tokens in ProcessEmitting at frame i were already used in ProcessNonEmitting at frame (i-1)
            // It means that those input tokens already have children. Those children token refer to the index of their 
            // input tokens in their token.prev_token data member. 
            // If we were to use PreprocessAndContract,
            // the pruning stage would change the tokens indexes - and we would have to reindex those prev_token,
            // hence the need for a PreprocessInPlace
            //
            // Pseudo code of the PreprocessInPlace kernel :
            // - For each token in the range [local_offset, end[ of the main_q, 
            //          compute bool is_best = (token.cost < cutoff) 
            //                                && (token.cost == d_state_costs[token.nextstate])
            //
            //          compute out_degree = is_best
            //                               ? (# of outgoing arcs from token.nextstate)
            //                               : 0
            // 
            //  By artifically setting the out_degree to 0 we tell the expand kernel to completely ignore that token
            //  (we rely on the 1 arc = 1 thread exact load balancing of the expand kernel)
            //
            // Then we compute the data needed by the expand kernel :
            //         1) compute the prefix sum of those out_degrees 
            //         3) save that prefix sum in d_degrees_scan
            //
            // After executing PreprocessInPlace,
            // The "active" main_q[local_offset, end[ range stays the same.
            //
            // Note : Only the first pass of the prefix sum is computed in that kernel. We then need to call
            // FinalizePreprocessInPlace
            //

            void PreprocessInPlace(PreprocessParams &params);

            // This kernel is responsible to compute the second pass of the
            // prefix sum. Must be called between PreprocessInPlace and ExpandArcs
            void FinalizePreprocessInPlace();

            /// This will decode until there are no more frames ready in the decodable
            /// object, but if max_num_frames is >= 0 it will decode no more than
            /// that many frames.  If it returns false, then no tokens are alive,
            /// which is a kind of error state.
            void AdvanceDecoding(DecodableInterface *decodable,
                    int32 max_num_frames = -1);

            /// Returns the number of frames already decoded.  
            int32 NumFramesDecoded() const { return num_frames_decoded_; }

            //
            // Data members
            //
            // Pointers in h_* refer to data on the CPU memory
            // Pointers in d_* refer to data on the GPU memory


            //
            // Tokens queues
            // 
            // We have two token queues : 
            // - the main queue
            // - the auxiliary queue
            // 
            // The auxiliary queue is used to store the raw output of ExpandArcs.
            // We then prune that aux queue and move the survival tokens in the main queue.
            // Tokens stored in the main q can then be used to generate new tokens (using ExpandArcs)
            //  
            // As a reminder, here's the data structure of a token :
            //
            // struct Token { state, cost, prev_token, arc_idx }
            //
            // For performance reasons, we split the tokens in three parts :
            // { state } , { cost }, { prev_token, arc_idx }
            // Each part has its associated queue
            // For instance, d_main_q_state[i], d_main_q_cost[i], d_main_q_info[i]
            // all refer to the same token (at index i)
            // The data structure InfoToken contains { prev_token, arc_idx }
            //
            // Note : We cannot use the aux queue to generate new tokens 
            // (ie we cannot use the aux queue as an input of ExpandArcs)
            // The generated tokens would have parents in the aux queue,
            // identifying them using their indexes in the queue. Those indexes
            // are not finals because the aux queue will be pruned.
            //

            StateId *d_main_q_state, *d_aux_q_state; 
            CostType *d_main_q_cost, *d_aux_q_cost;
            InfoToken *d_main_q_info, *d_aux_q_info;

            // ExpandArcs does not use at its input the complete main queue
            // It only reads from the index range [main_q_local_offset, end[
            int32 *h_main_q_local_offset; 
            int32 *d_main_q_local_offset;


            // end index of the main queue
            // only tokens at index i with i < main_q_end 
            // are valid tokens
            int32 *d_main_q_end;
            int32 *h_main_q_end; // pinned memory

            // Same thing for the aux queue
            int32 *d_aux_q_end;
            int32 *h_aux_q_end;

            // Each valid token the subqueue main_q[main_q_offset, main_q_end[ has 
            // a number of outgoing arcs (out-degree)
            // main_q_narcs is the sum of those numbers
            // To see when a token is considered as valid, please refer to the Preprocess kernels
            int32 *d_main_q_narcs;
            int32 *h_main_q_narcs; // pinned memory

            // Contains both main_q_end and narcs
            // The pointers refer to the same location than the d_main_q_end and d_main_q_narcs pointers,
            // ie : 
            // d_main_q_end = &d_main_q_end_and_narcs->split.ntokens
            // d_main_q_narcs = &d_main_q_end_and_narcs->split.narcs
            // We sometime need to update both end and narcs at the same time,
            // using an 64 bits atomic
            TokenAndArcCountUnion *d_main_q_end_and_narcs_i2; 

            // After each frame, we copy the main queue (GPU memory)
            // to the end of h_all_tokens_info (CPU memory)
            TokenVector h_all_tokens_info; 

            // The token at index i in the main queue has in reality 
            // a global index of (i + main_q_global_offset)
            // This global index is unique and takes into account that 
            // we've flushed the main_q back to the host. We need unique indexes 
            // for each token in order to have valid token.prev_token data members
            // and be able to backtrack at the end
            int32 main_q_global_offset;

            int32 *h_q_overflow;

            // Buffers for copies on host on the current main_q
            // Those are only buffers - and must be considered as containing 
            // uninitialized data
            // If you need to read from those,
            // please explicitely copy data from device first !
            StateId *h_main_q_state; 
            CostType *h_main_q_cost; 

            // Used to detect last CTA alive in some kernels
            int32 *d_n_CTA_done;

            // Scan of the outgoing arc degrees of tokens in [from,to[
            int32 *d_degrees_scan;
            // Scan of the total per block
            int32 *d_degrees_block_scan;

            // Cf Compute degrees
            int32 *d_main_q_arc_offsets;


            // Lookup table of all the costs
            // d_state_cost[state] -> best cost for that state
            // Resetted between frames
            // Costs is stored as an ordered int32 representing a float
            int32 *d_state_cost;

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
            void NonEmittingLongTail(uint32_t *d_arc_offsets, const ExpandArcParams &params);

            void GetBestCost(BaseFloat *min, int32 *arg, bool isfinal) const;
            void ProcessEmitting();
            void ProcessNonemitting();
            void Print32OverflowWarning();

            bool ProcessToken(uint32_t *d_arc_offsets, bool is_emitting);


            const CudaFst fst_;

            BaseFloat beam_;
            int32 max_tokens_, max_tokens_per_frame_;


            // Keep track of the number of frames decoded in the current file.
            int32 num_frames_decoded_;

            BaseFloat *cutoff;

            size_t bytes_cudaMalloc, bytes_cudaMallocManaged;

            KALDI_DISALLOW_COPY_AND_ASSIGN(CudaDecoder);
    };


} // end namespace kaldi.


#endif
