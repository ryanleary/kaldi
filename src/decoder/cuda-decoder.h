// decoder/cuda-decoder.h

// 2018 - Hugo Braun, Justin Luitjens, Ryan Leary

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

//
// To understand the following lines it is important to 
// know the difference between what we call a decoder lane and a decoder 
// channel
//
// A decoder channel is linked to one utterance. Frames
// from the same must be sent to the same channel.
// 
// A decoder lane is where the computation actually happens
// a decoder lane is given a frame and its associated channel
// and does the actual computation
//
// An analogy would be lane -> a core, channel -> a software thread
//


// Number of GPU decoder lanes
#define KALDI_CUDA_DECODER_MAX_N_LANES 40

// If we're at risk of filling the tokens queue,
// the beam is reduced to keep only the best candidates in the 
// remaining space
// We then slowly put the beam back to its default value
// beam_next_frame = min(default_beam, RECOVER_RATE * beam_previous_frame)
#define KALDI_CUDA_DECODER_ADAPTIVE_BEAM_RECOVER_RATE 1.2

// Defines for the cuda decoder kernels
// It shouldn't be necessary to change the DIMX of the kernels
#define KALDI_CUDA_DECODER_KERNEL_INIT_LOOKUP_DIMX 256
#define KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX 256
#define KALDI_CUDA_DECODER_KERNEL_EXPAND_ARCS_DIMX 256
#define KALDI_CUDA_DECODER_KERNEL_NONEM_LT_DIMX 1024

// Below that value, we launch the persistent kernel for NonEmitting
#define KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS 4096

// How many "heavy load" non emitting kernels to launch before attemping to start the persistent one
#define KALDI_CUDA_DECODER_NONEM_NEXPAND_PIPELINE_FIRST 2
// How many "heavy load" non emitting kernels to launch if previous attempt was not enough
#define KALDI_CUDA_DECODER_NONEM_NEXPAND_PIPELINE_RELAUNCH 2

// Moves data back to the CPU during computation and looks if everything looks ok
// Three levels 0 (no debugging), and 1 to 3, depending on how much we want to check things
// (performance will decrease)
#define KALDI_CUDA_DECODER_DEBUG_LEVEL 0

namespace kaldi {

   class CudaDecoder;

   struct CudaDecoderConfig {
       BaseFloat default_beam;
       int32 max_tokens;
       int32 max_tokens_per_frame;
       int32 nlanes;
       int32 nchannels;

       CudaDecoderConfig(): default_beam(15.0),
       max_tokens(50000000),
       max_tokens_per_frame(1000000) {}

       void Register(OptionsItf *opts) {
           opts->Register("beam", &default_beam, "Decoding beam.  Larger->slower, more accurate. The beam may be"
                                                          "decreased if we are generating too many tokens compared to "
                                                          "what the queue can hold (max_tokens_per_frame)");
           opts->Register("max-tokens-pre-allocated", &max_tokens, "Total number of tokens pre-allocated (equivalent to reserve in a std vector).  If actual usaged exceeds this performance will be degraded");
           opts->Register("max-tokens-per-frame", &max_tokens_per_frame, "Number of tokens allocated per frame. If actual usaged exceeds this the results are undefined.");
       }
       void Check() const {
           KALDI_ASSERT(default_beam > 0.0 && max_tokens > 0 && max_tokens_per_frame > 0);
       }
   };


   //
   // CudaDecoder
   // path-based (one-best) decoder 
   // Implementation of the CudaDecoder methods are in two files :
   // - cuda-decoder-kernels.cu for the CUDA kernels and their wrapper
   // - cuda-decoder.cu for everything else
   //

   class CudaDecoder {

        public:
            typedef fst::StdArc StdArc;
            typedef StdArc::Weight StdWeight;
            typedef StdArc::Label Label;
            typedef StdArc::StateId StateId;
            typedef float CostType;
            typedef int32 LaneId;
            typedef int32 ChannelId;

            // IntegerCostType is the type used in the lookup table d_state_best_cost
            // and the d_cutoff
            // We use a 1:1 conversion between CostType <--> IntegerCostType
            // IntegerCostType is used because it triggers native atomic operations
            typedef int32 IntegerCostType;

            CudaDecoder(const CudaFst &fst, 
                        const CudaDecoderConfig &config,
                        int nlanes=1,
                        int nchannels=1);  
            ~CudaDecoder();

            // Decode this utterance.
            /// Returns true if any tokens reached the end of the file (regardless of
            /// whether they are in a final state); query ReachedFinal() after Decode()
            /// to see whether we reached a final state.
            bool Decode(DecodableInterface *decodable);
            
            /// InitDecoding initializes the decoding, and should only be used if you
            /// int32end to call AdvanceDecoding().  If you call Decode(), you don't need
            /// to call this.  You can call InitDecoding if you have already decoded an
            /// utterance and want to start with a new utterance. 
            void InitDecoding(); 
            
            /// This will decode until there are no more frames ready in the decodable
            /// object, but if max_num_frames is >= 0 it will decode no more than
            /// that many frames.  If it returns false, then no tokens are alive,
            /// which is a kind of error state.
            void AdvanceDecoding(DecodableInterface *decodable,
                    int32 max_num_frames = -1);

            /// Returns the number of frames already decoded.  
            int32 NumFramesDecoded() const { return num_frames_decoded_; }

            // ReachedFinal returns true if the last frame's token queue 
            // contains at least one token associated with a final state
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
           
            // GetBestCost sets in *min the token's best cost in the main_q
            // it also sets in *arg the index of that token (argmin)
            // is isfinal is true, we take into account the final costs
            void GetBestCost(bool isfinal, CostType *best_cost, int32 *best_cost_arg) const;

            /// FinalRelativeCost() serves the same function as ReachedFinal(), but gives
            /// more information.  It returns the difference between the best (final-cost plus
            /// cost) of any token on the final frame, and the best cost of any token
            /// on the final frame.  If it is infinity it means no final-states were present
            /// on the final frame.  It will usually be nonnegative.
            CostType FinalRelativeCost() const;


            //
            // Data structures used by the kernels
            //


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


            //
            // Used for the cutoff
            // cutoff = min_cost + beam
            // We store both separatly because we have an adaptive beam
            // We may change the beam after discovering min_cost
            // we need to keep track of min_cost to apply the new beam
            // (we don't know what the old beam was)
            //
            // Native float and Integers version
            //

            struct MinCostAndBeam {
                CostType min_cost;
                CostType beam;
            };

            struct MinCostAndBeamIntegers {
                IntegerCostType min_cost;
                IntegerCostType beam;
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
                int32 *h_main_q_end_before_finalize_nonemitting_kernel;
                int32 *h_main_q_narcs; 

                int32 *h_q_overflow; 
                int32 q_capacity;

                StateId *d_aux_q_state; 
                CostType *d_aux_q_cost;
                InfoToken *d_aux_q_info; 
                int32 *d_aux_q_end; 
                int32 *h_aux_q_end;

                int32 *d_main_q_degrees_prefix_sum; 
                uint32_t *d_arc_offsets; 
                int32 *d_main_q_arc_offsets; // offsets, relative to the queue

                IntegerCostType *d_state_best_cost; 
                MinCostAndBeamIntegers *d_global_min_cost_and_beam;

                int32 *d_main_q_degrees_block_sums_prefix_sum; 
                int32 *d_n_CTA_done;

                CostType infinite_cost;
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
                int32 *d_main_q_degrees_prefix_sum; 

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

                CostType *arc_weights; 
                StateId *arc_nextstates; 
                CostType *d_loglikelihoods;
                CostType default_beam;
                MinCostAndBeamIntegers *d_global_min_cost_and_beam;

                int32 *d_state_best_cost;
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
            // main_q_narcs_estimate is used to decide how many threads to launch
            // it is not used inside the kernel, where the exact value will be used
            //

            void ExpandArcs(bool is_emitting, int32 main_q_narcs_estimate);

            //
            // PreprocessAndContract kernel
            //
            // The Preprocess* kernels are used before executing Expand 
            // Computing data members needed by Expand (preprocess) and prune tokens on the fly (contract) 
            //
            // Pseudo code of the PreprocessAndContract kernel : 
            // - For each token in the aux_q, 
            //          compute bool is_best = (token.cost < cutoff) 
            //                                && (token.cost == d_state_best_costs[token.nextstate])
            // If is_best, then :
            //         1) append this token to the main_q
            //         2) compute out_degree = (# of outgoing arcs from token.nextstate) 
            //         3) compute the prefix sum of those out_degrees for all tokens appended in the main_q
            //         4) save that prefix sum in d_main_q_degrees_prefix_sum
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
            // aux_q_size_estimate is used to decide how many threads to launch
            // it is not used inside the kernel, where the exact value will be used
            //

            void PreprocessAndContract(int32 aux_q_size_estimate);

            
            //
            // PreprocessInPlace kernel
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
            //                                && (token.cost == d_state_best_costs[token.nextstate])
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
            //         3) save that prefix sum in d_main_q_degrees_prefix_sum
            //
            // After executing PreprocessInPlace,
            // The "active" main_q[local_offset, end[ range stays the same.
            //
            // Note : Only the first pass of the prefix sum is computed in that kernel. We then need to call
            // ResetStateBestCostLookupAndFinalizePreprocessInPlace after PreprocessInPlace 
            //
            // main_q_size_estimate is used to decide how many threads to launch
            // it is not used inside the kernel, where the exact value will be used
            //

            void PreprocessInPlace(int32 main_q_size_estimate);


            //
            // FinalizeProcessNonemitting
            // This kernel is called at the end of the ProcessNonEmitting computation
            // it is used when ProcessNonEmitting generate a small number of new tokens at each iteration 
            // to avoid calling heavy-lifting kernels such as ExpandArcs too many times, we instead use
            // FinalizeProcessNonemitting that uses only one CTA 
            // By using one CTA, we can sync all threads inside the kernel, and iterate until convergence 
            // without lauching new kernels
            // This meta-kernel performs :
            // while we have non-emitting arcs to traverse:
            //      (1) Preprocess and contract 
            //      (2) Expand
            // This meta-kernel does not call the PreprocessAndContract or Expand kernels
            // it uses simplified implementations (for one CTA) of those 
            //
            void FinalizeProcessNonemitting();


            // InitStateCost initializes all costs to +INF in d_state_best_cost at the beginning of the computation
            void InitStateBestCostLookup();

            //
            // This kernel contains both ResetStateCostLookup and FinalizePreprocess in place.
            //
            // ResetStateCostLookup :
            //
            // We need to reset d_state_best_cost between frames. We could use InitStateCost
            // but a large portion of the lookup table has not been used
            // ResetStateBestCostLookupAndFinalizePreprocessInPlace resets only the costs that are not +INF, using the d_main_q_state to do it
            // d_main_q_state contains the list of states that have been considered and have a best cost < +INF
            //
            // FinalizePreprocessInPlace :
            //
            // This kernel is responsible to compute the second pass of the
            // prefix sum. Must be called between PreprocessInPlace and ExpandArcs
            //
            //
            // main_q_size_estimate is used to decide how many threads to launch
            // it is not used inside the kernel, where the exact value will be used
            //

            void ResetStateBestCostLookupAndFinalizePreprocessInPlace(int main_q_size_estimate);

            // Pre-computes log likelihoods for the current frame 
            void ComputeLogLikelihoods(DecodableInterface *decodable);

            // CheckOverflow
            // If a kernel sets the flag h_q_overflow, we send a warning to stderr 
            void CheckOverflow();

            //
            // Debug functions
            // Called only if necessary
            // depends on the value of KALDI_CUDA_DECODER_DEBUG_LEVEL
            //
            void DebugAssertsNewFrame();
            void DebugAssertsBeforeExpand(bool is_emitting);

            //
            // Data members
            //
            // Pointers in h_* refer to data on the CPU memory
            // Pointers in d_* refer to data on the GPU memory

            // The CudaFst data structure contains the FST graph
            // in the CSR format
            const CudaFst fst_;

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

            //
            // Parameters used by a decoder lane
            // At the end of each frame, we know everything stored
            // by those parameters (lookup table, aux_q, etc.)
            // is back to its original state
            // We can reuse it for another frame/channel without doing anything
            //
            struct LaneParams {
                // Ordering struct members by size for memory alignment (64bits, then 32bits members)

                // aux_q, tokens split in 3 parts (cf main_q)
                StateId *d_aux_q_state_; 
                CostType *d_aux_q_cost_;
                InfoToken *d_aux_q_info_;

                // End index of the aux queue
                // only tokens at index i with i < aux_q_end 
                // are valid tokens
                int32 *h_aux_q_end_; // copy of the value in pinned memory

                // d_main_q_info_ is only needed as a buffer when creating the 
                // tokens. It is not needed by the next frame computation
                // We send it back to the host, and at the end of a frame's computation,
                // it can be used by another channel. That's why it's in 
                // "LaneParams", and not "ChannelParams"
                InfoToken *d_main_q_info_;

                // Depending on the value of the parameter "max_tokens_per_frame"
                // we can end up with an overflow when generating the tokens for a frame
                // We try to prevent this from happening using an adaptive beam
                // if an overflow is about to happen, the kernels revert all data
                // to the last valid state, and set that flag to true
                // Even if that flag is set, we can continue the execution (quality
                // of the output can be lowered)
                // We use that flag to display a warning to stderr
                int32 *h_q_overflow_;

                // d_state_best_cost[state] -> best cost for that state for the current frame
                // reset between frames
                // type int32 to be able to use native atomicMin instruction
                // we use a 1:1 conversion float <---> sortable int
                IntegerCostType *d_state_best_cost_;
            
                // When generating d_main_q_degrees_prefix_sum we may need to do it in three steps
                // (1) First generate the prefix sum inside each CUDA blocks
                // (2) then generate the prefix sum of the sums of each CUDA blocks
                // (3) Use (1) and (2) to generate the global prefix sum
                // Data from step 1 and 3 is stored in d_main_q_degrees_prefix_sum
                // Data from step 2 is stored in d_main_q_degrees_block_sums_prefix_sum
                // Note : this is only used by PreprocessInPlace
                // PreprocessAndContract uses a trick to compute the global prefix sum in one pass
                int32 *d_main_q_degrees_block_sums_prefix_sum_;


                // Some kernels need to perform some operations before exiting
                // n_CTA_done is a counter that we increment when a CTA (CUDA blocks)
                // is done
                // Each CTA then tests the value for n_CTA_done to detect if it's the last to exit
                // If that's the cast, it does what it has to do, and sets n_CTA_done back to 0
                int32 n_CTA_done_;
                int32 aux_q_end_;
            };

            // 
            // Parameters used by a decoder channel
            // Their job is to save the state of the decoding 
            // channel between frames
            //
            struct ChannelParams {
                // Ordering struct members by size for memory alignment (64bits, then 32bits members)

                // Parts of the main_q that we need to compute next frame
                // d_main_q_info was sent back to the host, and we don't need
                // it to compute next frame
                StateId *d_main_q_state_;
                CostType *d_main_q_cost_;

                // ExpandArcs does not use at its input the complete main queue
                // It only reads from the index range [main_q_local_offset, end[
                int32 *h_main_q_local_offset_; 

                // Contains both main_q_end and narcs
                // The pointers refer to the same location than the d_main_q_end and d_main_q_narcs pointers,
                // ie : 
                // main_q_end = &d_main_q_end_and_narcs->split.ntokens
                // main_q_narcs = &d_main_q_end_and_narcs->split.narcs
                // End index of the main queue
                // only tokens at index i with i < main_q_end 
                // are valid tokens
                // Each valid token the subqueue main_q[main_q_offset, main_q_end[ has 
                // a number of outgoing arcs (out-degree)
                // main_q_narcs is the sum of those numbers 
                //
                // We sometime need to update both end and narcs at the same time,
                // using an 64 bits atomic, which is why we use an union
                TokenAndArcCountUnion main_q_end_and_narcs_i2_; 

                // Equivalent in pinned memory
                int32 *h_main_q_end_;
                int32 *h_main_q_narcs;

                // After each frame, we copy the main queue (GPU memory)
                // to the end of h_all_tokens_info (CPU memory)
                // We only move to the CPU what's needed for the final backtrack in GetBestPath
                // ie { prev_token, arc_idx } (= struct InfoToken)
                InfoTokenVector h_all_tokens_info_; 

                // The load balancing of the Expand kernel relies on the prefix sum of the degrees 
                // of the state in the queue (more info in the ExpandKernel implementation) 
                // That array contains that prefix sum. It is set by the "Preprocess*" kernels
                // and used by the Expand kernel
                int32 *d_main_q_degrees_prefix_sum_;

                // d_main_q_arc_offsets[i] = fst_.arc_offsets[d_main_q_state[i]]
                // we pay the price for the random memory accesses of fst_.arc_offsets in the preprocess kernel
                // we cache the results in d_main_q_arc_offsets which will be read in a coalesced fashion in expand
                int32 *d_main_q_arc_offsets_;


                // loglikelihoods computed by the acoustic model
                // we need it to compute the cost of emitting edges
                CostType *d_loglikelihoods_;

                // Cutoff for the current frame
                // Contains both the global min cost (min cost for that frame)
                // And the current beam
                // We use an adaptive beam, so the beam might change during computation
                MinCostAndBeamIntegers global_min_cost_and_beam_;

                // The token at index i in the main queue has in reality 
                // a global index of (i + main_q_global_offset)
                // This global index is unique and takes into account that 
                // we've flushed the main_q back to the host. We need unique indexes 
                // for each token in order to have valid token.prev_token data members
                // and be able to backtrack at the end
                int32 main_q_global_offset_;

                // Equivalent to the corresponding h_* 
                // members (cf above), but directly in memory
                int32 main_q_local_offset_;

            };

            // In AdvanceDecoding,
            // the lane lane_id will compute the channel
            // with channel_id = channel_to_compute[lane_id]
            struct GlobalParams {
                ChannelId channel_to_compute[KALDI_CUDA_DECODER_MAX_N_LANES];
                int32 nchannels_to_compute;

                int32 q_capacity;
                CostType infinite_cost;
                expand_params_.arc_ilabels = fst_.d_arc_ilabels_;
                expand_params_.arc_weights = fst_.d_arc_weights_; 
                expand_params_.arc_nextstates = fst_.d_arc_nextstates_; 
                expand_params_.default_beam = default_beam_;
            };

            GlobalParams global_params_;

            LaneParams *h_lane_params, *d_lane_params_; 
            int32 nlanes_;
            ChannelParams *h_lane_params, *d_channel_params_; 
            int32 nchannels_;

            // is_channel_busy[i] <=> channel i is currently 
            // being used by a channel lane
            std::bitset is_channel_busy;

            // CUDA streams
            // kernels are launched in compute_st
            // copies in copy_st
            // we use two streams to overlap copies and kernels
            // we synchronize the two using events
            cudaStream_t compute_st_, copy_st_;

            // CUDA events

            // We need to synchronize the streams copy_st and compute_st
            // because of data dependency : they both have to read or write to the main_q 
            // when we're done copying the old main_q to the CPU, we trigger can_write_to_main_q 
            cudaEvent_t can_write_to_main_q_;

            // At the end of Preprocess kernels we set h_main_q_narcs (pinned memory)
            // this event is set in the pipeline after Preprocess kernels to inform that data is ready to be read
            cudaEvent_t can_read_h_main_q_narcs_;

            //
            // This kernel is triggered when finalize non emitting is about to start
            //
            cudaEvent_t before_finalize_nonemitting_kernel_;

            // h_main_q_end is final for this frame
            // triggered at the end of a frame computation
            cudaEvent_t can_read_final_h_main_q_end_;

            // Buffers for copies on host on the current main_q
            // Those are only buffers - and must be considered as containing 
            // uninitialized data
            // If you need to read from those,
            // please explicitly copy data from device first !
            // We use them in "ReachedFinal" for instance
            StateId *h_main_q_state_; 
            CostType *h_main_q_cost_; 

            // When we generate a new tokens list we only keep candidates 
            // that have a cost < best_cost_in_the_queue + beam
            // At first beam = default_beam_
            // We may decrease that beam if we are generating too many tokens
            // (adaptive beam)
            CostType default_beam_;

            int32 max_tokens_;
            int32 max_tokens_per_frame_;

            // Keep track of the number of frames decoded in the current file.
            int32 num_frames_decoded_;

            // Max possible cost
            // used to init the min_cost and d_state_best_cost_
            CostType infinite_cost_;

            // Pointers directly returned by cudaMalloc in constructor
            // We used those pointers to set the others
            // We need to call cudaFree on those in the destructor
            // Those pointers are only used for that - not during computation
            StateId *d_all_main_q_state_; 
            CostType *d_all_main_q_cost_;
            InfoToken *d_all_main_q_info_;

            StateId *d_all_aux_q_state_; 
            CostType *d_all_aux_q_cost_;
            InfoToken *d_all_aux_q_info_;


            int32 *d_all_q_degrees_prefix_sum_;
            int32 *d_all_q_degrees_block_sums_prefix_sum_;
            int32 *d_all_q_arc_offsets_;
            int32 *d_all_loglikelihoods_;
            int32 *h_all_pinned_ints;
            IntegerCostType *d_all_state_best_cost_;

            // Used for debugging purposes
            // only malloc'ed if necessary
            int32 *h_debug_buf1_, *h_debug_buf2_;

            KALDI_DISALLOW_COPY_AND_ASSIGN(CudaDecoder);
    };

    

} // end namespace kaldi.


#endif
