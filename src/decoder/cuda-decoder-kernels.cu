// decoder/cuda-decoder-kernels.cu

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

#include <cub/cub.cuh>
#include "decoder/cuda-decoder.h"

#define KALDI_CUDA_DECODER_DIV_ROUND_UP(a,b) ((a+b-1)/b)

namespace kaldi {

typedef CudaDecoder::StateId StateId;
typedef CudaDecoder::TokenAndArcCount TokenAndArcCount;
typedef CudaDecoder::TokenAndArcCountUnion TokenAndArcCountUnion;
typedef CudaDecoder::CostType CostType;
typedef CudaDecoder::MinCostAndBeamIntegers MinCostAndBeamIntegers;
typedef CudaDecoder::MinCostAndBeam MinCostAndBeam;
typedef CudaDecoder::IntegerCostType IntegerCostType;
typedef CudaDecoder::PreprocessParams PreprocessParams; 
typedef CudaDecoder::ExpandArcParams ExpandArcParams; 

    //
    // Utils device function
    //


    //
    // 1:1 Conversion float <---> sortable int
    // We convert floats to sortable ints in order
    // to use native atomics operation, which are 
    // way faster than looping over atomicCAS 
    //

    __device__ int32 floatToOrderedInt(float floatVal) {
        int32 intVal = __float_as_int( floatVal );

        return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
    }



    __device__ float orderedIntToFloat(int32 intVal) {
        return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF );
    } 

    //
    // Kernels
    //

    // For description of what each kernel is doing, please refer to cuda-decoder.h
    // and look for the corresponding wrapper
    // for instance, for a description of _init_lookup_kernel,
    // look for the description of CudaDecoder::InitStateBestCostLookup() in cuda-decoder.h
    // Also resets the cutoff to +inf

    // Used before first frame
    __global__ void _init_state_best_cost_lookup_kernel(int32 state_best_cost_size,
                                                        CostType default_beam,
                                                        CostType infinite_cost,
                                                        IntegerCostType *state_best_cost, 
                                                        MinCostAndBeamIntegers *d_global_min_cost_and_beam) {
        // One thread resets the cutoff
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            d_global_min_cost_and_beam->min_cost = floatToOrderedInt(infinite_cost);
            d_global_min_cost_and_beam->beam = floatToOrderedInt(default_beam);
        }
        // Reset lookup table
        for(int32 idx = blockIdx.x*blockDim.x + threadIdx.x;
                idx < state_best_cost_size;
                idx += blockDim.x*gridDim.x) {
            state_best_cost[idx]  = floatToOrderedInt(infinite_cost);
        }
    }

    void CudaDecoder::InitStateBestCostLookup() {
        int32 nstates = fst_.NumStates();
        KALDI_ASSERT(nstates > 0);

        dim3 grid,block;
        block.x = KALDI_CUDA_DECODER_KERNEL_INIT_LOOKUP_DIMX;
        grid.x = KALDI_CUDA_DECODER_DIV_ROUND_UP(nstates, block.x);

        _init_state_best_cost_lookup_kernel<<<grid,block,0,compute_st_>>>(nstates, 
                                                                          default_beam_, 
                                                                          infinite_cost_,
                                                                          d_state_best_cost_,
                                                                          d_global_min_cost_and_beam_);
    }

        
     /*
     
     ResetStateBestCostLookupAndFinalizePreprocessInPlace
     Does two things :
     (1) Reset the lookup between frames :
         Using the queue to reset only the values needed
         Also takes care of resetting cutoff
    
     (2) Finalize PreprocessInPlace :
       Part 2 of the prefix sum for "PreprocessInPlace" 
       
       For PreprocessAndContract we were able to do the global prefix sum of degrees in one pass, so we should not call
       this kernel

       Our final goal is to have the prefix sum of the degrees of the token's state of the main_q
       and store that prefix sum in d_main_q_degrees_prefix_sum

       In PreprocessInPlace we've computed two things :
       
       - "local prefix sums" of the degree. Each CUDA block has computed the local prefix sum of its degrees. We've
       stored each of the local prefix sums in d_main_q_degrees_prefix_sum
       - the prefix sum of the local sums (local sum = sum of all degrees in a CUDA block). This gives us the offset
       to add to each local prefix sum to end up with a global prefix sum

       Note : If we want to speed up expand, we can compute lower and upper bound to restrain 
       the binary search in expand
       This can be done on the fly here, and removes main bottleneck of expand

     */

    __global__ void _reset_state_cost_lookup_kernel_and_finalize_preprocess_in_place(const StateId *d_main_q_state_, 
                                                    const int32 *d_main_q_end_, 
                                                    const int32 *d_local_sums_prefix_sum,  
                                                    int32 default_beam,
                                                    CostType infinite_cost,
                                                    MinCostAndBeamIntegers *d_global_min_cost_and_beam,
                                                    IntegerCostType *d_state_best_cost, 
                                                    int32 *d_main_q_degrees_prefix_sum) {
        // One thread resets the cutoff
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            d_global_min_cost_and_beam->min_cost = floatToOrderedInt(infinite_cost); 
            // Resetting the beam back to default between frames
            d_global_min_cost_and_beam->beam = floatToOrderedInt(default_beam); 
        }
        int32 main_q_end = *d_main_q_end_; 

        for(int32 main_q_idx = blockIdx.x*blockDim.x + threadIdx.x;
                main_q_idx < main_q_end;
                main_q_idx += blockDim.x*gridDim.x) {
            // d_main_q_state_ contains the list of states that we've considered in the last frame
            // it corresponds to the list of indexes i such as d_state_best_cost[i] < +INF
            // faster than init_state_cost_lookup_kernel by a factor of ~10
            StateId state = d_main_q_state_[main_q_idx];


            int32 local_sum_idx = main_q_idx / KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX;
            int32 local_sum_offset = d_local_sums_prefix_sum[local_sum_idx]; // we rely on the caches for this one

            d_state_best_cost[state]  = floatToOrderedInt(infinite_cost);
            d_main_q_degrees_prefix_sum[main_q_idx] += local_sum_offset;
        }
    }

    void CudaDecoder::ResetStateBestCostLookupAndFinalizePreprocessInPlace(int32 main_q_size_estimate) {
        KALDI_ASSERT(main_q_size_estimate > 0);

        dim3 grid,block;
        block.x = KALDI_CUDA_DECODER_KERNEL_INIT_LOOKUP_DIMX;
        grid.x = KALDI_CUDA_DECODER_DIV_ROUND_UP(main_q_size_estimate, block.x);

        _reset_state_cost_lookup_kernel_and_finalize_preprocess_in_place<<<grid,block,0,compute_st_>>>(d_main_q_state_, 
                                                                      d_main_q_end_,
                                                                      d_main_q_degrees_block_sums_prefix_sum_,   
                                                                      default_beam_,
                                                                      infinite_cost_,
                                                                      d_global_min_cost_and_beam_,
                                                                      d_state_best_cost_, 
                                                                      d_main_q_degrees_prefix_sum_);
    }


    // Sum operator for the TokenAndArcCount struct (2 ints) 
    // Used in preprocess_and_contract
    struct TokenAndArcCountSum {
        __device__ TokenAndArcCount operator()(const TokenAndArcCount &a, const TokenAndArcCount &b) const {
            TokenAndArcCount c;
            c.ntokens = a.ntokens + b.ntokens;
            c.narcs = a.narcs + b.narcs;

            return c;
        }
    };

    /*
       This kernel preprocess the necessary information for expand (scan of the outgoing degrees) 
       and explicitly prune the tokens

       The ExpandArc kernel writes the new raw token list in the aux_q. However, the cutoff 
       was progressively lowered during the computation, and some tokens now have a cost > cutoff.
       During the contract stage of this kernel, we remove such tokens. 
       We also remove duplicates, i.e. tokens pointing to the same state, but with token.cost > best_cost_for_that_state

       It contracts (by pruning) the queue list:
       raw output in aux_q ----contract----> pruned output in main q

       This kernel is responsible for :

       1) Read a token from the aux queue (raw output from previous expand)

       2) Compute the outgoing degree of that token.next_state. For that :
       -> If that token is suboptimal (cutoff, best_cost), we prune it
       -> Otherwise, we will move it to the main_q. We also read its arc degree in the FST graph 

       3) We move the non-pruned tokens into the main q. After a local prefix sum,
       we request a spot in the main_q for those tokens using the main_q_end_and_narcs counter. 
       main_q_end_and_narcs.split.end contains the number of tokens in the main q until now
       main_q_end_and_narcs.split.narcs contains the number of arcs in the main q until now

       We also compute the degrees prefix sum in one pass using the main_q_end_and_narcs.split.narcs

       This kernel is used before ProcessNonEmitting
    */

    // Important : pass the struct PreprocessParams by copy - passing it using a ref will not work (CPU -> GPU)
    __global__ void _preprocess_and_contract_kernel(PreprocessParams params) {
        // Prefix sum operator
        typedef cub::BlockScan<TokenAndArcCount, KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX> BlockScan;
        __shared__ typename BlockScan::TempStorage sh_temp_storage;

        // This CUDA block (CTA) will count the number of tokens it has to move to the main_q
        // and store the result in sh_nsurvival_tokens_in_CTA
        __shared__ int32 sh_nsurvival_tokens_in_CTA;

        // We need to move the survival tokens to the main_q
        // 
        // sh_main_q_global_block_offset has two purposes :
        // (1) to know where to store the survival tokens in the main_q
        // (2) to perform the prefix sum degrees of the survival degrees
        //
        // The reason why we store those two values together is because they are linked (see below)
        //
        // (1) We need a spot to store those tokens in the main_q 
        // We will ask the main_q counter where to store those tokens, the answer will be 
        // an offset of the main_q. We will store our tokens in positions :
        // d_main_q_state[sh_main_q_global_block_offset.ntokens], d_main_q_state[sh_main_q_global_block_offset.ntokens+1]...
        //
        // (2) sh_main_q_global_block_offset.narcs contains the number of arcs in the main_q up until index sh_main_q_global_block_offset.ntokens
        // ie the number of arcs going out of all states in d_main_q_state[0..sh_main_q_global_block_offset.ntokens]
        // it is used to compute the global prefix sum of degrees in one pass
        //
        __shared__ TokenAndArcCountUnion sh_main_q_global_block_offset;

        // This CTA will start working in aux_q on indexes >= first_block_offset
        // If we have nothing to do, stop here
        int32 first_block_offset = blockDim.x*blockIdx.x;
        const int32 aux_q_end = *params.d_aux_q_end;

        // Early exit if all threads in the CTA have nothing to do
        // it does break the finalize_kernel: code below, 
        // we take those early exists into account
        if(first_block_offset >= aux_q_end)
            return;

        // Final cutoff from last ExpandArc execution
        MinCostAndBeamIntegers global_min_cost_and_beam = *params.d_global_min_cost_and_beam;
        const CostType cutoff = orderedIntToFloat(global_min_cost_and_beam.min_cost)
                               + orderedIntToFloat(global_min_cost_and_beam.beam);

        // The condition of the for loop is the same for all threads in the CUDA block
        // we want to keep all threads alive at the same time for now
        // otherwise __syncthreads() would fail
        for(int32 block_offset = first_block_offset;
                block_offset < aux_q_end;
                block_offset += gridDim.x*blockDim.x) {

            int32 aux_q_idx = block_offset + threadIdx.x;
            int32 degree = 0;
            int32 arc_start = -1;

            StateId token_state;
            CostType token_cost;

            // if aux_q_idx is a valid index in the main_q
            if(aux_q_idx < aux_q_end) {
                // Cost and state associated with the token
                token_cost = params.d_aux_q_cost[aux_q_idx];
                token_state = params.d_aux_q_state[aux_q_idx];

                // Best cost for that token_state
                // We know we have a token associated with token_state in the queue with the cost state_best_cost
                CostType state_best_cost = orderedIntToFloat(params.d_state_best_cost[token_state]);

                // Cutoff may have decreased since the creation of the token
                if(token_cost < cutoff) {
                    
                    // We can have duplicates, ie token associated with the same states
                    // If this token is not the best candidate, get rid of it
                    if(token_cost == state_best_cost) {
                        arc_start = params.d_arc_offsets[token_state];
                        int32 arc_end = params.d_arc_offsets[token_state+1];
                        degree = arc_end - arc_start;
                    }
                }

                // the d_state_best_cost lookup table is reset to +INF for all states between frame
                // for perf. reason we only reset states that are in d_main_q_state
                // however if state_best_cost >= cutoff, all tokens associated with token_state 
                // will be pruned, and that state will not be in d_main_q_state
                // we need to reset the lookup table now

                if (state_best_cost >= cutoff)
                    params.d_state_best_cost[token_state] = floatToOrderedInt(params.infinite_cost);

            }

            int32 is_pruned = (arc_start == -1);


            TokenAndArcCount block_prefix_sum_token_arc_count;

            // We now know which tokens will be moved to the main_q, the remaining will be pruned
            // we now compute a prefix sum inside the CUDA block to determine the local indexes of the survival tokens
            // the first survival token will have a index of 0, the second 1, ...
            block_prefix_sum_token_arc_count.ntokens =  is_pruned ? 0 : 1;
            
            // We also need to compute the prefix sum of the degrees
            // we start by doing a local prefix sum inside the CUDA block
            block_prefix_sum_token_arc_count.narcs =  degree;

            TokenAndArcCount zero_struct;
            zero_struct.ntokens = zero_struct.narcs = 0;

            // Computing the prefix sum (exclusive)
            BlockScan(sh_temp_storage).ExclusiveScan(block_prefix_sum_token_arc_count, 
                                                    block_prefix_sum_token_arc_count, 
                                                    zero_struct,
                                                    TokenAndArcCountSum());

            
            TokenAndArcCountUnion token_and_arc_count_block_sum;
            if(threadIdx.x == (KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX-1)) {
                // This conditional branch is entered by the last thread
                // because it is the last, the prefix_sum of that thread contains the sum of all elts

                // We also add the value from this thread - the prefix sum is exclusive
                token_and_arc_count_block_sum.split.ntokens = block_prefix_sum_token_arc_count.ntokens + (is_pruned ? 0 : 1);
                token_and_arc_count_block_sum.split.narcs = block_prefix_sum_token_arc_count.narcs + degree;

                sh_nsurvival_tokens_in_CTA = token_and_arc_count_block_sum.split.ntokens;
                
                // Doing two things at the same time :
                // requesting a spot in the main_q to store the survival tokens from this CTA 
                // (we need space for token_and_arc_count_block_sum.split.ntokens tokens)
                // informing the main_q that our survival tokens contain token_arc_count_block_sum.split.narcs arcs
                //
                // We then store the return value, which is the global offset on where to store those tokens,
                // and the total number of arcs up until that global offset
                sh_main_q_global_block_offset.both = atomicAdd(&params.d_main_q_end_and_narcs_i2->both, token_and_arc_count_block_sum.both);
            }

            // Syncing for three reasons :
            // - Broadcasting sh_main_q_global_block_offset
            // - Broadcasting sh_nsurvival_tokens_in_CTA
            // - We may reuse sh_temp_storage (cf CUB doc)
            __syncthreads(); 

            // Checking if we are overflowing the main_q
            if((sh_main_q_global_block_offset.split.ntokens + sh_nsurvival_tokens_in_CTA) >= params.q_capacity) {
                if(threadIdx.x == (KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX-1)) {
                    // We are overflowing the main_q
                    // We first revert what this CTA has done, ie revert the previous atomicAdd
                    // because all CTAs will revert, we know we will have a valid state after completion of this kernel
                    atomicAdd(&params.d_main_q_end_and_narcs_i2->both, -token_and_arc_count_block_sum.both); // revert

                    // Setting the flag. It will print a warning to stderr
                    *params.h_q_overflow = 1;
                }

                // We abort computation, we no longer have space in the main_q.
                // We still jump to finalize_kernel, to do what's needed before completion
                goto finalize_kernel;
            }

            // If we are executing the following lines it means that we are not overflowing the queue
            // We then continue what we were doing

            if(!is_pruned) {
                // This thread is in charge of a survival token
                // we will move it to the main_q, at index main_q_idx

                // Note : we could remove the branch divergence here 

                int32 main_q_idx = sh_main_q_global_block_offset.split.ntokens + block_prefix_sum_token_arc_count.ntokens;

                InfoToken token_info = params.d_aux_q_info[aux_q_idx];

                // Moving the token to the main q
                params.d_main_q_state[main_q_idx] = token_state;
                params.d_main_q_cost[main_q_idx] = token_cost;
                params.d_main_q_info[main_q_idx] = token_info;

                // Saving the global prefix sum
                // = (narcs until now in the main queue) + (narcs until this thread in the CTA)
                params.d_main_q_degrees_prefix_sum[main_q_idx] = sh_main_q_global_block_offset.split.narcs 
                                                                 + block_prefix_sum_token_arc_count.narcs;
                // Saving the CSR arc offset for that token's state
                // it will be used by the expand kernel, and avoid doing a new random memory access in the expand kernel
                params.d_main_q_arc_offsets[main_q_idx] = arc_start;
            }
        }

        finalize_kernel:

        // Avoiding races 
        // We will write d_aux_q_end
        // And some threads may be still reading it 
        // At the beg of this kernel
        __syncthreads();
        
        if(threadIdx.x == 0) {
            // Declaring the CTA as done
            int32 old = atomicAdd(params.d_n_CTA_done, 1);
            // Taking into account the CTAs that have early exited at the begining of this kernel
            int32 total_active_CTAs = min(KALDI_CUDA_DECODER_DIV_ROUND_UP(aux_q_end, KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX), gridDim.x);
            // If we're the last CTA to exit, detect it
            bool is_last_CTA = (old == (total_active_CTAs -1));

            if(is_last_CTA) {
                __threadfence();

                // We added things to the main_q
                // d_main_q_end was modified
                // we update h_main_q_end to keep it consistent
                // the h_* pointers are in the pinned host memory, we can access them from the device

                int main_q_end = *params.d_main_q_end;
                int main_q_narcs = *params.d_main_q_narcs;
                *params.h_main_q_end = main_q_end;
                *params.h_main_q_end_before_finalize_nonemitting_kernel = main_q_end;
                *params.h_main_q_narcs = main_q_narcs;

                params.d_main_q_degrees_prefix_sum[main_q_end] = main_q_narcs;
                
                // We moved what we had to move from the aux q to the main q
                // We now empty the aux q 
                *params.d_aux_q_end = 0;
                *params.h_aux_q_end = 0; 

                // Reset the counter for next time
                *params.d_n_CTA_done = 0;
            }
        }

    }


    void CudaDecoder::PreprocessAndContract(int32 aux_q_size_estimate) {
        dim3 grid,block;

        KALDI_ASSERT(aux_q_size_estimate > 0);

        block.x = KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX;
        grid.x = KALDI_CUDA_DECODER_DIV_ROUND_UP(aux_q_size_estimate, block.x);

        // PreprocessAndContract is called when non emitting
        // using non emitting offsets
        preprocess_params_.d_arc_offsets = fst_.d_ne_offsets_;
        _preprocess_and_contract_kernel<<<grid,block,0,compute_st_>>>(preprocess_params_);
    }



/*
    PreprocessInPlace
    This kernel is also a preprocessing kernel, but this time does it in place
    ie it will not move tokens from the aux_q to the main_q
    It will do the preprocess operation directly on the main_q
    The tokens are already in the main q (they were placed here by a previous "contract and preprocess").

    We cannot prune non-optimal tokens, because the tokens are already in the main_q (we cannot prune 
    the main_q - it would break the prev_token indexes). To avoid doing unnecessary computation 
    in the expand kernel, we simulate the pruning by setting non-optimal token's degree to 0
    We then rely on the 1 thread = 1 arc exact load balacing of expand to ignore that token

    Please note that even if 0 threads will perform work on an ignored token in expand (degree = 0),
    it is not exactly the same as pruning it : the main_q accesses will not be perfectly coalesced
    in expand, because some "dead" tokens exist between living ones

    For the preprocess stage we have to compute the prefix sum of the tokens arc degrees
    Here we have to do the prefix sum in two passes : first local prefix sums inside CUDA block,
    then in a second kernel (finalize_preprocess_in_place), we add the necessary block offsets to end up 
    with the global prefix sum

    This preprocess step is used in ProcessEmitting. Tokens were placed in main_q by
    the ProcessNonEmitting of the previous frame. We cannot renumber them (it would break
    the prev_token index). We preprocess in place, leaving things as they are in main_q

*/

    __global__ void _preprocess_in_place_kernel(PreprocessParams params) {
        // Operator for the prefix sum inside the CUDA block
        typedef cub::BlockScan<int32, KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX> BlockScan;
        __shared__ typename BlockScan::TempStorage sh_temp_storage;


        // All threads in the last CUDA block (CTA) alive will have work to do at the end
        // this bool will be needed to broadcast the information from thread0 to all threads in the last CTA 
        __shared__ bool sh_is_last_CTA;

        const int32 main_q_end = *params.d_main_q_end;
        const int32 main_q_size = main_q_end - 0; // main_q_offset == 0 in that kernel

        // Final cutoff from last ExpandArc execution
        // The cutoff can have decreased since moving tokens to the main_q
        // min_cost cannot be lower than before (we only did non-emitting phases since then)
        // but the adaptive beam may have lowered the beam
        MinCostAndBeamIntegers global_min_cost_and_beam = *params.d_global_min_cost_and_beam;
        const CostType cutoff = orderedIntToFloat(global_min_cost_and_beam.min_cost)
                               + orderedIntToFloat(global_min_cost_and_beam.beam);

        // The condition of the for loop is the same for all threads in the CUDA block
        // we want to keep all threads alive at the same time for now
        // otherwise __syncthreads() would fail
        for(int32 block_offset = blockDim.x*blockIdx.x;
                block_offset < main_q_size;
                block_offset += gridDim.x*blockDim.x) {

            // Position of considered token in the main_q
            int32 main_q_idx = block_offset + threadIdx.x; 

            // Total number of arcs from that token's state
            int32 degree = 0; 

            if(main_q_idx < main_q_end) {
                StateId token_state = params.d_main_q_state[main_q_idx]; 
                CostType token_cost = params.d_main_q_cost[main_q_idx];

                // the cutoff may have decreased since the creation of that token
                if(token_cost < cutoff) {

                    // Best cost for that token_state
                    // We know we have a token associated with token_state in the queue with the cost state_best_cost
                    CostType state_best_cost = orderedIntToFloat(params.d_state_best_cost[token_state]); 
                    
                    // We can have duplicates, ie token associated with the same states
                    // If this token is not the best candidate, get rid of it
                    if(token_cost == state_best_cost) {
                        int32 start = params.d_arc_offsets[token_state]; 
                        int32 end = params.d_arc_offsets[token_state+1]; 
                        degree  = end - start;
                        
                        // Saving the start offset for the expand kernel
                        // avoid a new random memory access
                        params.d_main_q_arc_offsets[main_q_idx] = start;
                    }
                }
            }

            int32 degree_local_prefix_sum;

            // Computing a local prefix sum inside that CUDA block
            // A second kernel will take care of adding the necessary offset to those local prefix sums
            BlockScan(sh_temp_storage).ExclusiveSum(degree, degree_local_prefix_sum);

            if(main_q_idx < main_q_end) {
                // This is not the final global prefix sum
                // A second kernel will add the necessary offset
                params.d_main_q_degrees_prefix_sum[main_q_idx] = degree_local_prefix_sum; 
            }

            if(threadIdx.x == (KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX-1)) {
                // Saving the local sum of degrees of that CUDA block
                // That's necessary to compute the global offset of that CUDA block,
                // and that offset is what we need to transform the local prefix sum into a global prefix sum

                int local_sum_index = block_offset/KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX;
                int local_sum = degree_local_prefix_sum + degree; // the prefix sum was exclusive, adding missing value
                params.d_main_q_degrees_block_sums_prefix_sum[local_sum_index] = local_sum; 
            }


            // Synchronization for two reasons :
            // - we may need to reuse sh_temp_storage if the for loop iterates (cf CUB's doc)
            // - we need all threads to be done before considering the CTA as done (see below)
            __syncthreads(); 

        }

        //
        // The last CUDA block alive will compute the prefix sum of the block degrees sum
        // We need that prefix sum, because it represents the offsets that each CUDA block has in the global prefix sum
        // we will then add those offsets in finalize_preprocess_in_place

        if(threadIdx.x == 0) {
            // We indicate that this CTA is done
            int32 old = atomicAdd(params.d_n_CTA_done, 1); 
            
            // If we're the last CTA to exit, detect it
            sh_is_last_CTA = (old == (gridDim.x -1));
        }

        // Synchronization for two reasons :
        // - Broadcasting sh_is_last_CTA
        // - reusing sh_temp_storage (cf CUB's doc)
        __syncthreads();
        
        if(sh_is_last_CTA)
        {
            //
            // Our goal here is to compute the prefix sum of the previous local sums
            // What we call local sum is what contains the local_sum variables in the previous lines
            // it is the sum of degrees inside a given CUDA block, at a given for loop iteration
            // all local sums are stored in params.d_main_q_degrees_block_sums_prefix_sum
            // we want to do the prefix sum of that array
            //
            // Once this is done, params.d_main_q_degrees_block_sums_prefix_sum[i] will contain the 
            // offset that we need to add to the local prefix sum #i to convert it to a global
            // prefix sum
            // Right now we are only computing the offsets ; adding them to the local prefix sums will be 
            // done in FinalizePreprocessInPlace
            //

            //
            // We are the last CTA alive
            // which means that all local sums have been written to params.d_main_q_degrees_block_sums_prefix_sum
            // We can now do the prefix sum of that array   
            //

            // Making sure that we see changes from other CTAs 
            __threadfence();

            //
            // How many local sums values do we have ?
            // Please note that this number can be different from gridDim.x
            // We may have applied a upper limit on gridDim.x, and in that case
            // gridDim.x < number_of_local_sums
            //

            int32 number_of_local_sums = KALDI_CUDA_DECODER_DIV_ROUND_UP(main_q_size, KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX);

            // We may iterate the following for loop multiple times
            // on iteration > 0, we will have to consider the offset from previous iterations
            int32 prefix_sum_of_local_sums_offset = 0;

            // local_sum_index is an index in the array d_main_q_degrees_block_prefix
            // 
            // The condition inside the loop is common to all threads in the CTA
            // we want to keep all threads active, we will use syncthreads()
            for(int32 local_sum_index_offset = 0; 
                      local_sum_index_offset < number_of_local_sums; 
                      local_sum_index_offset += blockDim.x) {

                int32 local_sum_index = local_sum_index_offset + threadIdx.x; 

                int32 local_sum = (local_sum_index < number_of_local_sums) 
                                ? params.d_main_q_degrees_block_sums_prefix_sum[local_sum_index] 
                                : 0; // neutral element

                int32 prefix_sum_of_local_sums, total_sum_of_local_sums_for_this_iteration;

                BlockScan(sh_temp_storage).ExclusiveSum(local_sum, prefix_sum_of_local_sums, total_sum_of_local_sums_for_this_iteration);

                prefix_sum_of_local_sums += prefix_sum_of_local_sums_offset;
                prefix_sum_of_local_sums_offset += total_sum_of_local_sums_for_this_iteration;

                if(local_sum_index < number_of_local_sums) {
                    params.d_main_q_degrees_block_sums_prefix_sum[local_sum_index] = prefix_sum_of_local_sums;
                }

                // Sync'ing to be able to reuse sh_temp_storage (cf CUB's doc)
                __syncthreads();
            }

            if(threadIdx.x == 0)
            {
                // Final offset is the overall total
                int total_sum_of_local_sums = prefix_sum_of_local_sums_offset;
                int main_q_narcs = total_sum_of_local_sums;

                *params.d_main_q_narcs = main_q_narcs; 
                // h_main_q_narcs is in pinned memory, we can write to it from the device
                *params.h_main_q_narcs = main_q_narcs; 

                params.d_main_q_degrees_prefix_sum[main_q_end] = main_q_narcs;

                // reset for next time
                *params.d_n_CTA_done = 0;
            }
        }
    }


    void CudaDecoder::PreprocessInPlace(int32 main_q_size_estimate) {
        dim3 grid,block;
        block.x = KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX;

        KALDI_ASSERT(main_q_size_estimate > 0);
        grid.x = KALDI_CUDA_DECODER_DIV_ROUND_UP(main_q_size_estimate, block.x);

        // PreprocessInPlace is called when emitting
        // using emitting arc offsets
        preprocess_params_.d_arc_offsets = fst_.d_e_offsets_;
        _preprocess_in_place_kernel<<<grid,block,0,compute_st_>>>(preprocess_params_);
    }



   //
   // Helper functions/data structure for the ExpandArc kernel
   //


   // We'll need to do a BlockScan on both an int and a cost
   // data struct and its associated binary operation

    struct CostTypeAndInt {
        CostType cost;
        int32 i;
    };

    // 
    // We'll use the same BlockScan to compute two things :
    //     1) The prefix sum of indexes
    //     1) The minimum cost overall all costs in the CUDA Block 
    //
    // We use a + for the prefix sum, and a min for the min
    //

    struct MinCostPlusInt {
        __device__ CostTypeAndInt operator()(const CostTypeAndInt &a, const CostTypeAndInt &b) const {
            CostTypeAndInt c;
            c.cost = fmin(a.cost, b.cost);
            c.i = a.i + b.i;
            return c;
        }
    };

    //
    // GetAdaptiveBeam is used by ExpandArc and FinalizeProcessNonemitting
    //
    // Given the fact that the token queues are too small to store 
    // all possible tokens in the worst case scenario (where we could generate "nstates" tokens),
    // we need to tighten the beam if we notice that we are at risk of overflowing either the aux_q
    // or the main_q
    //

    __device__ __forceinline__ CostType GetAdaptiveBeam(const CostType default_beam,
            const int32 q_size,
            const int32 q_capacity) {

        // Doing something simple for now
        // We have to keep beam large enough,
        // the final cutoff will be used for the final
        // prune. If it is too small, we won't keep enough tokens

        CostType beam = default_beam;

        // TODO do something better 
        if(q_size >= q_capacity/2) 
            beam /= 2;

        return beam;
    }

    __device__ __forceinline__ int32 binsearch_maxle(const int32 *vec, const int32 val, int32 low, int32 high) {
        while(true) {
            if(low == high)
                return low; //we know it exists
            if((low + 1) == high)
                return (vec[high] <= val) ? high : low;

            int32 mid = low + (high- low) / 2;

            if(vec[mid] > val)
                high = mid-1;
            else
                low = mid;
        }
    }


    //
    // ExpandArc kernel
    // This kernel does the actual work of traversing arcs 
    //
    // Pseudo code :
    // for all token tok in main_q[main_q_offset...end]:
    //      u = tok.next_state
    //      for all arc a(u->v) in the FST:
    //          v_cost = tok.cost + a.cost + accoustic_cost
    // 
    //          if v_cost < cutoff and v_cost < best_state_cost[v]
    //              generate token associated to v, add to aux_q
    //              update best_state_cost[v]
    //              if necessary update cutoff
    //
    // For more information please refer to http://kaldi-asr.org/doc/decoders.html
    //
    // ExpandArc rely on some preprocessed data to be able to function 
    // for instance, it needs the prefix sum of the arc degree of all token.state in the
    // main_q
    // We need to call a Preprocess kernel before ExpandArc
    //
    // ExpandArc is used for both emitting and nonemitting phases
    // Differences between emitting and nonemitting :
    //      1) params.d_q_arc_offset contains offsets to either emitting or nonemitting arcs. 
    //         It is transparent for this kernel. The differentiation was done in the Preprocess kernel,
    //         which is responsible for filling the params.d_q_arc_offset array
    //      2) Computation of the acoustic cost. If nonemitting, it is equal to 0. If emitting, we need
    //         to use values from the acoustic model (through the d_loglikelihoods array)
    //
    //
    //
    // Note : ExpandArc is not the only kernel able to traverse arcs. 
    // FinalizeProcessNonemitting contains a simplified version of expand for only one CUDA block
    //

    void __global__ _expand_arcs_kernel(ExpandArcParams params) {
        // BlockScan that we will use to compute token indexes in the output queue, 
        // and to find the min cost in the block
        typedef cub::BlockScan<CostTypeAndInt, KALDI_CUDA_DECODER_KERNEL_EXPAND_ARCS_DIMX> BlockScan;
        __shared__ typename BlockScan::TempStorage sh_temp_storage_scan;

        // This kernel writes the new token to the output queue aux_q
        // We will request a spot to store all the new tokens created by threads in this CUDA block
        // sh_aux_q_index_block_offset indicates where to store them in the aux_q
        // tokens created in this CUDA block will be store in :
        // aux_q[sh_aux_q_index_block_offset], aux_q[sh_aux_q_index_block_offset + 1], ...
        __shared__ int32 sh_aux_q_index_block_offset;

        //
        // Cutoff, stored in shared for caching purposes
        // maybe we could rely on cache ?
        //
        __shared__ MinCostAndBeam sh_global_min_cost_and_beam;

        const int32 first_main_q_arc_index_block_offset = blockDim.x*blockIdx.x;
        const int32 total_narcs = *params.d_main_q_narcs;

        // Early exit ASAP if nothing to do
        // does not break the finalize_kernel: code,
        if(first_main_q_arc_index_block_offset >= total_narcs)
            return;

        const int32 main_q_offset = *params.d_main_q_local_offset;
        const int32 main_q_end = *params.d_main_q_end;

        
        if(threadIdx.x == 0) {
            MinCostAndBeamIntegers global_min_cost_and_beam_integers = *params.d_global_min_cost_and_beam;
            sh_global_min_cost_and_beam.min_cost = orderedIntToFloat(global_min_cost_and_beam_integers.min_cost);
            sh_global_min_cost_and_beam.beam = orderedIntToFloat(global_min_cost_and_beam_integers.beam);
        }

        __syncthreads();

        // The condition of this for loop is common for all threads in the block
        // We need to keep all threads in the block alive in the same time
        // We'll have syncs inside the for loop, and we need to have all threads alive during
        // those syncs
        // in the future we may rely on coop groups
        for(int32 main_q_arc_index_block_offset = first_main_q_arc_index_block_offset;
                  main_q_arc_index_block_offset < total_narcs;
                  main_q_arc_index_block_offset += gridDim.x*blockDim.x) {

            //
            // Important : this thread is not responsible for a token in the input queue main_q
            // but for an arc, going out of a token in the main_q
            // The main_q contains in total total_narcs
            // and this thread will compute the main_q_arc_index-th arc of the main_q
            // For instance, first thread in the grid with threadIdx.x == 0 and blockIdx.x == 0 
            // will process the first arc of the token in main_q[main_q_offset + 0] 
            // (if that token has at least one arc)
            //
            // This insure a perfect one thread = one arc load balancing
            // but we have work to do to know exactly which arc is the main_q_arc_index-th arc
            // (what's its source ? its destination ? its arc_idx the FST CSR ?)
            //

            int32 main_q_arc_index = main_q_arc_index_block_offset + threadIdx.x;
        
            // We'll need those variables later in the kernel
            // we declare them outside of the "valid_input" scope
            // to be able to access them later

            int32 main_q_idx;
            int32 arc_idx;
            StateId arc_next_state;
            CostType total_cost = FLT_MAX;

            if(main_q_arc_index < total_narcs) {

                //
                // Current thread must take care of main_q_arc_index-th arc
                // we need to now what's the source of that arc
                // ie which token.state in main_q does it start from ? 
                // We use a binary search in the prefix sum of the token's degree to get that information
                // 
                // Example : main_q contains 3 tokens
                // - First token is associated to a state which has 3 outgoing arc
                // - Second token is associated to a state which has 0 outgoing arc
                // - Third token is associated to a state which has 2 outgoing arc
                //
                // We store the degrees in an array :
                // [3, 0, 2]
                //
                // We then compute the exclusive prefix sum of that array :
                // [0, 3, 3, 5]
                //
                // In total, we have 5 arcs in the main_q. ExpandArc will use 5 threads.
                //
                // Let's say we are the fifth thread in ExpandArc. 
                // we have threadIdx.x == 4, and blockIdx.x == 0
                // it gives us main_q_arc_index == 4
                // From there we have no idea what we're supposed to do next, we need to have information about the
                // arc that we're supposed to traverse
                //
                // To do that, we look for the maximum index maxle_i in the prefix sum array such prefix_sum[i] <= 4
                //
                // [0, 3, 3, 5]
                //         /\
                //         here
                // maxle_i = 2
                // it means that our source token is at index 2 in the main_q
                // and we are computing the arc at index (main_q_arc_index - prefix_sum[maxle_i]) of that token 
                // ie the arc at index (4-3) = 1, the second arc of the second token in main_q
                //

                // Searching for the source of the arc that we will process (main_q_arc_index)
                // we could preprocess the search in the preprocess kernels - for now this kernel is fast enough
                main_q_idx = binsearch_maxle(params.d_main_q_degrees_prefix_sum, main_q_arc_index, main_q_offset, main_q_end-1); 

                // state_first_arc_idx_in_main_q
                // d_main_q_degrees_prefix_sum contains the prefix sum of the 
                // degrees of all tokens in the main_q
                // d_main_q_degrees_prefix_sum[main_q_idx] contains the number of arc
                // in the main_q until that token
                int32 state_first_arc_idx_in_main_q = params.d_main_q_degrees_prefix_sum[main_q_idx];

                // arc_offset_start is the offset in the CSR, to find the arcs 
                // related to the state main_q_state_[main_q_idx]
                // it was set by the preprocess kernel
                int32 arc_offset_start = params.d_q_arc_offsets[main_q_idx];

                // local_arc_index is the arc index for that state
                // if local_arc_index == 2, we will process the second arc
                // of state main_q_state_[main_q_idx]
                int32 local_arc_index = main_q_arc_index - state_first_arc_idx_in_main_q;

                // corresponding arc_idx in the FST
                arc_idx = arc_offset_start + local_arc_index; 

                // Destination of that arc
                arc_next_state = params.arc_nextstates[arc_idx];

                // Building the total cost incrementally 
                // we'll add the acoustic cost and the old token's cost
                CostType arc_fixed_cost = params.arc_weights[arc_idx];

                int32 arc_ilabel = params.is_emitting ? params.arc_ilabels[arc_idx] : 0;
                
                CostType acoustic_cost = (arc_ilabel != 0) ? -params.d_loglikelihoods[arc_ilabel] : 0.0; 
                CostType prev_token_cost  = params.d_main_q_cost[main_q_idx];

                total_cost = prev_token_cost + arc_fixed_cost + acoustic_cost;
 
                // If we're emitting, reload the min_cost, it may have decreased
                CostType min_cost =  params.is_emitting 
                                     ? orderedIntToFloat(params.d_global_min_cost_and_beam->min_cost)
                                     : sh_global_min_cost_and_beam.min_cost;

                // If the total_cost is too large compared to our cutoff (beam search)
                // then let's drop it
                const CostType cutoff = min_cost + sh_global_min_cost_and_beam.beam;

                if(total_cost >= cutoff)
                    total_cost = FLT_MAX;
                else {
                    // We need to check if we already have a token going to that next_state,
                    // and if that token has a lower cost that we have
                    // params.d_state_best_cost[state] contains the best cost for that state in the current frame
                    CostType next_state_best_cost = orderedIntToFloat(params.d_state_best_cost[arc_next_state]);

                    // If that token is the best for that state, drop it
                    if(total_cost >= next_state_best_cost)
                        total_cost = FLT_MAX;
                }
            }

            //
            // If total_cost < FLT_MAX, it means that : 
            // - this thread had a valid input (main_q_arc_index < total_narcs)
            // - the total_cost of the generated token is < cutoff
            // - the generated token is the best candidate for that next_state
            // We will then add that new token in the output queue, aux_q
            // We need to know where to put that token in the aux_q
            // we'll first compute its index inside the CUDA block
            // the first valid output token in the CUDA block will have index 0, 
            // the second index 1... We compute that using a prefix sum
            //
            // We also need to find the overall min cost in the CUDA block
            // a prefix sum is a scan operation, and a min a reduce operation
            // we can perform a reduce operation using a scan (using the last value)
            // we compute the prefix sum and the min in one scan, using the data 
            // struct CostTypeAndInt
            //

            int32 has_successor = (total_cost < FLT_MAX) ? 1 : 0; 

            // Updating the best_state_cost lookup table with our new best cost
            if(has_successor)
                atomicMin(&params.d_state_best_cost[arc_next_state],
                            floatToOrderedInt(total_cost));

            CostTypeAndInt cost_and_index;
            cost_and_index.cost = total_cost; 
            cost_and_index.i = has_successor;

            // This is an /inclusive/ scan
            BlockScan(sh_temp_storage_scan).InclusiveScan(cost_and_index, cost_and_index, MinCostPlusInt());

            if(threadIdx.x == (KALDI_CUDA_DECODER_KERNEL_EXPAND_ARCS_DIMX - 1)) {
                // We can find a lower global_min_cost only in the emitting stage
                if(params.is_emitting) {
                    CostType current_global_min_cost = sh_global_min_cost_and_beam.min_cost;
                    // if we found a lower min_cost, update the global value
                    if(cost_and_index.cost < current_global_min_cost) {
                        current_global_min_cost = cost_and_index.cost;
                        atomicMin(&params.d_global_min_cost_and_beam->min_cost, floatToOrderedInt(current_global_min_cost));
                    }

                    // Reload latest global min cost
                    CostType latest_global_min_cost = orderedIntToFloat(params.d_global_min_cost_and_beam->min_cost);
                    current_global_min_cost = min(current_global_min_cost, latest_global_min_cost);
                    // Updating shared memory
                    sh_global_min_cost_and_beam.min_cost = current_global_min_cost;
                }

                // This is the last thread. The last value of the inclusive scan is the total
                int32 total_successors_in_block = cost_and_index.i;
                
                // Requesting a spot of size total_successors_in_block in the aux_q
                int aux_q_index_block_offset = atomicAdd(params.d_aux_q_end, total_successors_in_block);


                // All threads will need this value
                // Saving in shared memory
                sh_aux_q_index_block_offset = aux_q_index_block_offset;

                //
                // Here we detect an overflow of the aux_q
                // we detect it before actually using the aux_q
                // We try to prevent an overflow from happening using an adaptive beam (cf GetAdaptiveBeam)
                //
                if((sh_aux_q_index_block_offset + total_successors_in_block) >= params.q_capacity) {
                    // sh_aux_q_index_block_offset is in shared memory
                    // its value is currently invalid (overflow)
                    // we set it to a special value and use it as a flag to broadcast
                    // the fact that we have an overflow and that all threads should exit
                    sh_aux_q_index_block_offset = params.q_capacity;

                    // We revert the last operation. All threads that detected the overflow 
                    // will revert what they've done. It means that at the end of the kernel,
                    // we'll be back to the last valid state 
                    // We'll be able to continue computation, but quality of the output
                    // may be lower (we weren't able to save all tokens)
                    atomicAdd(params.d_aux_q_end, -total_successors_in_block); 

                    // Setting the flag for the host. It will be used to print a warning to stderr
                    *params.h_q_overflow = 1; 

                    // We do not jump to finalize_kernel now, because only threadIdx.x == 0 
                    // is executing this if !
                    // We wait until the end of the divergent branch
                } else {
                    // If we are not overflowing the queue, let's check if we need to 
                    // tighten the beam. If the occupancy of the aux_q gets too high,
                    // the adaptive beam will reduce the beam
                    
                    CostType new_beam = GetAdaptiveBeam(params.default_beam, 
                                                        aux_q_index_block_offset,
                                                        params.q_capacity);

                    if(new_beam < sh_global_min_cost_and_beam.beam) {
                        atomicMin(&params.d_global_min_cost_and_beam->beam, floatToOrderedInt(new_beam));
                        sh_global_min_cost_and_beam.beam = new_beam;
                    }

                }
            }

            // Sync'ing for two reasons :
            // - Broadcasting sh_aux_q_index_block_offset
            // - reusing sh_temp_storage (cf CUB's doc)
            __syncthreads(); 


            // The only case where we can have that condition met,
            // if we detected an overflow if the previous lines
            // we need to finalize our work and quit 
            // Now all threads are executing this code. We can jump
            // to finalize_kernel
            if(sh_aux_q_index_block_offset == params.q_capacity) 
                goto finalize_kernel; // keeping things clean before aborting

            //
            // If we're executing the following lines it means everything
            // is valid and we are not overflowing the aux_q
            //
            cost_and_index.i -= has_successor; // we want the exclusive sum now

            int32 aux_q_block_index = cost_and_index.i;
            int32 aux_q_index = sh_aux_q_index_block_offset + aux_q_block_index;

            if(has_successor) {
                // We save the new token to the aux_q
                
                params.d_aux_q_cost[aux_q_index] = total_cost;
                params.d_aux_q_state[aux_q_index] = arc_next_state;
                
                // We've updated the cutoff since created the new token
                // there's a chance that we no longer need to use that token
                // we've saved the cost and the state to be able to ignore it in the following prune operation
                // but there's no need to write out the infos

                InfoToken new_tok_info;
                // Index of the parent token
                // the parent is the token used as input 
                // that parent is at index main_q_idx in the GPU memory
                // However, the main_q is emptied before processing a new frame
                // we need to add the offset related to the previous frames index
                // we add params.main_q_global_offset
                new_tok_info.prev_token = params.main_q_global_offset + main_q_idx;
                new_tok_info.arc_idx = arc_idx;

                params.d_aux_q_info[aux_q_index] = new_tok_info;
            }
        }

        finalize_kernel:

        // We want to be sure that all threads are done before declaring this CUDA block as done
        __syncthreads(); 

        if(threadIdx.x == 0) {
            // Declaring this CTA as done
            int32 old = atomicAdd(params.d_n_CTA_done, 1);

            // Taking into account the CTAs that have early exited at the begining of this kernel
            int32 total_active_CTAs = min(KALDI_CUDA_DECODER_DIV_ROUND_UP(total_narcs, KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX), gridDim.x);

            // If we're the last CTA to exit - detect it
            if(old == (total_active_CTAs -1)) {
                __threadfence(); // we want last value of d_aux_q_end

                // h_* pointers are in pinned memory, we can update them from the GPU
                *params.h_aux_q_end = *params.d_aux_q_end;
                *params.d_main_q_narcs = 0;
                *params.h_main_q_narcs = 0;
                *params.d_n_CTA_done = 0; 

                if(params.is_emitting) {
                    // It was the last time that we were using tokens in the main_q
                    // flushing it now
                    *params.d_main_q_end = 0;
                    *params.h_main_q_end = 0;
                } else {
                    // Tokens processed in that nonemitting iteration will be ignored in the next iteration
                    *params.d_main_q_local_offset = main_q_end;
                    *params.h_main_q_local_offset = main_q_end;
                }

            }
        }

    }

    void CudaDecoder::ExpandArcs(bool is_emitting, int32 main_q_narcs_estimate) {
        dim3 grid,block;

        KALDI_ASSERT(main_q_narcs_estimate > 0);

        block.x = KALDI_CUDA_DECODER_KERNEL_EXPAND_ARCS_DIMX;
        grid.x = KALDI_CUDA_DECODER_DIV_ROUND_UP(main_q_narcs_estimate, block.x);

        // The two members of params we need to update
        expand_params_.main_q_global_offset = main_q_global_offset_;
        expand_params_.is_emitting = is_emitting;
            
        _expand_arcs_kernel<<<grid,block,0,compute_st_>>>(expand_params_);
    }



    /*
        
       FinalizeProcessNonemitting
       Meta-kernel (merging preprocess and expand) but only works with 1 CUDA block

       Used to avoid calling multiple "heavy lifting" kernels for the tail of non emitting
       (lots of iterations with small number of arcs)

       Code is greatly simplified because we can have only one CTA alive

       Repeat until new queue empty:
       1) Preprocess 
       2) Expand arcs

       The preprocess stage is not done on the first iteration, because it was
       already done by the ProcessAndContract kernel. We always call ProcessAndContract
       before calling FinalizeProcessNonemitting 

       At the end, this kernel finalize the computation for current frame,
       so that it's ready for next ProcessEmitting

       TODO This kernel could be easily optimized  

       Note : For a detailed description on how the Preprocess and Expand operation work,
       please refer to the PreprocessInPlace and ExpandArc kernel implemention. The algorithm are 
       described there. In this kernel, we compute simplified version of preprocess and expand, because
       we do not need inter-block communication (we launch only one CUDA block)

       Important : in ExpandArc, the input is the main_q, the ouput is the aux_q. We then call PreprocessAndContract
       that move the tokens from the aux_q to the main_q.
       Here we directly output the tokens in the main_q. It helps use simplify the code, and we are not generating a lot
       of tokens anyway (so the pruning stage of PreprocessAndContract is less critical)

     */


    __launch_bounds__(KALDI_CUDA_DECODER_KERNEL_NONEM_LT_DIMX, 1)
        __global__ void _finalize_process_non_emitting(const uint32_t *d_arc_offsets, 
                ExpandArcParams params) {
           
            // Used to compute the index in the output queue
            typedef cub::BlockScan<TokenAndArcCount, KALDI_CUDA_DECODER_KERNEL_NONEM_LT_DIMX> BlockScanTokenAndArcCount;
            __shared__ typename BlockScanTokenAndArcCount::TempStorage sh_temp_storage_scan_token_arc;

            typedef cub::BlockScan<int, KALDI_CUDA_DECODER_KERNEL_NONEM_LT_DIMX> BlockScanInt;
            __shared__ typename BlockScanInt::TempStorage sh_temp_storage_scan_int;

            int32 total_narcs = *params.d_main_q_narcs;

            // We launch the kernel before knowing the value of total_narcs
            // if it was too early, we exit and the host will relaunch heavy load
            // non emitting kernels (using Preprocess and ExpandArc kernels)
            if(total_narcs >= KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS) 
                return;

            int32 main_q_offset = *params.d_main_q_local_offset;
            int32 main_q_end = *params.d_main_q_end;
            
            // aux_q is empty when this kernel is called
            int32 aux_q_end = 0;

            MinCostAndBeamIntegers global_min_cost_and_beam = *params.d_global_min_cost_and_beam;
            CostType global_min_cost = orderedIntToFloat(global_min_cost_and_beam.min_cost);
            CostType beam = orderedIntToFloat(global_min_cost_and_beam.beam);

            while(total_narcs > 0) {

                // Step 1 : ExpandArcs

                for(int32 main_q_arc_index_block_offset = 0;
                          main_q_arc_index_block_offset < total_narcs;
                          main_q_arc_index_block_offset += blockDim.x) {

                    int32 main_q_arc_index = main_q_arc_index_block_offset + threadIdx.x;

                    // For details on how this code works, please refer to ExpandArc's comments
                    CostType total_cost = FLT_MAX;
                    int32 arc_idx;
                    StateId arc_next_state;
                    int32 main_q_idx;

                    if(main_q_arc_index < total_narcs) {
                        main_q_idx = binsearch_maxle(params.d_main_q_degrees_prefix_sum, 
                                                main_q_arc_index, 
                                                main_q_offset,
                                                main_q_end-1); 

                        int32 state_first_arc_idx_in_main_q = params.d_main_q_degrees_prefix_sum[main_q_idx];
                        int32 arc_offset_start = params.d_q_arc_offsets[main_q_idx];

                        arc_idx = arc_offset_start + (main_q_arc_index - state_first_arc_idx_in_main_q);

                        arc_next_state = params.arc_nextstates[arc_idx];
                        CostType arc_weight = params.arc_weights[arc_idx];
                        CostType next_state_cost = orderedIntToFloat(params.d_state_best_cost[arc_next_state]);
                        CostType old_tok_cost = params.d_main_q_cost[main_q_idx];

                        total_cost = arc_weight + old_tok_cost;

                        CostType cutoff = global_min_cost + beam;
                        if(total_cost >= cutoff || total_cost >= next_state_cost) {
                            total_cost = FLT_MAX;
                        } 
                    }
                     
                    int32 has_successor = (total_cost < FLT_MAX) ? 1 : 0;

                    if(has_successor) {
                        //TODO _block
                        atomicMin(&params.d_state_best_cost[arc_next_state], floatToOrderedInt(total_cost)); 
                    }

                    int32 local_aux_q_idx;
                    int32 total_ntokens_to_aux_q;
                    BlockScanInt(sh_temp_storage_scan_int).ExclusiveSum(has_successor, 
                                                                        local_aux_q_idx,
                                                                        total_ntokens_to_aux_q);

                    // Checking if we are not overflowing the aux_q
                    if((aux_q_end + total_ntokens_to_aux_q) >= params.q_capacity) {
                        *params.h_q_overflow = 1;
                        
                        goto finalize_kernel;
                    }


                    if(has_successor) {
                        int32 aux_q_idx = aux_q_end + local_aux_q_idx;
                        params.d_aux_q_state[aux_q_idx] = arc_next_state;
                        params.d_aux_q_cost[aux_q_idx] = total_cost;

                        InfoToken new_tok_info;
                        new_tok_info.prev_token = params.main_q_global_offset + main_q_idx;

                        new_tok_info.arc_idx = arc_idx;
                        params.d_aux_q_info[aux_q_idx] = new_tok_info;
                   }

                    aux_q_end += total_ntokens_to_aux_q;

                    // Getting new beam using aux_q_end
                    beam = GetAdaptiveBeam(params.default_beam, 
                            aux_q_end,
                            params.q_capacity);


                    // reusing sh_temp_storage_scan_int
                    __syncthreads();
                }

                // Step 2 : PreprocessAndContract
                // Sync : reusing some data pointers, like d_main_q_prefix_sum

                // Reset for new iteration
                total_narcs = 0;
                main_q_offset = main_q_end;

                for(int32 block_off = 0;
                        block_off < aux_q_end;
                        block_off += blockDim.x) {

                    int32 aux_q_idx = block_off + threadIdx.x;

                    int32 degree = 0;
                    int32 start = -1;

                    StateId token_state;
                    CostType token_cost;

                    if(aux_q_idx < aux_q_end) {
                        token_state = params.d_aux_q_state[aux_q_idx];
                        token_cost = params.d_aux_q_cost[aux_q_idx];

                        // beam may have changed since generation
                        CostType cutoff = global_min_cost + beam;
                        if(token_cost < cutoff) {
                            CostType best_cost = orderedIntToFloat(params.d_state_best_cost[token_state]);

                            if(token_cost == best_cost) {
                                start = d_arc_offsets[token_state];
                                int32 end = d_arc_offsets[token_state+1];
                                degree = end - start;
                            }
                        }
                    }

                    bool has_valid_nonpruned_token = (start != -1);

                    TokenAndArcCount token_and_arc_count;
                    token_and_arc_count.ntokens = has_valid_nonpruned_token ? 1 : 0;
                    token_and_arc_count.narcs   = degree;
                    TokenAndArcCount scan_aggregate;

                    TokenAndArcCount zero_struct;
                    zero_struct.ntokens = zero_struct.narcs = 0;

                    BlockScanTokenAndArcCount(sh_temp_storage_scan_token_arc).ExclusiveScan(token_and_arc_count, 
                            token_and_arc_count,
                            zero_struct,
                            TokenAndArcCountSum(),
                            scan_aggregate);

                    // Checking if we are not overflowing the main_q
                    int32 total_ntokens_to_main_q = scan_aggregate.ntokens;
                    if((main_q_end + total_ntokens_to_main_q) >= params.q_capacity) {
                        *params.h_q_overflow = 1;

                        goto finalize_kernel;
                    }
                    
                    int32 degree_this_iteration_prefix_sum = token_and_arc_count.narcs;
                    int32 degree_sum_for_this_iteration = scan_aggregate.narcs;

                    int32 degree_prefix_sum = total_narcs + degree_this_iteration_prefix_sum;
                    total_narcs += degree_sum_for_this_iteration;

                    if(has_valid_nonpruned_token) {
                        int32 local_main_q_idx = token_and_arc_count.ntokens;
                        int32 main_q_idx = main_q_end + local_main_q_idx;

                        params.d_q_arc_offsets[main_q_idx] = start;
                        params.d_main_q_degrees_prefix_sum[main_q_idx] = degree_prefix_sum;
                        params.d_main_q_state[main_q_idx] = token_state;
                        params.d_main_q_cost[main_q_idx] = token_cost;

                        InfoToken info_token = params.d_aux_q_info[aux_q_idx];
                        params.d_main_q_info[main_q_idx] = info_token;
                    }
                    
                    main_q_end += total_ntokens_to_main_q; 

                    __syncthreads(); // reusing sh_temp_storage_scan
                }

                aux_q_end = 0; // aux_q is now considered empty
            }

            finalize_kernel:

            if(threadIdx.x == 0) {
                // Next step is ProcessEmitting of next frame, from is currToken_offset
                *params.d_main_q_end = main_q_end; 
                *params.h_main_q_end = main_q_end; 

                *params.d_main_q_local_offset = 0; 
                *params.h_main_q_local_offset = 0; 

                // No need to update the cutoff - maybe the beam
                //*params.d_cutoff = floatToOrderedInt(sh_cutoff);
            }

        }

    void CudaDecoder::FinalizeProcessNonemitting() {
        dim3 grid,block;
        block.x = KALDI_CUDA_DECODER_KERNEL_NONEM_LT_DIMX;
        grid.x = 1; // this kernel is designed for one CTA 

        expand_params_.main_q_global_offset = main_q_global_offset_;
        expand_params_.is_emitting = false;

        _finalize_process_non_emitting<<<grid,block,0,compute_st_>>>(fst_.d_ne_offsets_, expand_params_);
    }


} // end namespace kaldi
