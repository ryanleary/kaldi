// decoder/cuda-decoder-kernels.cu

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

#define DIV_ROUND_UP(a,b) ((a+b-1)/b)

namespace kaldi {

typedef CudaDecoder::StateId StateId;
typedef CudaDecoder::TokenAndArcCount TokenAndArcCount;
typedef CudaDecoder::TokenAndArcCountUnion TokenAndArcCountUnion;
typedef CudaDecoder::CostType CostType;
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

    // Temporary used for cutoff - will be TODO removed
    __device__ float fatomicMin(float *addr, float value)

    {

        float old = *addr, assumed;
        if(old <= value) return old;

        do
        {
            assumed = old;
            old = atomicCAS((uint32_t*)addr,
                    __float_as_int(assumed),
                    __float_as_int(value));

        } while(old!=assumed); // TODO <

        return old;

    }

    //
    // Kernels
    //

    // For description of what each kernel is doing, please refer to cuda-decoder.h
    // and look for the corresponding wrapper
    // for instance, for a description of _init_lookup_kernel,
    // look for the description of CudaDecoder::InitStateCostLookup() in cuda-decoder.h

    // Used before first frame
    __global__ void _init_state_cost_lookup_kernel(int32 size, int32 *state_cost) {
        for(int32 idx = blockIdx.x*blockDim.x + threadIdx.x;
                idx < size;
                idx += blockDim.x*gridDim.x) {
            state_cost[idx]  = floatToOrderedInt(FLT_MAX);
        }
    }

    void CudaDecoder::InitStateCostLookup() {
        int32 nstates = fst_.numStates;
        KALDI_ASSERT(nstates > 0);

        dim3 grid,block;
        block.x = KALDI_CUDA_DECODER_KERNEL_INIT_LOOKUP_DIMX;
        grid.x = DIV_ROUND_UP(nstates, block.x);

        _init_state_cost_lookup_kernel<<<grid,block>>>(nstates, d_state_cost_);
    }

    // Used to reset lookup table between frames
    // Using the queue to reset only the values needed
    // Also takes care of resetting cutoff
    __global__ void _reset_state_cost_lookup_kernel(const StateId *d_main_q_state_, const int32 *d_main_q_end_, int32 *d_state_cost, CostType *d_cutoff) {
        int32 main_q_end = *d_main_q_end_; 

        for(int32 idx = blockIdx.x*blockDim.x + threadIdx.x;
                idx < main_q_end;
                idx += blockDim.x*gridDim.x) {
            // d_main_q_state_ contains the list of states that we've considered in the last frame
            // it corresponds to the list of indexes i such as d_state_cost[i] < +INF
            // faster than init_state_cost_lookup_kernel by a factor of ~10
            StateId state = d_main_q_state_[idx];
            d_state_cost[state]  = floatToOrderedInt(FLT_MAX);
        }

        if(blockIdx.x == 0 && threadIdx.x == 0)
            *d_cutoff = FLT_MAX; // we also reset the cutoff
    }

    void CudaDecoder::ResetStateCostLookup() {
        int32 size = *h_main_q_end_;

        KALDI_ASSERT(size > 0);

        dim3 grid,block;
        block.x = KALDI_CUDA_DECODER_KERNEL_INIT_LOOKUP_DIMX;
        grid.x = DIV_ROUND_UP(size, block.x);

        _reset_state_cost_lookup_kernel<<<grid,block,0,compute_st_>>>(d_main_q_state_, d_main_q_end_, d_state_cost_, d_cutoff);
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
        __shared__ typename BlockScan::TempStorage temp_storage;

        // This CUDA block (CTA) will count the number of tokens it has to move to the main_q
        // and store the result in nsurvival_tokens_in_CTA
        __shared__ int32 nsurvival_tokens_in_CTA;

        // We need to move the survival tokens to the main_q
        // 
        // main_q_global_block_offset has two purposes :
        // (1) to know where to store the survival tokens in the main_q
        // (2) to perform the prefix sum degrees of the survival degrees
        //
        // The reason why we store those two values together is because they are linked (see below)
        //
        // (1) We need a spot to store those tokens in the main_q 
        // We will ask the main_q counter where to store those tokens, the answer will be 
        // an offset of the main_q. We will store our tokens in positions :
        // d_main_q_state[main_q_global_block_offset.ntokens], d_main_q_state[main_q_global_block_offset.ntokens+1]...
        //
        // (2) main_q_global_block_offset.narcs contains the number of arcs in the main_q up until index main_q_global_block_offset.ntokens
        // ie the number of arcs going out of all states in d_main_q_state[0..main_q_global_block_offset.ntokens]
        // it is used to compute the global prefix sum of degrees in one pass
        //
        __shared__ TokenAndArcCountUnion main_q_global_block_offset;

        // Final cutoff from last ExpandArc execution
        const BaseFloat cutoff = *params.d_cutoff;

        const int32 aux_q_end = *params.d_aux_q_end;

        // The condition of the for loop is the same for all threads in the CUDA block
        // we want to keep all threads alive at the same time for now
        // otherwise __syncthreads() would fail
        for(int32 block_offset = blockDim.x*blockIdx.x;
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
                BaseFloat state_best_cost = orderedIntToFloat(params.d_state_cost[token_state]);

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

                // the d_state_cost lookup table is reset to +INF for all states between frame
                // for perf. reason we only reset states that are in d_main_q_state
                // however if state_best_cost >= cutoff, all tokens associated with token_state 
                // will be pruned, and that state will not be in d_main_q_state
                // we need to reset the lookup table now

                if (state_best_cost >= cutoff)
                    params.d_state_cost[token_state] = floatToOrderedInt(FLT_MAX);

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
            BlockScan(temp_storage).ExclusiveScan(block_prefix_sum_token_arc_count, 
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

                nsurvival_tokens_in_CTA = token_and_arc_count_block_sum.split.ntokens;
                
                // Doing two things at the same time :
                // requesting a spot in the main_q to store the survival tokens from this CTA 
                // (we need space for token_and_arc_count_block_sum.split.ntokens tokens)
                // informing the main_q that our survival tokens contain token_arc_count_block_sum.split.narcs arcs
                //
                // We then store the return value, which is the global offset on where to store those tokens,
                // and the total number of arcs up until that global offset
                main_q_global_block_offset.both = atomicAdd(&params.d_main_q_end_and_narcs_i2->both, token_and_arc_count_block_sum.both);
            }

            // Syncing for three reasons :
            // - Broadcasting main_q_global_block_offset
            // - Broadcasting nsurvival_tokens_in_CTA
            // - We may reuse temp_storage (cf CUB doc)
            __syncthreads(); 

            // Checking if we are overflowing the main_q
            if((main_q_global_block_offset.split.ntokens + nsurvival_tokens_in_CTA) >= params.q_capacity) {
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

                int32 main_q_idx = main_q_global_block_offset.split.ntokens + block_prefix_sum_token_arc_count.ntokens;

                InfoToken token_info = params.d_aux_q_info[aux_q_idx];

                // Moving the token to the main q
                params.d_main_q_state[main_q_idx] = token_state;
                params.d_main_q_cost[main_q_idx] = token_cost;
                params.d_main_q_info[main_q_idx] = token_info;

                // Saving the global prefix sum
                // = (narcs until now in the main queue) + (narcs until this thread in the CTA)
                params.d_main_q_degrees_prefix_sum[main_q_idx] = main_q_global_block_offset.split.narcs 
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

            // If we're the last CTA to exit, detect it
            bool is_last_CTA = (old == (gridDim.x -1));

            if(is_last_CTA) {
                __threadfence();

                // We added things to the main_q
                // d_main_q_end was modified
                // we update h_main_q_end to keep it consistent
                // the h_* pointers are in the pinned host memory, we can access them from the device
                *params.h_main_q_end = *params.d_main_q_end;
                *params.h_main_q_narcs = *params.d_main_q_narcs;

                // We moved what we had to move from the aux q to the main q
                // We now empty the aux q 
                *params.d_aux_q_end = 0;
                *params.h_aux_q_end = 0; 

                // Reset the counter for next time
                *params.d_n_CTA_done = 0;
            }
        }

    }


    void CudaDecoder::PreprocessAndContract(const PreprocessParams &params) {
        dim3 grid,block;
        block.x = KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX;
        grid.x = DIV_ROUND_UP(*h_aux_q_end_, block.x);

        KALDI_ASSERT(grid.x > 0);

        _preprocess_and_contract_kernel<<<grid,block,0,compute_st_>>>(params);
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
        __shared__ typename BlockScan::TempStorage temp_storage;


        // All threads in the last CUDA block (CTA) alive will have work to do at the end
        // this bool will be needed to broadcast the information from thread0 to all threads in the last CTA 
        __shared__ bool is_last_CTA;

        const int32 main_q_offset = *params.d_main_q_local_offset;
        const int32 main_q_end = *params.d_main_q_end;
        const int32 main_q_size = main_q_end - main_q_offset;

        // Final cutoff from the expand kernel
        const BaseFloat cutoff = *params.d_cutoff;

        // The condition of the for loop is the same for all threads in the CUDA block
        // we want to keep all threads alive at the same time for now
        // otherwise __syncthreads() would fail
        for(int32 block_offset = blockDim.x*blockIdx.x;
                block_offset < main_q_size;
                block_offset += gridDim.x*blockDim.x) {

            // Position of considered token in the main_q
            int32 main_q_idx = main_q_offset + block_offset + threadIdx.x; 

            // Total number of arcs from that token's state
            int32 degree = 0; 

            if(main_q_idx < main_q_end) {
                StateId token_state = params.d_main_q_state[main_q_idx]; 
                BaseFloat token_cost = params.d_main_q_cost[main_q_idx];

                // the cutoff may have decreased since the creation of that token
                if(token_cost < cutoff) {

                    // Best cost for that token_state
                    // We know we have a token associated with token_state in the queue with the cost state_best_cost
                    BaseFloat state_best_cost = orderedIntToFloat(params.d_state_cost[token_state]); 
                    
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
            BlockScan(temp_storage).ExclusiveSum(degree, degree_local_prefix_sum);

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
                params.d_main_q_degrees_block_prefix_sum[local_sum_index] = local_sum; 
            }


            // Synchronization for two reasons :
            // - we may need to reuse temp_storage if the for loop iterates (cf CUB's doc)
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
            is_last_CTA = (old == (gridDim.x -1));
        }

        // Synchronization for two reasons :
        // - Broadcasting is_last_CTA
        // - reusing temp_storage (cf CUB's doc)
        __syncthreads();
        
        if(is_last_CTA)
        {
            //
            // Our goal here is to compute the prefix sum of the previous local sums
            // What we call local sum is what contains the local_sum variables in the previous lines
            // it is the sum of degrees inside a given CUDA block, at a given for loop iteration
            // all local sums are stored in params.d_main_q_degrees_block_prefix_sum
            // we want to do the prefix sum of that array
            //
            // Once this is done, params.d_main_q_degrees_block_prefix_sum[i] will contain the 
            // offset that we need to add to the local prefix sum #i to convert it to a global
            // prefix sum
            // Right now we are only computing the offsets ; adding them to the local prefix sums will be 
            // done in FinalizePreprocessInPlace
            //

            //
            // We are the last CTA alive
            // which means that all local sums have been written to params.d_main_q_degrees_block_prefix_sum
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

            int32 number_of_local_sums = DIV_ROUND_UP(main_q_size, KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX);

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
                                ? params.d_main_q_degrees_block_prefix_sum[local_sum_index] 
                                : 0; // neutral element

                int32 prefix_sum_of_local_sums, total_sum_of_local_sums_for_this_iteration;

                BlockScan(temp_storage).ExclusiveSum(local_sum, prefix_sum_of_local_sums, total_sum_of_local_sums_for_this_iteration);

                prefix_sum_of_local_sums += prefix_sum_of_local_sums_offset;
                prefix_sum_of_local_sums_offset += total_sum_of_local_sums_for_this_iteration;

                if(local_sum_index < number_of_local_sums) {
                    params.d_main_q_degrees_block_prefix_sum[local_sum_index] = prefix_sum_of_local_sums;
                }

                // Sync'ing to be able to reuse temp_storage (cf CUB's doc)
                __syncthreads();
            }

            if(threadIdx.x == 0)
            {
                // Final offset is the overall total
                int total_sum_of_local_sums = prefix_sum_of_local_sums_offset;
                *params.d_main_q_narcs = total_sum_of_local_sums; 
                // h_main_q_narcs is in pinned memory, we can write to it from the device
                *params.h_main_q_narcs = total_sum_of_local_sums; 
                // reset for next time
                *params.d_n_CTA_done = 0;
            }
        }
    }


    void CudaDecoder::PreprocessInPlace(const PreprocessParams &params) {
        dim3 grid,block;
        block.x = KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX;
        int32 main_q_size = *h_main_q_end_ - *h_main_q_local_offset_;

        grid.x = DIV_ROUND_UP(main_q_size, block.x);

        KALDI_ASSERT(grid.x > 0);

        _preprocess_in_place_kernel<<<grid,block,0,compute_st_>>>(params);
    }



    /*

       Part 2 of the scan for "PreprocessEmitting". For NonEmitting scan is already final

       Computes global prefix sum with block prefix sum and block offsets

       If we want to speed up expand, we can compute lower and upper bound to restrain 
       the binary search in expand
       This can be done on the fly here, and removes main bottleneck of expand
       Not done for now, because expand is fast enough

     */
    __global__ void _finalize_degrees_scan_kernel(int32 *d_scan, int32 *d_blk_scan, const int32 *d_main_q_local_offset_, const int32
            *d_main_q_end_) {

        int32 q_off = *d_main_q_local_offset_;
        int32 q_end = *d_main_q_end_;
        int32 q_size = q_end - q_off;

        for(int32 idx = q_off + blockDim.x*blockIdx.x + threadIdx.x;
                idx < q_size;
                idx += blockDim.x*gridDim.x) {

            int32 blk_idx = (idx - q_off) / KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX;
            int32 blk_scan_offset = d_blk_scan[blk_idx]; // we rely on L1 for this one, avoiding syncs

            d_scan[idx] += blk_scan_offset;
        }

    }

    void CudaDecoder::FinalizePreprocessInPlace() {
        dim3 grid,block;
        block.x = KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX;
        int32 main_q_size = *h_main_q_end_ - *h_main_q_local_offset_;
        grid.x = DIV_ROUND_UP(main_q_size, block.x);

        // If the main_q is empty, we will not be able to continue
        KALDI_ASSERT(grid.x > 0);

        _finalize_degrees_scan_kernel<<<grid,block,0,compute_st_>>>(d_main_q_degrees_prefix_sum_, d_main_q_degrees_block_prefix_sum_, d_main_q_local_offset_,
                d_main_q_end_); 
    }




    /*
       This kernel propagates arcs from the main q [main_q_local_offset, main_q_end[
       to the aux

       The main bottleneck is the first binary search. 
       If we want to remove it, preprocess it on the fly in preprocess

     */

    struct CostTInt {
        CostType cost;
        int32 i;
    };

    struct CISum {
        __device__ CostTInt operator()(const CostTInt &a, const CostTInt &b) const {
            CostTInt c;
            c.cost = fmin(a.cost, b.cost);
            c.i = a.i + b.i;

            return c;
        }
    };


__device__ __inline__ CostType GetCutoffCandidate(const CostType current_cutoff,
                                const CostType min_cost,
                                const CostType default_beam,
                                const int32 q_size,
                                const int32 q_capacity) {
                                 

    // Doing something simple for now
    // We have to keep beam large enough,
    // the final cutoff will be used for the final
    // prune. If it is too small, we won't keep enough tokens

   CostType beam = default_beam;

   if(q_size >= q_capacity/2) 
       beam /= 2;

    return fmin(current_cutoff, min_cost + beam);
}

    __forceinline__ __device__ int32 binsearch_maxle(const int32 *vec, const int32 val, int32 low, int32 high) {
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


    void __global__ _expand_arcs_kernel(ExpandArcParams params) {
        typedef cub::BlockScan<CostTInt, KALDI_CUDA_DECODER_KERNEL_EXPAND_ARCS_DIMX> BlockScan;

        __shared__ typename BlockScan::TempStorage temp_storage_scan;

        __shared__ int32 to_q_block_offset;
        __shared__ CostType blk_cutoff;

        const int32 total_narcs = *params.d_main_q_narcs;
        const int32 main_q_offset = *params.d_main_q_local_offset;
        const int32 main_q_end = *params.d_main_q_end;

        
        if(threadIdx.x == 0) {
            blk_cutoff = *params.d_cutoff;
        }

        __syncthreads();

        // Keeping the whole CTA alive, we'll have syncs
        for(int32 block_offset = blockDim.x*blockIdx.x;
                block_offset < total_narcs;
                block_offset += gridDim.x*blockDim.x) {

            int32 th_idx = block_offset + threadIdx.x;
            bool valid_input = (th_idx < total_narcs);

            BaseFloat total_cost = FLT_MAX;
            int32 arc_idx;
            StateId arc_next_state;
            int32 main_q_idx;

            if(valid_input) {
                //we can do better than that
                main_q_idx = binsearch_maxle(params.d_main_q_degrees_prefix_sum, th_idx, main_q_offset, main_q_end-1); 

                int32 lower_bound = params.d_main_q_degrees_prefix_sum[main_q_idx];
                int32 arc_offset_start = params.d_q_arc_offsets[main_q_idx];

                arc_idx = arc_offset_start + (block_offset + threadIdx.x - lower_bound);
                arc_next_state = params.arc_nextstates[arc_idx];

                total_cost = params.arc_weights[arc_idx];

                int32 arc_ilabel = params.is_emitting ? params.arc_ilabels[arc_idx] : 0;
                total_cost += (arc_ilabel != 0) ? -params.d_loglikelihoods[arc_ilabel] : 0.0; 
                total_cost += params.d_main_q_cost[main_q_idx];

                if(total_cost >= blk_cutoff)
                    valid_input = false;
                else {
                    // switch back to red, worst case is bad
                    BaseFloat next_state_cost = orderedIntToFloat(params.d_lookup[arc_next_state]);

                    if(total_cost >= next_state_cost)
                        valid_input = false;
                }
            }

                            int32 has_successor = valid_input ? 1 : 0;  // Need a spot in the new q
                            CostTInt ci;
                            ci.cost = valid_input ? total_cost : FLT_MAX; 
                            ci.i = has_successor;

                            BlockScan(temp_storage_scan).InclusiveScan(ci, ci, CISum());

                            if(threadIdx.x == (KALDI_CUDA_DECODER_KERNEL_EXPAND_ARCS_DIMX - 1)) {
                                int32 total_successors_in_block = ci.i;
                                to_q_block_offset = atomicAdd(params.d_aux_q_end, total_successors_in_block);
                                if((to_q_block_offset + total_successors_in_block) >= params.q_capacity) {
                                    to_q_block_offset = params.q_capacity; // used to broadcast the info

                                }
                                /*
                                
                                GetCutoffCandidate takes int32o account the current value of 
                                d_aux_q_end and compares it with its maximum capacity.
                                If necessary it progressively cuts down the beam 
                                (reducing the cutoff) to only keep the best candidates
                                and avoiding an overflow

                                */
                                CostType cutoff_candidate = GetCutoffCandidate(blk_cutoff,
                                                                  ci.cost,
                                                                  params.beam,
                                                                  to_q_block_offset + total_successors_in_block,
                                                                  params.q_capacity);

                                blk_cutoff = (cutoff_candidate < blk_cutoff) 
                                             ? fmin(fatomicMin(params.d_cutoff, cutoff_candidate), cutoff_candidate)
                                             : fmin(*params.d_cutoff, blk_cutoff);
                            }

                            __syncthreads(); // to_q_block_offset


                            // aux_q is full. UpdateCutoff should prevent this from happening
                            if(to_q_block_offset == params.q_capacity) {
                                if(threadIdx.x == (KALDI_CUDA_DECODER_KERNEL_EXPAND_ARCS_DIMX - 1)) {
                                    // Revert
                                    int32 total_successors_in_block = ci.i;
                                    atomicAdd(params.d_aux_q_end, -total_successors_in_block); 
                                    *params.h_q_overflow = 1; 
                                }

                                goto finalize_kernel; // keeping things clean before aborting
                            }

                            ci.i -= has_successor; // we want the exclusive sum now
                            int32 to_q_index = to_q_block_offset + ci.i;

                            if(has_successor) {
                                params.d_aux_q_cost[to_q_index] = total_cost;
                                params.d_aux_q_state[to_q_index] = arc_next_state;
                                
                                atomicMin(&params.d_lookup[arc_next_state],
                                floatToOrderedInt(total_cost)
                                );

                                //print32f("cost = %f, cutoff = %f, beam=%f \n", total_cost, blk_cutoff, params.beam);
                                if(total_cost < blk_cutoff) { // cutoff may have changed
                                    // We write the rest of the token only if necessary
                                    // if the cost is higher than cutoff, 
                                    // the token will be ignored anyway 


                                    InfoToken new_tok_info;
                                    new_tok_info.prev_token = params.main_q_global_offset + main_q_idx;
                                    new_tok_info.arc_idx = arc_idx;
                            

                                    params.d_aux_q_info[to_q_index] = new_tok_info;

                                    /*
                                    print32f("expand, adding %i (%i)  -> %i \n", new_tok_info.prev_token,
                                    params.main_q_global_offset, arc_next_state);
                                    */
                                }
                            }
        }

        finalize_kernel:

        __syncthreads(); // avoiding races on d_main_q_narcs for instance

        // Last block alive sets h_aux_q_end_ (pinned memory)
        if(threadIdx.x == 0) {
            int32 old = atomicAdd(params.d_n_CTA_done, 1);
            if(old == (gridDim.x -1)) {
                __threadfence(); // we want last value of d_aux_q_end
                *params.h_aux_q_end = *params.d_aux_q_end;
                *params.d_n_CTA_done = 0;
                *params.d_main_q_narcs = 0;
                *params.h_main_q_narcs = 0;

                if(params.is_emitting) {
                    *params.d_main_q_local_offset = 0; // not needed
                    *params.h_main_q_local_offset = 0; // not needed
                    *params.d_main_q_end = 0;
                    *params.h_main_q_end = 0;
                } else {
                    *params.d_main_q_local_offset = main_q_end;
                    *params.h_main_q_local_offset = main_q_end;
                }

            }
        }

    }

    void CudaDecoder::ExpandArcs(const ExpandArcParams &params, int32 nthreads) {
        dim3 grid,block;
        block.x = 256;
        grid.x = DIV_ROUND_UP(nthreads, block.x);

        // It's possible to have zero threads and still be valid
        if(grid.x > 0)
            _expand_arcs_kernel<<<grid,block,0,compute_st_>>>(params);
    }


    // Wrote for single CTA

    /*

       Persistent kernel

       Used to avoid calling multiple "heavy lifting" kernels for the tail of non emitting
       (lots of iterations with small number of arcs)

       Code is greatly simplified because we can have only one CTA alive

       Repeat until new queue empty:
       1) Computes degrees (cf ComputeDegrees) 
       2) Compute scan
       3) Expand arcs

       1 and 2 are not done on the first iteration, because it's already done
       (by corresponding kernels)

       At the end, this kernel finalize the computation for current frame,
       so that it's ready for next ProcessEmitting

       We could optimize and speed up this kernel
       It will only gives us a better latency for 1 stream, which is low enough
       Instead, we let it compute while we use the GPU for other streams
       This kernel only uses one block

     */


    __launch_bounds__(KALDI_CUDA_DECODER_KERNEL_NONEM_LT_DIMX, 1)
        __global__ void _process_nonem_longtail(const uint32_t *d_arc_offsets, 
                ExpandArcParams params) {

            typedef cub::BlockScan<int32, KALDI_CUDA_DECODER_KERNEL_NONEM_LT_DIMX> BlockScan;
            typedef cub::BlockReduce<float, KALDI_CUDA_DECODER_KERNEL_NONEM_LT_DIMX> BlockReduce;

            __shared__ typename BlockScan::TempStorage temp_storage_scan;
            __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

            __shared__ BaseFloat cutoff;


            int32 old_q_offset = *params.d_main_q_local_offset;
            int32 new_q_offset = *params.d_main_q_end;
            int32 new_q_end = new_q_offset;

            int32 total_narcs = *params.d_main_q_narcs;
    
            int32 old_q_size = new_q_offset - old_q_offset;  // move to end

            cutoff = *params.d_cutoff;

            // We'll switch queue at the beg of the loop
            // Cleaner that way - we need the offsets ready for
            // the global updates at the very end of this kernel
            new_q_offset = old_q_offset;

            bool first = true;

            while(old_q_size > 0) {
                // Step 0 : move queues        
                old_q_offset = new_q_offset;
                new_q_offset = new_q_end;

                if(!first) {
                    __syncthreads(); // old_q_ready
                    total_narcs = 0;

                    // Step 1 : compute_degrees
                    // TODO fuse 1 and 2
                    for(int32 q_idx = old_q_offset + threadIdx.x;
                            q_idx < new_q_offset; // = old_q_end
                            q_idx += blockDim.x) {

                        StateId state = params.d_main_q_state[q_idx];
                        BaseFloat cost = params.d_main_q_cost[q_idx];

                        int32 degree = 0;
                        if(cost < cutoff) {
                            BaseFloat best_cost = orderedIntToFloat(params.d_lookup[state]);

                            if(cost == best_cost) {
                                int32 start = d_arc_offsets[state];
                                int32 end = d_arc_offsets[state+1];
                                degree = end - start;
                                params.d_q_arc_offsets[q_idx] = start;
                            }
                        }

                        params.d_main_q_degrees_prefix_sum[q_idx] = degree;
                    }

                    __syncthreads(); // will be removed

                    // Step 2 : Scan

                    for(int32 block_off = 0;
                            block_off < old_q_size;
                            block_off += blockDim.x) {

                        int32 q_idx = old_q_offset + block_off + threadIdx.x;

                        int32 degree = (q_idx < new_q_offset) 
                            ? params.d_main_q_degrees_prefix_sum[q_idx]
                            : 0;
                        int32 lscan;
                        int32 total_in_blk;
                        BlockScan(temp_storage_scan).ExclusiveSum(degree, lscan, total_in_blk);
                        int32 scan = lscan + total_narcs;
                        total_narcs += total_in_blk;

                        if(q_idx < new_q_offset)
                            params.d_main_q_degrees_prefix_sum[q_idx] = scan;

                         __syncthreads(); // reusing temp_storage_scan + degrees ready
                    }


                } else {
                    first = false;    
                }


                // We already sync'ed

                // Step 3 : expand arcs

                for(int32 block_offset = 0;
                        block_offset < total_narcs;
                        block_offset += blockDim.x) {

                    int32 th_idx = block_offset + threadIdx.x;
                    bool valid_input = (th_idx < total_narcs);

                    BaseFloat total_cost = FLT_MAX;
                    int32 arc_idx;
                    StateId arc_next_state;
                    int32 q_idx;

                    if(valid_input) {
                        //we can do better than that
                        q_idx = binsearch_maxle(params.d_main_q_degrees_prefix_sum, th_idx, old_q_offset, new_q_offset-1); 

                        int32 lower_bound = params.d_main_q_degrees_prefix_sum[q_idx];
                        int32 arc_offset_start = params.d_q_arc_offsets[q_idx];

                        arc_idx = arc_offset_start + (th_idx - lower_bound);

                        arc_next_state = params.arc_nextstates[arc_idx];
                        BaseFloat arc_weight = params.arc_weights[arc_idx];
                        BaseFloat next_state_cost = orderedIntToFloat(params.d_lookup[arc_next_state]);
                        BaseFloat old_tok_cost = params.d_main_q_cost[q_idx];

                        total_cost = arc_weight + old_tok_cost;

                        if(total_cost >= next_state_cost) {
                            total_cost = FLT_MAX;
                            valid_input = false; 
                        } 
                    }

                    BaseFloat min_cost = BlockReduce(temp_storage_reduce).Reduce(total_cost, cub::Min());

                    if(threadIdx.x == 0) {
                        cutoff = GetCutoffCandidate(cutoff,
                                min_cost,
                                params.beam,
                                new_q_end,
                                params.q_capacity);
                    }

                    __syncthreads();

                    int32 has_successor = (total_cost < cutoff && valid_input) ? 1 : 0;

                    if(has_successor) 
                        atomicMin(&params.d_lookup[arc_next_state], floatToOrderedInt(total_cost));

                    int32 new_q_idx_block = has_successor;
                    int32 total_in_blk;
                    BlockScan(temp_storage_scan).ExclusiveSum(new_q_idx_block, new_q_idx_block, total_in_blk);

                    if((new_q_end + total_in_blk) >= params.q_capacity) {
                        *params.h_q_overflow = 1;
                        
                        goto finalize_kernel; // keeping things clean before aborting
                    }

                    if(has_successor) {
                        int32 new_q_index = new_q_end + new_q_idx_block;
                        params.d_main_q_state[new_q_index] = arc_next_state;

                        params.d_main_q_cost[new_q_index] = total_cost;

                        InfoToken new_tok_info;
                        new_tok_info.prev_token = params.main_q_global_offset + q_idx;

                        new_tok_info.arc_idx = arc_idx;
                        params.d_main_q_info[new_q_index] = new_tok_info;
                        
                        //print32f("new q index = %i (%i+%i) (tot=%i) \n", new_q_index, new_q_end, new_q_idx_block,
                        //total_in_blk);
                   }

                    new_q_end += total_in_blk;
                }

                old_q_size = new_q_end - new_q_offset; 
            }

            finalize_kernel:

            if(threadIdx.x == 0) {
                // Next step is ProcessEmitting of next frame, from is currToken_offset
                *params.d_main_q_end = new_q_end; 
                *params.d_main_q_narcs = 0;

                *params.h_main_q_end = new_q_end; 
                *params.h_main_q_narcs = 0; 

                *params.d_main_q_local_offset = 0; 
                *params.h_main_q_local_offset = 0; 

                *params.d_cutoff = cutoff;
            }

        }

    void CudaDecoder::NonEmittingLongTail(const uint32_t *d_arc_offsets, 
            const ExpandArcParams &params) {

        dim3 grid,block;
        block.x = KALDI_CUDA_DECODER_KERNEL_NONEM_LT_DIMX;
        grid.x = 1; // it is designed for the long tail
        _process_nonem_longtail<<<grid,block,0,compute_st_>>>(d_arc_offsets, params);
    }


} // end namespace kaldi
