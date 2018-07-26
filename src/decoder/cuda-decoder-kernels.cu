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
typedef CudaDecoder::QEndAndNarcs QEndAndNarcs;
typedef CudaDecoder::CostType CostType;
typedef CudaDecoder::PreprocessParams PreprocessParams; 
typedef CudaDecoder::ExpandArcParams ExpandArcParams; 


//
// Utils device function
//

    // Used to trigger the fire&forget version of atomicMin (only av for int/long)
    __device__ int floatToOrderedInt(float floatVal) {

        int intVal = __float_as_int( floatVal );

        return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
    }



    __device__ float orderedIntToFloat(int intVal) {

        return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF );

    } 
    __forceinline__ __device__ int binsearch_maxle(const int *vec, const int val, int low, int high) {
        while(true) {
            if(low == high)
                return low; //we know it exists
            if((low + 1) == high)
                return (vec[high] <= val) ? high : low;

            int mid = low + (high- low) / 2;

            if(vec[mid] > val)
                high = mid-1;
            else
                low = mid;
        }
    }


    // Temporary used for cutoff - will be removed
    __device__ float fatomicMin(float *addr, float value)

    {

        float old = *addr, assumed;
        if(old <= value) return old;

        do
        {
            assumed = old;
            old = atomicCAS((unsigned int*)addr,
                    __float_as_int(assumed),
                    __float_as_int(value));

        } while(old!=assumed);

        return old;

    }

//
// Kernels
//

    // Used before first frame
    __global__ void init_lookup_kernel(int *state_cost, int size) {
        for(int idx = blockIdx.x*blockDim.x + threadIdx.x;
                idx < size;
                idx += blockDim.x*gridDim.x) {
            state_cost[idx]  = floatToOrderedInt(FLT_MAX);
        }
    }

    void CudaDecoder::InitLookup() {
        int nstates = fst_.numStates;

        KALDI_ASSERT(nstates > 0);

        dim3 grid,block;
        block.x = 256;
        grid.x = DIV_ROUND_UP(nstates, block.x);

        init_lookup_kernel<<<grid,block>>>(d_state_cost, nstates);
    }

        // Used to reset lookup table between frames
    // Using the queue to reset only the values needed
    // Also takes care of resetting cutof
    // TODO rename to something like "ResetForNewFrame"
    __global__ void reset_lookup_kernel(StateId *d_main_q_state, const int *d_main_q_end, int *state_cost, CostType *d_cutoff) {
        int q_from_end = *d_main_q_end; 

        for(int idx = blockIdx.x*blockDim.x + threadIdx.x;
                idx < q_from_end;
                idx += blockDim.x*gridDim.x) {

            StateId state = d_main_q_state[idx];
            state_cost[state]  = floatToOrderedInt(FLT_MAX);
        }

        // Avoiding a kernel call just to reset the cutoff
        if(blockIdx.x == 0 && threadIdx.x == 0)
            *d_cutoff = FLT_MAX; 
    }

    void CudaDecoder::ResetLookup() {
        int size = *h_main_q_end;

        KALDI_ASSERT(size > 0);

        dim3 grid,block;
        block.x = 256;
        grid.x = DIV_ROUND_UP(size, block.x);

        reset_lookup_kernel<<<grid,block,0,compute_st>>>(d_main_q_state, d_main_q_end, d_state_cost, d_cutoff);
    }



    // TODO rename
    struct F2Sum {
        __device__ int2 operator()(const int2 &a, const int2 &b) const {
            int2 c;
            c.x = a.x + b.x;
            c.y = a.y + b.y;

            return c;
        }
    };

    /*
       This kernel preprocess the necessary information for expand (scan of the outgoing degrees) 
       and explicitly prune the tokens

       It contracts (by pruning) the queue list:
       raw output from aux_q ----contract----> pruned output in main q

       This kernel is responsible for :

       1) Read a token from the aux queue (raw output from previous expand)

       2) Compute the outgoing degree of that token.next_state. For that :
       -> If that token is suboptimal (cutoff, best_cost), we prune it
       -> Otherwise, we set degree using CSR graph

       3) We move the non-pruned tokens into the main q. After a local prefix sum,
       we request a spot using the main_q_end_and_narcs counter. 
       main_q_end_and_narcs.split.end contains the number of tokens in the main q until now
       main_q_end_and_narcs.split.narcs contains the number of arcs in the main q until now

       We also do the degrees scan in one pass using the maind_q_end_and_narcs.split.narcs

       This kernel is used before ProcessNonEmitting
    */

    __global__ void contract_and_preprocess_kernel(PreprocessParams params) {
        typedef cub::BlockScan<int2, KERNEL_PREPROCESS_DIMX> BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage;

        __shared__ QEndAndNarcs blk_local_offset_i2;
        __shared__ int total_in_CTA;

        const int aux_q_end = *params.d_aux_q_end;
        BaseFloat cutoff = *params.d_cutoff;

        for(int block_offset = blockDim.x*blockIdx.x;
                block_offset < aux_q_end;
                block_offset += gridDim.x*blockDim.x) {

            int aux_q_idx = block_offset + threadIdx.x;
            int degree = 0;
            int arc_start = -1;

            StateId state_idx;
            CostType cost;

            if(aux_q_idx < aux_q_end) {
                cost = params.d_aux_q_cost[aux_q_idx];
                state_idx = params.d_aux_q_state[aux_q_idx];

                BaseFloat best_cost = orderedIntToFloat(params.d_state_cost[state_idx]);

                if(cost < cutoff) {
                    if(cost == best_cost) {
                        arc_start = params.d_arc_offsets[state_idx];
                        int arc_end = params.d_arc_offsets[state_idx+1];
                        degree = arc_end - arc_start;
                    }
                }

                if (best_cost >= cutoff)
                    params.d_state_cost[state_idx] = floatToOrderedInt(FLT_MAX);

            }

            int is_pruned = (arc_start == -1);
            int2 scan_i2;
            scan_i2.x =  is_pruned ? 0 : 1;
            scan_i2.y =  degree;

            int2 zero_i2;
            zero_i2.x = zero_i2.y = 0;

            BlockScan(temp_storage).ExclusiveScan(scan_i2, scan_i2, zero_i2, F2Sum());

            
            QEndAndNarcs inclusive_scan;
            if(threadIdx.x == (KERNEL_PREPROCESS_DIMX-1)) {
                // CUB Scan is exclusive
                inclusive_scan.split.end = scan_i2.x + (is_pruned ? 0 : 1);
                inclusive_scan.split.narcs = scan_i2.y + degree;

                blk_local_offset_i2.both = atomicAdd(&params.d_main_q_end_and_narcs_i2->both, inclusive_scan.both);
                total_in_CTA = inclusive_scan.split.end;
            }

            __syncthreads(); // blk_local_offset + temp_storage

            // main_q overflow
            if((blk_local_offset_i2.split.end + total_in_CTA) >= params.q_capacity) {
                if(threadIdx.x == (KERNEL_PREPROCESS_DIMX-1)) {
                    atomicAdd(&params.d_main_q_end_and_narcs_i2->both, -inclusive_scan.both); // revert
                    *params.h_q_overflow = 1;
                }

                goto finalize_kernel; // keeping things clean before aborting
            }

            if(!is_pruned) {
                // Moving non-pruned to the main q
                int main_q_idx = blk_local_offset_i2.split.end + scan_i2.x;

                InfoToken info = params.d_aux_q_info[aux_q_idx];

                params.d_main_q_state[main_q_idx] = state_idx;
                params.d_main_q_cost[main_q_idx] = cost;
                params.d_main_q_info[main_q_idx] = info;

                params.d_degrees_scan[main_q_idx] = blk_local_offset_i2.split.narcs + scan_i2.y;

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
            int old = atomicAdd(params.d_n_CTA_done, 1);
            bool is_last_CTA = (old == (gridDim.x -1));

            if(is_last_CTA) {
                __threadfence();

                // Avoid a mem copy
                *params.h_main_q_end = *params.d_main_q_end; // pinned memory
                *params.h_main_q_narcs = *params.d_main_q_narcs; // pinned memory
                *params.d_aux_q_end = 0; // we flushed the aux q
                *params.h_aux_q_end = 0; 

                *params.d_n_CTA_done = 0;
                *params.d_aux_q_end = 0; // we flushed the aux q

            }
        }

    }



/*
    This kernel is also a preprocessing kernel, but this time does it in place
    The tokens are already in the main q (they were placed here by a previous "contract and preprocess").
    We avoid performing the next phase on non-optimal ones by setting the degree to 0 and
    computing a degrees scan.

    Here we have to do the scan in two passes : the scan will be finished in "finalize_preprocess"

    This preprocess step is used in ProcessEmitting. Tokens were placed in main_q by
    the ProcessNonEmitting of the previous frame. We cannot renumber them (it would break
    the prev_token index). We preprocess in place, leaving things as they are in main_q

*/

    __global__ void preprocess_in_place_kernel(PreprocessParams params) {
    
        typedef cub::BlockScan<int, KERNEL_PREPROCESS_DIMX> BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage;

        __shared__ int is_last_CTA;

        int queue_offset = *params.d_main_q_local_offset;
        int queue_end = *params.d_main_q_end;
        int queue_size = queue_end - queue_offset;

        BaseFloat cutoff = *params.d_cutoff;

        for(int block_offset = blockDim.x*blockIdx.x;
                block_offset < queue_size;
                block_offset += gridDim.x*blockDim.x)
        {
            int idx = queue_offset + block_offset + threadIdx.x; 
            int degree = 0; 
            if(idx < queue_end) {
                StateId state_idx = params.d_main_q_state[idx]; 
                BaseFloat cost = params.d_main_q_cost[idx];

                if(cost < cutoff) {
                    BaseFloat best_cost = orderedIntToFloat(params.d_state_cost[state_idx]); 
                    if(cost == best_cost) {
                        int start = params.d_arc_offsets[state_idx]; 
                        int end = params.d_arc_offsets[state_idx+1]; 
                        degree  = end - start;
                        params.d_main_q_arc_offsets[idx] = start;
                    }
                }
            }

            int scan;
            BlockScan(temp_storage).ExclusiveSum(degree, scan);
            if(idx < queue_end) 
                params.d_degrees_scan[idx] = scan;


            if(threadIdx.x == (KERNEL_PREPROCESS_DIMX-1))
                params.d_degrees_block_scan[block_offset/KERNEL_PREPROCESS_DIMX] = (scan + degree); 

            __syncthreads(); // we'll reuse temp_storage
        }

        if(threadIdx.x == 0) {
            int old = atomicAdd(params.d_n_CTA_done, 1); 
            is_last_CTA = (old == (gridDim.x -1));
        }

        // is_last_CTA + temp_storage reuse
        __syncthreads();
        
        if(is_last_CTA)
        {
            // The last block alive takes care of scan of block sums 
            __threadfence();

            if(threadIdx.x == 0) {
                *params.d_n_CTA_done = 0;
            }

            // following value can be different than gridDim.x 
            int total_blk_val = (queue_size + KERNEL_PREPROCESS_DIMX -1) / KERNEL_PREPROCESS_DIMX;
            int scan_offset = 0;

            for(int blk_idx_off = 0; blk_idx_off < total_blk_val; blk_idx_off += blockDim.x) {
                int blk_idx = blk_idx_off + threadIdx.x; 

                int blk_sum = (blk_idx < total_blk_val) ?  params.d_degrees_block_scan[blk_idx] : 0; 
                int blk_scan, iteration_total;
                BlockScan(temp_storage).ExclusiveSum(blk_sum, blk_scan, iteration_total);
                blk_scan += scan_offset;
                scan_offset += iteration_total;

                if(blk_idx < total_blk_val) {
                    params.d_degrees_block_scan[blk_idx] = blk_scan;
                }

                // temp storage
                __syncthreads();
            }

            if(threadIdx.x == 0)
            {
                *params.d_main_q_narcs = scan_offset; 
                *params.h_main_q_narcs = scan_offset; // pinned memory
            }
        }
    }


    void CudaDecoder::ContractAndPreprocess(PreprocessParams &params) {
        dim3 grid,block;
        block.x = KERNEL_PREPROCESS_DIMX;
        grid.x = DIV_ROUND_UP(*h_aux_q_end, block.x);

        // We can have grid.x == 0 and still have a valid execution
        if(grid.x)
            contract_and_preprocess_kernel<<<grid,block,0,compute_st>>>(params);
    }


    void CudaDecoder::PreprocessInPlace(PreprocessParams &params) {
        dim3 grid,block;
        block.x = KERNEL_PREPROCESS_DIMX;
        int main_q_size = *h_main_q_end - *h_main_q_local_offset;

        grid.x = DIV_ROUND_UP(main_q_size, block.x);

        // If the main_q is empty, we will not be able to continue
        KALDI_ASSERT(grid.x > 0);

        preprocess_in_place_kernel<<<grid,block,0,compute_st>>>(params);
    }



    /*

       Part 2 of the scan for "PreprocessEmitting". For NonEmitting scan is already final

       Computes global prefix sum with block prefix sum and block offsets

       If we want to speed up expand, we can compute lower and upper bound to restrain 
       the binary search in expand
       This can be done on the fly here, and removes main bottleneck of expand
       Not done for now, because expand is fast enough

     */
    __global__ void finalize_degrees_scan_kernel(int *d_scan, int *d_blk_scan, const int *d_main_q_local_offset, const int
            *d_main_q_end) {

        int q_off = *d_main_q_local_offset;
        int q_end = *d_main_q_end;
        int q_size = q_end - q_off;

        for(int idx = q_off + blockDim.x*blockIdx.x + threadIdx.x;
                idx < q_size;
                idx += blockDim.x*gridDim.x) {

            int blk_idx = (idx - q_off) / KERNEL_PREPROCESS_DIMX;
            int blk_scan_offset = d_blk_scan[blk_idx]; // we rely on L1 for this one, avoiding syncs

            d_scan[idx] += blk_scan_offset;
        }

    }

    void CudaDecoder::FinalizePreprocessInPlace() {
        dim3 grid,block;
        block.x = KERNEL_PREPROCESS_DIMX;
        int main_q_size = *h_main_q_end - *h_main_q_local_offset;
        grid.x = DIV_ROUND_UP(main_q_size, block.x);

        // If the main_q is empty, we will not be able to continue
        KALDI_ASSERT(grid.x > 0);

        finalize_degrees_scan_kernel<<<grid,block,0,compute_st>>>(d_degrees_scan, d_degrees_block_scan, d_main_q_local_offset,
                d_main_q_end); 
    }




    /*
       This kernel propagates arcs from the main q [main_q_local_offset, main_q_end[
       to the aux

       The main bottleneck is the first binary search. 
       If we want to remove it, preprocess it on the fly in preprocess

     */

    struct CostTInt {
        CostType cost;
        int i;
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
                                const int q_size,
                                const int q_capacity) {
                                 

    // Doing something simple for now
    // We have to keep beam large enough,
    // the final cutoff will be used for the final
    // prune. If it is too small, we won't keep enough tokens

   CostType beam = default_beam;

   if(q_size >= q_capacity/2) 
       beam /= 2;

    return fmin(current_cutoff, min_cost + beam);
}


    void __global__ expand_arcs_kernel(ExpandArcParams params) {
        typedef cub::BlockScan<CostTInt, KERNEL_EXPAND_ARCS_DIMX> BlockScan;

        __shared__ typename BlockScan::TempStorage temp_storage_scan;

        __shared__ int to_q_block_offset;
        __shared__ CostType blk_cutoff;

        const int total_narcs = *params.d_main_q_narcs;
        const int main_q_offset = *params.d_main_q_local_offset;
        const int main_q_end = *params.d_main_q_end;

        
        if(threadIdx.x == 0) {
            blk_cutoff = *params.d_cutoff;
        }

        __syncthreads();

        // Keeping the whole CTA alive, we'll have syncs
        for(int block_offset = blockDim.x*blockIdx.x;
                block_offset < total_narcs;
                block_offset += gridDim.x*blockDim.x) {

            int th_idx = block_offset + threadIdx.x;
            bool valid_input = (th_idx < total_narcs);

            BaseFloat total_cost = FLT_MAX;
            int arc_idx;
            StateId arc_next_state;
            int main_q_idx;

            if(valid_input) {
                //we can do better than that
                main_q_idx = binsearch_maxle(params.d_degrees_scan, th_idx, main_q_offset, main_q_end-1); 

                int lower_bound = params.d_degrees_scan[main_q_idx];
                int arc_offset_start = params.d_q_arc_offsets[main_q_idx];

                arc_idx = arc_offset_start + (block_offset + threadIdx.x - lower_bound);
                arc_next_state = params.arc_nextstates[arc_idx];

                total_cost = params.arc_weights[arc_idx];

                int arc_ilabel = params.is_emitting ? params.arc_ilabels[arc_idx] : 0;
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

                            int has_successor = valid_input ? 1 : 0;  // Need a spot in the new q
                            CostTInt ci;
                            ci.cost = valid_input ? total_cost : FLT_MAX; 
                            ci.i = has_successor;

                            BlockScan(temp_storage_scan).InclusiveScan(ci, ci, CISum());

                            if(threadIdx.x == (KERNEL_EXPAND_ARCS_DIMX - 1)) {
                                int total_successors_in_block = ci.i;
                                to_q_block_offset = atomicAdd(params.d_aux_q_end, total_successors_in_block);
                                if((to_q_block_offset + total_successors_in_block) >= params.q_capacity) {
                                    to_q_block_offset = params.q_capacity; // used to broadcast the info

                                }
                                /*
                                
                                GetCutoffCandidate takes into account the current value of 
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
                                if(threadIdx.x == (KERNEL_EXPAND_ARCS_DIMX - 1)) {
                                    // Revert
                                    int total_successors_in_block = ci.i;
                                    atomicAdd(params.d_aux_q_end, -total_successors_in_block); 
                                    *params.h_q_overflow = 1; 
                                }

                                goto finalize_kernel; // keeping things clean before aborting
                            }

                            ci.i -= has_successor; // we want the exclusive sum now
                            int to_q_index = to_q_block_offset + ci.i;

                            if(has_successor) {
                                params.d_aux_q_cost[to_q_index] = total_cost;
                                params.d_aux_q_state[to_q_index] = arc_next_state;
                                
                                atomicMin(&params.d_lookup[arc_next_state],
                                floatToOrderedInt(total_cost)
                                );

                                //printf("cost = %f, cutoff = %f, beam=%f \n", total_cost, blk_cutoff, params.beam);
                                if(total_cost < blk_cutoff) { // cutoff may have changed
                                    // We write the rest of the token only if necessary
                                    // if the cost is higher than cutoff, 
                                    // the token will be ignored anyway 


                                    InfoToken new_tok_info;
                                    new_tok_info.prev_token = params.main_q_global_offset + main_q_idx;
                                    new_tok_info.arc_idx = arc_idx;
                            

                                    params.d_aux_q_info[to_q_index] = new_tok_info;

                                    /*
                                    printf("expand, adding %i (%i)  -> %i \n", new_tok_info.prev_token,
                                    params.main_q_global_offset, arc_next_state);
                                    */
                                }
                            }
        }

        finalize_kernel:

        __syncthreads(); // avoiding races on d_main_q_narcs for instance

        // Last block alive sets h_aux_q_end (pinned memory)
        if(threadIdx.x == 0) {
            int old = atomicAdd(params.d_n_CTA_done, 1);
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

    void CudaDecoder::ExpandArcs(int nthreads, const ExpandArcParams &params) {
        dim3 grid,block;
        block.x = 256;
        grid.x = DIV_ROUND_UP(nthreads, block.x);

        // It's possible to have zero threads and still be valid
        if(grid.x > 0)
            expand_arcs_kernel<<<grid,block,0,compute_st>>>(params);
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


    __launch_bounds__(KERNEL_NONEM_LT_DIMX, 1)
        __global__ void process_nonem_longtail(unsigned int *d_arc_offsets, 
                ExpandArcParams params) {

            typedef cub::BlockScan<int, KERNEL_NONEM_LT_DIMX> BlockScan;
            typedef cub::BlockReduce<float, KERNEL_NONEM_LT_DIMX> BlockReduce;

            __shared__ typename BlockScan::TempStorage temp_storage_scan;
            __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

            __shared__ BaseFloat cutoff;


            int old_q_offset = *params.d_main_q_local_offset;
            int new_q_offset = *params.d_main_q_end;
            int new_q_end = new_q_offset;

            int total_narcs = *params.d_main_q_narcs;
    
            int old_q_size = new_q_offset - old_q_offset;  // move to end

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
                    for(int q_idx = old_q_offset + threadIdx.x;
                            q_idx < new_q_offset; // = old_q_end
                            q_idx += blockDim.x) {

                        StateId state = params.d_main_q_state[q_idx];
                        BaseFloat cost = params.d_main_q_cost[q_idx];

                        int degree = 0;
                        if(cost < cutoff) {
                            BaseFloat best_cost = orderedIntToFloat(params.d_lookup[state]);

                            if(cost == best_cost) {
                                int start = d_arc_offsets[state];
                                int end = d_arc_offsets[state+1];
                                degree = end - start;
                                params.d_q_arc_offsets[q_idx] = start;
                            }
                        }

                        params.d_degrees_scan[q_idx] = degree;
                    }

                    __syncthreads(); // will be removed

                    // Step 2 : Scan

                    for(int block_off = 0;
                            block_off < old_q_size;
                            block_off += blockDim.x) {

                        int q_idx = old_q_offset + block_off + threadIdx.x;

                        int degree = (q_idx < new_q_offset) 
                            ? params.d_degrees_scan[q_idx]
                            : 0;
                        int lscan;
                        int total_in_blk;
                        BlockScan(temp_storage_scan).ExclusiveSum(degree, lscan, total_in_blk);
                        int scan = lscan + total_narcs;
                        total_narcs += total_in_blk;

                        if(q_idx < new_q_offset)
                            params.d_degrees_scan[q_idx] = scan;

                         __syncthreads(); // reusing temp_storage_scan + degrees ready
                    }


                } else {
                    first = false;    
                }


                // We already sync'ed

                // Step 3 : expand arcs

                for(int block_offset = 0;
                        block_offset < total_narcs;
                        block_offset += blockDim.x) {

                    int th_idx = block_offset + threadIdx.x;
                    bool valid_input = (th_idx < total_narcs);

                    BaseFloat total_cost = FLT_MAX;
                    int arc_idx;
                    StateId arc_next_state;
                    int q_idx;

                    if(valid_input) {
                        //we can do better than that
                        q_idx = binsearch_maxle(params.d_degrees_scan, th_idx, old_q_offset, new_q_offset-1); 

                        int lower_bound = params.d_degrees_scan[q_idx];
                        int arc_offset_start = params.d_q_arc_offsets[q_idx];

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

                    int has_successor = (total_cost < cutoff && valid_input) ? 1 : 0;

                    if(has_successor) 
                        atomicMin(&params.d_lookup[arc_next_state], floatToOrderedInt(total_cost));

                    int new_q_idx_block = has_successor;
                    int total_in_blk;
                    BlockScan(temp_storage_scan).ExclusiveSum(new_q_idx_block, new_q_idx_block, total_in_blk);

                    if((new_q_end + total_in_blk) >= params.q_capacity) {
                        *params.h_q_overflow = 1;
                        
                        goto finalize_kernel; // keeping things clean before aborting
                    }

                    if(has_successor) {
                        int new_q_index = new_q_end + new_q_idx_block;
                        params.d_main_q_state[new_q_index] = arc_next_state;

                        params.d_main_q_cost[new_q_index] = total_cost;

                        InfoToken new_tok_info;
                        new_tok_info.prev_token = params.main_q_global_offset + q_idx;

                        new_tok_info.arc_idx = arc_idx;
                        params.d_main_q_info[new_q_index] = new_tok_info;
                        
                        //printf("new q index = %i (%i+%i) (tot=%i) \n", new_q_index, new_q_end, new_q_idx_block,
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

    void CudaDecoder::NonEmittingLongTail(unsigned int *d_arc_offsets, 
            const ExpandArcParams &params) {

        dim3 grid,block;
        block.x = KERNEL_NONEM_LT_DIMX;
        grid.x = 1; // it is designed for the long tail
        process_nonem_longtail<<<grid,block,0,compute_st>>>(d_arc_offsets, params);
    }


} // end namespace kaldi
