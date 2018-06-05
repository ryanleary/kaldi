// decoder/simple-decoder.cc

// Copyright 2009-2011 Microsoft Corporation
//           2012-2013 Johns Hopkins University (author: Daniel Povey)

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

#include "decoder/cuda-decoder.h"
#include "fstext/remove-eps-local.h"
#include <algorithm>
#include <nvToolsExt.h>
#include <cuda_runtime_api.h>
#include <float.h>
#include <math.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#define MEMADVISE

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

#define DIV_ROUND_UP(a,b) ((a+b-1)/b)
namespace kaldi {

  /***************************************CudaFst Implementation*****************************************/
  HOST DEVICE inline float CudaFst::Final(StateId state) const {
    #ifdef __CUDA_ARCH__
    return final_d[state];
    #else
    return final_h[state];
    #endif

  }
  void CudaFst::initialize(const fst::Fst<StdArc> &fst) {
    nvtxRangePushA("CudaFst constructor");
    bytes_cudaMalloc=0;
    //count states since Fst doesn't provide this functionality
    numStates=0;
    for( fst::StateIterator<fst::Fst<StdArc> > iter(fst); !iter.Done(); iter.Next()) {
      numStates++;
    }
    start=fst.Start();
    cudaMallocHost(&final_h,sizeof(float)*numStates);
    cudaMalloc(&final_d,sizeof(float)*numStates);

    //allocate and initialize offset arrays
    e_offsets_h=(unsigned int *)malloc(sizeof(unsigned int)*(numStates+1));
    ne_offsets_h=(unsigned int *)malloc(sizeof(unsigned int)*(numStates+1));

    cudaMalloc((void**)&e_offsets_d,sizeof(unsigned int)*(numStates+1)); bytes_cudaMalloc+=sizeof(unsigned int)*(numStates+1);
    cudaMalloc((void**)&ne_offsets_d,sizeof(unsigned int)*(numStates+1)); bytes_cudaMalloc+=sizeof(unsigned int)*(numStates+1);

    memset(e_offsets_h,0,sizeof(unsigned int)*(numStates+1));
    memset(ne_offsets_h,0,sizeof(unsigned int)*(numStates+1));

    //iterate through states and arcs and count number of arcs per state
    e_count=0;
    ne_count=0;
    max_ilabel=0;

    for(int i=0;i<numStates;i++) {
      final_h[i]=fst.Final(i).Value();
      //count emmiting and non_emitting arcs
      for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done(); aiter.Next()) {
        StdArc arc = aiter.Value();
        int32 ilabel = arc.ilabel;
        int32 olabel = arc.olabel;

        if(ilabel>max_ilabel) {
          max_ilabel=ilabel;
        }

        if(ilabel!=0) { //emitting
          e_count++;
        } else { //non-emitting
          ne_count++;
        }
      }
      ne_offsets_h[i+1]=ne_count;
      e_offsets_h[i+1]=e_count;
    }

    //offset ne_offsets by the number of emitting arcs
    for(int i=0;i<numStates+1;i++) {
      e_offsets_h[i]+=1;          //add dummy arc at the beginingg.
      ne_offsets_h[i]+=e_count+1;   //add dummy arc and put e_arcs before
    }

    arc_count=e_count+ne_count+1;

    cudaMemcpy(final_d,final_h,sizeof(float)*numStates,cudaMemcpyHostToDevice);
    
    cudaMemcpy(e_offsets_d,e_offsets_h,sizeof(unsigned int)*(numStates+1),cudaMemcpyHostToDevice);
    cudaMemcpy(ne_offsets_d,ne_offsets_h,sizeof(unsigned int)*(numStates+1),cudaMemcpyHostToDevice);


    //Allocate non-zero arrays
    cudaMallocHost(&arc_weights_h,arc_count*sizeof(BaseFloat));
    cudaMallocHost(&arc_nextstates_h,arc_count*sizeof(StateId));
    cudaMallocHost(&arc_ilabels_h,arc_count*sizeof(int32));
    cudaMallocHost(&arc_olabels_h,arc_count*sizeof(int32));

    cudaMalloc((void**)&arc_weights_d,arc_count*sizeof(BaseFloat));
    cudaMalloc((void**)&arc_nextstates_d,arc_count*sizeof(StateId));
    cudaMalloc((void**)&arc_ilabels_d,arc_count*sizeof(int32)); 

        //now populate arc data
    int e_idx=1;          //save room for dummy arc (so start at 1)
    int ne_idx=e_count+1; //starts where e_offsets ends

    //create dummy arc
    arc_weights_h[0]=StdWeight::One().Value();
    arc_nextstates_h[0]=fst.Start();
    arc_ilabels_h[0]=0;
    arc_olabels_h[0]=0;

    for(int i=0;i<numStates;i++) {
      //count emiting and non_emitting arcs

      for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done(); aiter.Next()) {
        StdArc arc = aiter.Value();
        int idx;
        if(arc.ilabel!=0) { //emitting
          idx=e_idx++;
        } else {
          idx=ne_idx++;
        }
        arc_weights_h[idx]=arc.weight.Value();
        arc_nextstates_h[idx]=arc.nextstate;
        arc_ilabels_h[idx]=arc.ilabel;
        arc_olabels_h[idx]=arc.olabel;
      }
    }

    cudaMemcpy(arc_weights_d,arc_weights_h,arc_count*sizeof(BaseFloat),cudaMemcpyHostToDevice);
    cudaMemcpy(arc_nextstates_d,arc_nextstates_h,arc_count*sizeof(StateId),cudaMemcpyHostToDevice);
    cudaMemcpy(arc_ilabels_d,arc_ilabels_h, arc_count*sizeof(int32),cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaCheckError();

    nvtxRangePop();
  }

  void CudaFst::finalize() {
    nvtxRangePushA("CudaFst destructor");
    cudaFreeHost(final_h);
    cudaFree(final_d);
    free(e_offsets_h);
    free(ne_offsets_h);

    cudaFree(e_offsets_d);
    cudaFree(ne_offsets_d);

    cudaFreeHost(arc_weights_h);
    cudaFreeHost(arc_nextstates_h);
    cudaFreeHost(arc_ilabels_h);
    cudaFreeHost(arc_olabels_h);

    cudaFree(arc_weights_d);
    cudaFree(arc_nextstates_d);
    cudaFree(arc_ilabels_d);
    nvtxRangePop();
  }

  /***************************************End CudaFst****************************************************/

  CudaDecoder::CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config): fst_(fst), beam_(config.beam),
  bytes_cudaMalloc(0), max_tokens(config.max_tokens) {
    printf("CudaDecoder2 Constructor\n");

    int max_token = config.max_tokens; // for CUB

    // Comments about variables are in the .h file

    cudaStreamCreate(&compute_st);

    cudaEventCreate(&q_token_from_narcs_evt);

    cudaMalloc(&d_curr_token, sizeof(int));
    cudaMalloc(&d_q_token_from, sizeof(int));
    cudaMalloc(&d_q_token_to, sizeof(int));
    cudaMalloc(&d_q_token_end, sizeof(int));

    cudaMalloc(&d_q_token_from_narcs, sizeof(int));
  
    cudaMalloc(&d_allToken, config.max_tokens * sizeof(StateId));
    cudaMalloc(&d_allTokenInfo, config.max_tokens * sizeof(InfoToken));

    cudaMallocHost(&h_q_token_from_size, sizeof(int));  

    // TODO move back to params
    int max_token_frame = 5000000;
    // we could use same pointer
    cudaMalloc(&d_degrees_scan, max_token_frame * sizeof(int));
    cudaMalloc(&d_block_sums_scan, (max_token_frame / 256 + 2)* sizeof(int)); // TODO remove hardcoded
    cudaMalloc(&d_q_arc_offset, max_token_frame * sizeof(int));

    cudaMalloc(&loglikelihoods_d, sizeof(BaseFloat)*(fst_.max_ilabel+1));  

    cudaMalloc(&d_state_cost,sizeof(BaseFloat)*fst_.numStates);

    cudaMallocHost(&h_reached_final, sizeof(int));
    cudaMallocHost(&h_q_token_from_narcs, sizeof(int));

    // TODO use directly pinned, no device mem
    // TODO hardcoded params
    cudaMalloc(&d_reversed_path, 50000 * sizeof(int)); // TODO pinned
    h_reversed_path = (int*)malloc(50000 * sizeof(int));

    cudaMalloc(&d_cutoff, sizeof(float));
    
    cudaMalloc(&d_path_size, sizeof(int));
    cudaMalloc(&d_n_CTA_done, sizeof(int));

    cudaCheckError();
  }

  CudaDecoder::~CudaDecoder() {
        printf("CUDA DECODER DESTRUCTOR\n");
      // TODO
  }

  void CudaDecoder::InitDecoding() {
    printf("CUDA DECODER InitDecoding\n");


    InitLookup();

    StateId start_state = fst_.Start();
    KALDI_ASSERT(start_state != fst::kNoStateId);

    cudaCheckError();
    InfoToken it_init;
    it_init.cost = StdWeight::One().Value();
    it_init.prev_token = INT_MIN;
    it_init.arc_idx = -1;

    cudaMemcpy(d_allToken, &start_state, sizeof(StateId), cudaMemcpyHostToDevice);
    cudaMemcpy(d_allTokenInfo, &it_init, sizeof(InfoToken), cudaMemcpyHostToDevice);

    // We simulate a regular execution for the first iteration
    cudaMemcpy(&d_state_cost[start_state], &(it_init.cost), sizeof(BaseFloat), cudaMemcpyHostToDevice);

    cudaMemset(d_curr_token, 0, sizeof(int));
    cudaMemset(d_q_token_from, 0, sizeof(int));

    // Init state is in queue
    int one = 1;
    cudaMemcpy(d_q_token_to, &one, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_token_end, &one, sizeof(int), cudaMemcpyHostToDevice);
    *h_q_token_from_size = 1;

    float cutoff = FLT_MAX;
    cudaMemcpy(d_cutoff, &cutoff, sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_n_CTA_done, 0, sizeof(int));
    
    cudaCheckError();

    debug_max_narcs = 0;
    num_frames_decoded_ = 0;

    printf("CUDA DECODER InitDecoding 1/2\n");
    ProcessNonemitting();
    printf("CUDA DECODER InitDecoding 2/2\n");
 }


// Used to trigger the fire&forget version of atomicMin (only av for int/long)
__device__ int floatToOrderedInt(float floatVal) {

    int intVal = __float_as_int( floatVal );

    return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;

}



__device__ float orderedIntToFloat(int intVal) {

    return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF );

} 


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


    dim3 grid,block;
    block.x = 256;
    grid.x = DIV_ROUND_UP(nstates, block.x);

    init_lookup_kernel<<<grid,block,0,cudaStreamPerThread>>>(d_state_cost, nstates);
}

typedef CudaDecoder::StateId StateId;

// Used to reset lookup table between frames
// Using the queue to reset only the values needed
// Also takes care of resetting cutof
// TODO rename to something like "ResetForNewFrame"
__global__ void reset_lookup_kernel(StateId *d_q, int *d_q_offset, int *d_q_end, int *state_cost, float *d_cutoff) {
    int q_offset = *d_q_offset;
    int q_end = *d_q_end; 

    for(int idx = q_offset + blockIdx.x*blockDim.x + threadIdx.x;
            idx < q_end;
            idx += blockDim.x*gridDim.x) {

        StateId state = d_q[idx];

        state_cost[state]  = floatToOrderedInt(FLT_MAX);
    }

    // Avoiding a kernel call just to reset the cutoff
    if(blockIdx.x == 0 && threadIdx.x == 0)
        *d_cutoff = FLT_MAX; 
}

void CudaDecoder::ResetLookup() {
    int size = *h_q_token_from_size;

    dim3 grid,block;
    block.x = 256;
    grid.x = DIV_ROUND_UP(size, block.x);

    reset_lookup_kernel<<<grid,block,0,compute_st>>>(d_allToken, d_q_token_from, d_q_token_to, d_state_cost, d_cutoff);
}


void CudaDecoder::AdvanceDecoding(DecodableInterface *decodable,
        int32 max_num_frames) {
    printf("AdvanceDecoding\n");

    KALDI_ASSERT(num_frames_decoded_ >= 0 &&
        "You must call InitDecoding() before AdvanceDecoding()");
    int32 num_frames_ready = decodable->NumFramesReady();

    // num_frames_ready must be >= num_frames_decoded, or else
    // the number of frames ready must have decreased (which doesn't
    // make sense) or the decodable object changed between calls
    // (which isn't allowed).
    KALDI_ASSERT(num_frames_ready >= num_frames_decoded_);
    int32 target_frames_decoded = num_frames_ready;
    if (max_num_frames >= 0)
      target_frames_decoded = std::min(target_frames_decoded,
          num_frames_decoded_ + max_num_frames);

    ComputeLogLikelihoods(decodable);

    while (num_frames_decoded_ < target_frames_decoded) {
        //KALDI_LOG << "New frame";

        num_frames_decoded_++; 
        ComputeLogLikelihoods(decodable);

        //KALDI_LOG << "Emitting, frame=" << num_frames_decoded_;
        ProcessEmitting();

        //KALDI_LOG << "Non Emitting";
        ProcessNonemitting(); 


        if(num_frames_decoded_ > 3) {
            //KALDI_ASSERT(0); 
        }

        //computes log likelihoods for the next frame - check order
    }   


    printf("AdvanceDecoding Done\n");
    nvtxRangePop();
}


  void CudaDecoder::ComputeLogLikelihoods(DecodableInterface *decodable) {

    int32 frame = num_frames_decoded_;

    decodable->ComputeLogLikelihoods(loglikelihoods_d,frame,fst_.max_ilabel+1, compute_st);
  }


// Below that value, we launch the persistent kernel for NonEmitting
#define NONEM_LT_MAX_NARCS 4096
bool CudaDecoder::ProcessToken(unsigned int *d_arc_offsets,
                        bool is_emitting) {


    // Compute degrees, reduce by key, apply cutoff
    // Compute first part of the prefix sums of the degrees
    // At the end of that step, the kernel
    // set the value of h_q_token_from_narcs
    // (the number of arcs in the current queue processed)
    // TODO rename to something more explicit
    ComputeDegrees(d_arc_offsets);
    
    // Recording an event to signal h_q_token_from_narcs 
    // as ready to use 
    cudaEventRecord(q_token_from_narcs_evt, compute_st);
            cudaCheckError();

    // last time we use the lookup for old_q is in compute degrees
    if(is_emitting)
        ResetLookup();

    // Finalize the scan 
    // partial scans + block offsets -> global scan
    // If we want to speed up the binary search in expand
    // This is where we can compute lower and upper bound 
    // on the fly
    FinalizeDegreesScan();
    
    // We need h_q_token_from_narcs to be ready
    cudaEventSynchronize(q_token_from_narcs_evt);
    int h_old_q_narcs = *h_q_token_from_narcs;

    ExpandArcParams params;

    params.d_q = d_allToken; 
    params.d_q_info = d_allTokenInfo;

    params.d_q_token_from = d_q_token_from;
    params.d_q_token_to = d_q_token_to;
    params.d_q_token_end = d_q_token_end;

    params.d_degrees_scan = d_degrees_scan; 

    params.d_q_arc_offsets = d_q_arc_offset;
    params.arc_ilabels = fst_.arc_ilabels_d;
    params.d_q_token_from_narcs = d_q_token_from_narcs;
 
    params.arc_weights = fst_.arc_weights_d; 
    params.arc_nextstates = fst_.arc_nextstates_d; 
    params.d_cutoff = d_cutoff;
    params.beam = beam_;
    params.d_loglikelihoods= loglikelihoods_d;
    params.d_lookup = d_state_cost;
    params.is_emitting = is_emitting;

    params.d_curr_token = d_curr_token;
    params.h_q_token_from_size = h_q_token_from_size;
    params.d_n_CTA_done = d_n_CTA_done;

    bool done = false;

    if(h_old_q_narcs) {
        if(!params.is_emitting 
            && h_old_q_narcs < NONEM_LT_MAX_NARCS) { 
            NonEmittingLongTail(d_arc_offsets, params); 

            cudaCheckError();

            // Persistent kernel finishes the job
            done = true;
        }
        else {
            ExpandArcs(h_old_q_narcs, params);
        }

        cudaStreamSynchronize(compute_st); 
    }

    cudaCheckError();
    return done;
}


void CudaDecoder::ProcessEmitting() {
    nvtxRangePushA("ProcessEmitting");
    
    // Using emitting arc offsets
    ProcessToken(fst_.e_offsets_d, true); 

    cudaCheckError();
    nvtxRangePop();
}

  void CudaDecoder::ProcessNonemitting() {
    nvtxRangePushA("ProcessNonemitting");

    // While not done, call it
    while(!ProcessToken(fst_.ne_offsets_d, false));

    cudaCheckError();
    nvtxRangePop();
  }


// TODO use struct for params, 
// large # of args slow things down

/*

This kernel is responsible for :

1) Read a token from the input queue [from, to[
2) Compute the outgoing degree of that token.next_state. For that :
   -> If that token is suboptimal (cutoff, best_cost), degree = 0
   -> Otherwise, we set degree using CSR graph

The distinction between emitting / non emitting depends on the argument passed
as "d_q_arc_offset"

3) Compute prefix sums of those degrees within the block :
    -> We store those "local prefix sums" in d_degrees_scan. Another kernel will finish the job
    -> We save the sum of all degrees in that block (block_sums)

4) The last block alive compute the prefix sums of block_sums. 
    -> We save it, it will be needed to compute global_scan
    -> We now have the total number of arcs overall, we save it to h_q_token_from_narcs

*/

#define COMPUTE_DEGREES_DIMX 256
  __global__ void compute_degrees_kernel(StateId *d_q, InfoToken *d_q_info, const int *d_q_token_from, const int
  *d_q_token_to, int *d_degrees_scan, unsigned int
  *d_offsets, int *d_state_cost, BaseFloat *d_cutoff, int *d_q_arc_offset,
  int *d_block_sums, int *d_block_sums_scan, int *h_q_token_from_narcs, int *d_q_token_from_narcs, int *d_n_CTA_done) {

       typedef cub::BlockScan<int, COMPUTE_DEGREES_DIMX> BlockScan;
       __shared__ typename BlockScan::TempStorage temp_storage;

       __shared__ int blk_scan_offset;
       __shared__ int is_last_CTA;


        int queue_offset = *d_q_token_from;
        int queue_end = *d_q_token_to;
        int queue_size = queue_end - queue_offset;

        BaseFloat cutoff = *d_cutoff;

        for(int block_offset = blockDim.x*blockIdx.x;
                block_offset < queue_size;
                block_offset += gridDim.x*blockDim.x) {
            int idx = queue_offset + block_offset + threadIdx.x;
            int degree = 0;

            if(idx < queue_end) {

                StateId state_idx = d_q[idx];
                BaseFloat cost = d_q_info[idx].cost;

                if(cost < cutoff) {
                    BaseFloat best_cost = orderedIntToFloat(d_state_cost[state_idx]);
                    if(cost == best_cost) {
                        int start = d_offsets[state_idx];
                        int end = d_offsets[state_idx+1];
                        degree = end - start;
                        d_q_arc_offset[idx-queue_offset] = start;
                    }
                }
            }

            int scan;
            BlockScan(temp_storage).ExclusiveSum(degree, scan);

            if(idx < queue_end)
                d_degrees_scan[idx-queue_offset] = scan;

            if(threadIdx.x == (COMPUTE_DEGREES_DIMX-1)) {
                d_block_sums[block_offset/COMPUTE_DEGREES_DIMX] = (scan + degree); // scan is exclusive 
            }

            if((block_offset + gridDim.x*blockDim.x) < queue_end) {
                // if there's another iteration, we'll reuse temp_storage
                __syncthreads();
            }
        }

        if(threadIdx.x == 0) {
            int old = atomicAdd(d_n_CTA_done, 1);
            blk_scan_offset = 0; // will be used if last CTA, avoiding a second sync
            is_last_CTA = (old == (gridDim.x -1));
        }

        __syncthreads(); // is_last_CTA + temp_storage reuse if last CTA

        if(is_last_CTA) {
                // The last block alive takes care of scan of block sums 
                __threadfence();
                if(threadIdx.x == 0) {
                    *d_n_CTA_done = 0;
                }

                // following value can be different than gridDim.x
                int total_blk_val = (queue_size + COMPUTE_DEGREES_DIMX -1) / COMPUTE_DEGREES_DIMX;

                for(int blk_idx_off = 0;
                    blk_idx_off < total_blk_val;
                    blk_idx_off += blockDim.x) {
                    int blk_idx = blk_idx_off + threadIdx.x;

                    int blk_sum = (blk_idx < total_blk_val) ? d_block_sums[blk_idx] : 0;

                    int blk_scan;
                    BlockScan(temp_storage).ExclusiveSum(blk_sum, blk_scan);
                    blk_scan += blk_scan_offset; 
                
                    if(blk_idx < total_blk_val) {
                        d_block_sums_scan[blk_idx] = blk_scan;
                    }
                    
                    if(threadIdx.x == (COMPUTE_DEGREES_DIMX-1)) {
                        int total = blk_scan + blk_sum;
                        blk_scan_offset = total;
                    }

                    __syncthreads(); // blk_scan_offset + reuse temp_storage
                }

            if(threadIdx.x == 0) {
                *d_q_token_from_narcs = blk_scan_offset; // pinned memory
                *h_q_token_from_narcs = blk_scan_offset; // pinned memory
            }
        }
  }

  void CudaDecoder::ComputeDegrees(unsigned int *d_offsets) {
    dim3 grid,block;
    block.x = COMPUTE_DEGREES_DIMX;
    grid.x = DIV_ROUND_UP(*h_q_token_from_size, block.x);

    compute_degrees_kernel<<<grid,block,0,compute_st>>>(d_allToken, d_allTokenInfo, d_q_token_from, d_q_token_to, d_degrees_scan,
    d_offsets, d_state_cost, d_cutoff, d_q_arc_offset, d_block_sums_scan, d_block_sums_scan, h_q_token_from_narcs,
    d_q_token_from_narcs, d_n_CTA_done);
  }


/*

Part 2 of the scan. Computes global prefix sum with block prefix sum and block offsets

If we want to speed up expand, we can compute lower and upper bound to restrain 
the binary search in expand
This can be done on the fly here, and removes main bottleneck of expand
Not done for now, because expand is fast enough

*/
 __global__ void finalize_degrees_scan_kernel(int *d_scan, int *d_blk_scan, const int *d_q_token_from, const int
  *d_q_token_to) {

        int q_off = *d_q_token_from;
        int q_end = *d_q_token_to;
        int q_size = q_end - q_off;

        for(int idx = blockDim.x*blockIdx.x + threadIdx.x;
                idx < q_size;
                idx += blockDim.x*gridDim.x) {

            int blk_idx = idx / blockDim.x;
            int blk_scan_offset = d_blk_scan[blk_idx]; // we rely on L1 for this one, avoiding syncs

            d_scan[idx] += blk_scan_offset;
        }

 }

  void CudaDecoder::FinalizeDegreesScan() {
      dim3 grid,block;
      block.x = COMPUTE_DEGREES_DIMX;
      grid.x = DIV_ROUND_UP(*h_q_token_from_size, block.x);

      finalize_degrees_scan_kernel<<<grid,block,0,compute_st>>>(d_degrees_scan, d_block_sums_scan, d_q_token_from, d_q_token_to); 
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

typedef CudaDecoder::ExpandArcParams ExpandArcParams; // TODO

#define EXPAND_ARCS_DIMX 256

/*

This kernel propagates arcs from the current queue [from,to[
to the new queue [to,end[

The main bottleneck is the first binary search. 
If we want to remove that bottleneck, cf comments on FinalizeScan


TODO merge reduce and scan for code simplicity + remove syncs

The last block alive moves the queues indexes :
new from is old to
new to is new end
new end stays new end


*/


void __global__ expand_arcs_kernel(ExpandArcParams params) {
    typedef cub::BlockScan<int, EXPAND_ARCS_DIMX> BlockScan;
    typedef cub::BlockReduce<BaseFloat, EXPAND_ARCS_DIMX> BlockReduce;
    
    __shared__ typename BlockScan::TempStorage temp_storage_scan;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

    __shared__ int new_q_block_off;
    __shared__ BaseFloat global_cutoff;
 
    const int total_narcs = *params.d_q_token_from_narcs;
    const int old_q_offset = *params.d_q_token_from;
    const int old_q_size = *params.d_q_token_to - old_q_offset;

    if(threadIdx.x == 0) {
        global_cutoff = *params.d_cutoff;
    }

    __syncthreads();
 
    // Keeping the whole CTA alive, we'll have syncs
    for(int block_offset = blockDim.x*blockIdx.x;
            block_offset < total_narcs;
            block_offset += gridDim.x*blockDim.x) {

        int th_idx = block_offset + threadIdx.x;
        bool valid_input = (th_idx < total_narcs);

        StateId prev_state;
        BaseFloat total_cost = FLT_MAX;
        int arc_idx;
        StateId arc_next_state;
        int q_idx;

        if(valid_input) {
            //we can do better than that
            q_idx = old_q_offset + binsearch_maxle(params.d_degrees_scan, th_idx, 0, old_q_size-1); 
            
            int lower_bound = params.d_degrees_scan[q_idx - old_q_offset];
            prev_state = params.d_q[q_idx];

            int arc_offset_start = params.d_q_arc_offsets[q_idx - old_q_offset];
            arc_idx = arc_offset_start + (block_offset + threadIdx.x - lower_bound);

            arc_next_state = params.arc_nextstates[arc_idx];
            BaseFloat arc_weight = params.arc_weights[arc_idx];
            
            int arc_ilabel = params.is_emitting ? params.arc_ilabels[arc_idx] : 0;

            BaseFloat accoustic_cost = (arc_ilabel != 0) ? -params.d_loglikelihoods[arc_ilabel] : 0.0; 
            BaseFloat next_state_cost = orderedIntToFloat(params.d_lookup[arc_next_state]);

            BaseFloat old_tok_cost = params.d_q_info[q_idx].cost;

            total_cost = accoustic_cost + arc_weight + old_tok_cost;

            if(total_cost >= next_state_cost) {
                total_cost = FLT_MAX;
                valid_input = false; 
            } 
        }
        
        BaseFloat thread_cutoff = (total_cost < FLT_MAX) ? (total_cost + params.beam) : FLT_MAX;
        BaseFloat new_block_cutoff = BlockReduce(temp_storage_reduce).Reduce(thread_cutoff, cub::Min());

        if(threadIdx.x == 0) {
            if(new_block_cutoff < global_cutoff) {
                BaseFloat new_global_cutoff = fatomicMin(params.d_cutoff, new_block_cutoff);
                new_global_cutoff = min(new_global_cutoff, new_block_cutoff);
                global_cutoff = new_global_cutoff;
            }
        }
        
        __syncthreads();

        BaseFloat cutoff = global_cutoff;

        int has_successor = (total_cost < cutoff && valid_input) ? 1 : 0;

        if(has_successor) {
            // reduce, not atomic (no return)
            atomicMin(&params.d_lookup[arc_next_state], floatToOrderedInt(total_cost));
        }

        int new_q_idx_block;

        BlockScan(temp_storage_scan).ExclusiveSum(has_successor, new_q_idx_block); // we could merge the reduce and
        //the scan

        
        //printf("thx=%i, next_state=%i, new_arc_idx=%i, arc_idx=%i \n", threadIdx.x, arc_next_state, new_q_idx_block,
        //arc_idx);

        if(threadIdx.x == (EXPAND_ARCS_DIMX - 1)) {
            int total_block = new_q_idx_block + has_successor; // exclusive sum
            new_q_block_off = atomicAdd(params.d_q_token_end, total_block);
        }

        __syncthreads(); // newQueue_block_off + we'll reuse temp_storage_scan + global cutoff

        int new_q_index = new_q_block_off + new_q_idx_block;

        if(has_successor) {
            params.d_q[new_q_index] = arc_next_state;

            InfoToken new_tok_info;
            new_tok_info.cost = total_cost;
            // Negative means we'll have to reindex at the end of advancedecoding
            new_tok_info.prev_token = q_idx;
            new_tok_info.arc_idx = arc_idx;
    
            params.d_q_info[new_q_index] = new_tok_info;

            //printf("Posted one NOT NULL tok=%i to %i, arc_idx=%i, with pred=%i (q=%i), cost=%f\n", new_q_index,
            //arc_next_state, arc_idx, prev_state, q_idx, total_cost);
        }
    }


    // Last block alive moves queue 

    if(threadIdx.x == 0) {
        int old = atomicAdd(params.d_n_CTA_done, 1);
        if(old == (gridDim.x -1)) {
            // The last block alive takes care of preparing for next iter
            __threadfence(); // we want last value of d_q_token_end
            int final_end = *params.d_q_token_end;

            *params.h_q_token_from_size = final_end - *params.d_q_token_to;

            *params.d_n_CTA_done = 0;
            *params.d_q_token_from = *params.d_q_token_to;
            *params.d_q_token_to = final_end;

            if(params.is_emitting) {
                // Saving position of curr_token for this frame
                // We'll need to reset d_q_token_from for next frame
                *params.d_curr_token = *params.d_q_token_from;
            }
        }
    }

}

void CudaDecoder::ExpandArcs(int nthreads, const ExpandArcParams &params) {
    dim3 grid,block;
    block.x = 256;
    grid.x = DIV_ROUND_UP(nthreads, block.x);

    expand_arcs_kernel<<<grid,block,0,compute_st>>>(params);
}



// Reached final kernel
__global__ void reached_final_kernel(StateId *d_q, const int *d_q_token_from, const int *d_q_token_to, BaseFloat *final, float fst_zero, int *h_reached_final) {
    int q_offset = *d_q_token_from;
    int q_end = *d_q_token_to;

    for(int idx = q_offset + blockDim.x*blockIdx.x + threadIdx.x;
            idx < q_end;
            idx += blockDim.x*gridDim.x) {

       StateId state = d_q[idx];
       float final_val = final[state]; 

       if(final_val != fst_zero) {
            *h_reached_final = 1; // we could exit
       }
    }

}

  bool CudaDecoder::ReachedFinal() const {
      dim3 grid, block;
      block.x = 256;
      grid.x = DIV_ROUND_UP(*h_q_token_from_size, block.x);

      reached_final_kernel<<<grid,block,0,cudaStreamPerThread>>>(d_allToken, d_q_token_from, d_q_token_to, fst_.final_d, StdWeight::Zero().Value(), h_reached_final);
      cudaDeviceSynchronize(); //TODO...

      return *h_reached_final;
  }



// Used to find best costs.
// TODO Needs to be rewritten

#define FILL_COSTS_DIMX 256
__global__ void fill_costs_kernel(StateId *d_q, InfoToken *d_q_it, const int *d_q_token_from, const int *d_q_token_to,
int *d_costs, BaseFloat *d_final, bool final) {
    int q_offset = *d_q_token_from;
    int q_end = *d_q_token_to;

    for(int idx = q_offset + blockIdx.x*blockDim.x + threadIdx.x;
            idx < q_end;
            idx += blockDim.x*gridDim.x) {
        BaseFloat cost = d_q_it[idx].cost;
        
        if(final) {
            StateId state = d_q[idx];
            cost += d_final[state];
        }
        
        //printf("idx=%i, final=%i, cost=%f \n", idx, final, cost);

        d_costs[idx-q_offset] = floatToOrderedInt(cost);
    }

}


void CudaDecoder::GetBestCost(BaseFloat *min, int *arg, bool isfinal) const {
    dim3 grid, block;
    block.x = FILL_COSTS_DIMX;

    grid.x = DIV_ROUND_UP(*h_q_token_from_size, block.x);

    // TODO using lookup as float buffer for now - NEED TO CHANGE
    fill_costs_kernel<<<grid,block,0,cudaStreamPerThread>>>(d_allToken, d_allTokenInfo,
    d_q_token_from, d_q_token_to, d_state_cost, fst_.final_d, isfinal);

    cub::KeyValuePair<int, int> *d_argmin;
    cudaMalloc(&d_argmin, sizeof(cub::KeyValuePair<int, int>));
    
    void *d_temp_storage_amin = NULL;
    size_t temp_storage_amin_bytes = 0;

    int max_t = max_tokens;
    cub::DeviceReduce::ArgMin(d_temp_storage_amin, temp_storage_amin_bytes, d_state_cost, d_argmin, *h_q_token_from_size);
    cudaMalloc(&d_temp_storage_amin, temp_storage_amin_bytes);

    cub::DeviceReduce::ArgMin(d_temp_storage_amin, temp_storage_amin_bytes, d_state_cost, d_argmin, *h_q_token_from_size);

    cub::KeyValuePair<int, int> h_argmin;

    cudaMemcpy(&h_argmin, d_argmin, sizeof(cub::KeyValuePair<int, int>), cudaMemcpyDeviceToHost);
   

    cudaFree(d_temp_storage_amin);
    cudaFree(d_argmin);

    //InitLookup(); // reset lookup

    //*min = orderedIntToFloat(h_argmin.value);
    *min = -10; // TODO switch back to real value once new kernel ready
    *arg = h_argmin.key;
}

  BaseFloat CudaDecoder::FinalRelativeCost() const {
    if(*h_q_token_from_size == 0)
        return FLT_MAX;

      BaseFloat best_cost;
      int arg_best;
      GetBestCost(&best_cost, &arg_best, false);


      BaseFloat best_cost_final;
      int arg_best_final;
      GetBestCost(&best_cost_final, &arg_best_final, true);

      return (best_cost_final - best_cost);
  }

// brutal - one thread, multiple global memory load. But avoids a massive memcpy D2H
// Will disappear with better memory management 
void __global__ get_best_path_kernel(int best_token_idx_in_all_tokens, StateId *d_all_tokens, InfoToken
*d_all_tokens_info, int *d_reversed_path, int *path_size) {

    int tok_idx = best_token_idx_in_all_tokens;
    int idx = 0;

    //printf("start from %i \n", tok_idx);

    printf("backtrack = ");
    while(tok_idx != INT_MIN) {
        //printf("%i -> ", tok_idx);
        int state = d_all_tokens[tok_idx];
        int arc_idx = d_all_tokens_info[tok_idx].arc_idx;
        //printf("state=%i, tok=%i, arc=%i \n", state, tok_idx, arc_idx);
        //printf("at %i, arc=%i, state=%s  \n", idx, arc_idx, state);
        d_reversed_path[idx++] = arc_idx;

        int old_tok_idx = tok_idx; 
        tok_idx = d_all_tokens_info[tok_idx].prev_token;
        if(old_tok_idx <= tok_idx) 
            printf("FAIL\n");
    }
    
    printf("\n");

    *path_size = idx;
}

// Outputs an FST corresponding to the single best path
  // through the lattice.
  bool CudaDecoder::GetBestPath(Lattice *fst_out, bool use_final_probs) const {
      nvtxRangePushA("GetBestPath");

      BaseFloat best_cost;
      int arg_best;
      GetBestCost(&best_cost, &arg_best, false);

      BaseFloat best_cost_final;
      int arg_best_final;
      GetBestCost(&best_cost_final, &arg_best_final, true);

      bool isfinal = ReachedFinal();

      int h_curr_token_offset;
      cudaMemcpy(&h_curr_token_offset, d_q_token_from, sizeof(int), cudaMemcpyDeviceToHost);

      int h_best_token_idx = isfinal ? arg_best_final : arg_best; 
      h_best_token_idx += h_curr_token_offset;
  
    printf("is final = %i \n", isfinal);
    printf("curr token off=%i \n", h_curr_token_offset);
    printf("best token idx=%i \n", h_best_token_idx);
    printf("final costs : %f  final = %f \n", best_cost, best_cost_final);
    printf("final costs idx : %i  final idx = %i \n", arg_best, arg_best_final);

    cudaMemset(d_path_size, 0, sizeof(int));

    get_best_path_kernel<<<1,1,0,cudaStreamPerThread>>>(h_best_token_idx, d_allToken, d_allTokenInfo, d_reversed_path, d_path_size);

    cudaDeviceSynchronize();

    printf("flush \n");
    
    int h_path_size;
    cudaMemcpy(&h_path_size, d_path_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reversed_path, d_reversed_path, h_path_size * sizeof(int), cudaMemcpyDeviceToHost);
    

    fst_out->DeleteStates();
     
     // We can assert first state equals to root
    
    StateId cur_state = fst_out->AddState();
    fst_out->SetStart(cur_state);

    // -1 for 0-indexing, -1 for ignoring starting arc
    for (int i = h_path_size-1-1; i >= 1; i--) {
      int arc_idx = h_reversed_path[i];
      LatticeArc arc(fst_.arc_ilabels_h[arc_idx], fst_.arc_olabels_h[arc_idx], LatticeWeight(fst_.arc_weights_h[arc_idx], 0), fst_.arc_nextstates_h[arc_idx]);

      arc.nextstate = fst_out->AddState();
      fst_out->AddArc(cur_state, arc);
      cur_state = arc.nextstate;
    }

    if (isfinal && use_final_probs)
      fst_out->SetFinal(cur_state,
          LatticeWeight(fst_.Final(fst_.arc_nextstates_h[h_reversed_path[0]]), 0.0));
    else
      fst_out->SetFinal(cur_state, LatticeWeight::One());

    fst::RemoveEpsLocal(fst_out);

    nvtxRangePop();
      return true;
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
setting the queue [from,to[ to the complete curr_token queue
so that it's ready for next ProcessEmitting

We could optimize and speed up this kernel
It will only gives us a better latency for 1 stream, which is low enough
Instead, we let it compute while we use the GPU for other streams
This kernel only uses one block, and is a free rider on the GPU

*/


#define NONEM_LT_DIMX 1024
__launch_bounds__(NONEM_LT_DIMX, 1)
__global__ void process_nonem_longtail(unsigned int *d_arc_offsets, 
                                ExpandArcParams params) {

    typedef cub::BlockScan<int, NONEM_LT_DIMX> BlockScan;
    typedef cub::BlockReduce<float, NONEM_LT_DIMX> BlockReduce;

    __shared__ typename BlockScan::TempStorage temp_storage_scan;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

    __shared__ BaseFloat cutoff;
    
    __shared__ int total_narcs;

    __shared__ int new_q_end;

    int old_q_offset = *params.d_q_token_from;
    int new_q_offset = *params.d_q_token_to;

    if(threadIdx.x == 0) {
        new_q_end = *params.d_q_token_end;
        total_narcs = *params.d_q_token_from_narcs;
    }

    __syncthreads();

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

            if(threadIdx.x == 0)  {
                total_narcs = 0;
            }

            __syncthreads();


            // Step 1 : compute_degrees
            for(int local_q_idx = threadIdx.x;
                    local_q_idx < old_q_size;
                    local_q_idx += blockDim.x) {

                int global_q_idx = old_q_offset + local_q_idx;

                StateId state = params.d_q[global_q_idx];
                BaseFloat cost = params.d_q_info[global_q_idx].cost;

                int degree = 0;
                if(cost < cutoff) {
                    BaseFloat best_cost = orderedIntToFloat(params.d_lookup[state]);

                    if(cost == best_cost) {
                        int start = d_arc_offsets[state];
                        int end = d_arc_offsets[state+1];
                        degree = end - start;
                        params.d_q_arc_offsets[local_q_idx] = start;
                    }
                }

                params.d_degrees_scan[local_q_idx] = degree;
            }

            __syncthreads();

            // Step 2 : Scan

            for(int block_off = 0;
                    block_off < old_q_size;
                    block_off += blockDim.x) {

                int local_q_idx = block_off + threadIdx.x;

                int degree = (local_q_idx < old_q_size) 
                    ? params.d_degrees_scan[local_q_idx]
                    : 0;
                int lscan;
                BlockScan(temp_storage_scan).ExclusiveSum(degree, lscan);
                int scan = lscan + total_narcs;

                if(local_q_idx < old_q_size)
                    params.d_degrees_scan[local_q_idx] = scan;

                if(threadIdx.x == (NONEM_LT_DIMX-1)) {
                    int total_in_block = lscan + degree;
                    total_narcs += total_in_block;
                }

                __syncthreads();
            }

        } else {
            first = false;    
        }

        //if(threadIdx.x == 0)
        //    printf("narcs=%i \n", total_narcs);

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
                int local_q_idx = binsearch_maxle(params.d_degrees_scan, th_idx, 0, old_q_size-1); 

                //printf("thx=%i, q_idx=%i, oldqsize=%i, oldqoff=%i \n", threadIdx.x, q_idx, old_q_size, old_q_offset);

                int lower_bound = params.d_degrees_scan[local_q_idx];
                int arc_offset_start = params.d_q_arc_offsets[local_q_idx];
                q_idx = old_q_offset + local_q_idx;

                arc_idx = arc_offset_start + (th_idx - lower_bound);

                arc_next_state = params.arc_nextstates[arc_idx];
                BaseFloat arc_weight = params.arc_weights[arc_idx];
                BaseFloat next_state_cost = orderedIntToFloat(params.d_lookup[arc_next_state]);
                BaseFloat old_tok_cost = params.d_q_info[q_idx].cost;

                total_cost = arc_weight + old_tok_cost;

                if(total_cost >= next_state_cost) {
                    total_cost = FLT_MAX;
                    valid_input = false; 
                } 
            }

            BaseFloat thread_cutoff = (total_cost < FLT_MAX) ? (total_cost + params.beam) : FLT_MAX;
            BaseFloat new_block_cutoff = BlockReduce(temp_storage_reduce).Reduce(thread_cutoff, cub::Min());

            if(threadIdx.x == 0) {
                if(new_block_cutoff < cutoff) {
                    cutoff = new_block_cutoff;
                }
            }

            __syncthreads();

            int has_successor = (total_cost < cutoff && valid_input) ? 1 : 0;

            if(has_successor) 
                atomicMin(&params.d_lookup[arc_next_state], floatToOrderedInt(total_cost));
            

            int new_q_idx_block;

            BlockScan(temp_storage_scan).ExclusiveSum(has_successor, new_q_idx_block);

            if(has_successor) {
                int new_q_index = new_q_end + new_q_idx_block;
                params.d_q[new_q_index] = arc_next_state;

                InfoToken new_tok_info;
                new_tok_info.cost = total_cost;
                new_tok_info.prev_token = q_idx;
                new_tok_info.arc_idx = arc_idx;

                params.d_q_info[new_q_index] = new_tok_info;
 
            }

            if(threadIdx.x == (NONEM_LT_DIMX - 1)) {
                int total_in_block = new_q_idx_block + has_successor; // exclusive sum
                new_q_end += total_in_block;
            }
        }

        __syncthreads(); // new_q_end

        old_q_size = new_q_end - new_q_offset; 

    }

    if(threadIdx.x == 0) {
        // Next step is ProcessEmitting of next frame, from is currToken_offset
        *params.d_q_token_from = *params.d_curr_token; 
        *params.d_q_token_to = new_q_end;
        *params.d_q_token_end = new_q_end;
        *params.d_cutoff = cutoff;

        *params.h_q_token_from_size = new_q_end - *params.d_q_token_from;
    }

}
  
void CudaDecoder::NonEmittingLongTail(unsigned int *d_arc_offsets, 
                                const ExpandArcParams &params) {

    dim3 grid,block;
    block.x = NONEM_LT_DIMX;
    grid.x = 1; // it is designed for the long tail
    process_nonem_longtail<<<grid,block,0,compute_st>>>(d_arc_offsets, params);
}

} // end namespace kaldi.
