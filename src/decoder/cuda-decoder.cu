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

    printf("narcs=%i \n", arc_count);

    cudaDeviceSynchronize();
    cudaCheckError();

    nvtxRangePop();
  }

  void CudaFst::finalize() {
    nvtxRangePushA("CudaFst destructor");
    printf("CudaFst::finalize()\n");
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
    cudaStreamCreate(&copy_st);

    cudaEventCreate(&loglikelihood_evt);
    cudaEventCreate(&q_token_from_narcs_evt);

    cudaMalloc(&d_debug_cnt, 20*sizeof(int));


    int max_token_frame = 5000000; // move back to params
    // we could use same pointer
    cudaMalloc(&d_q_from_state, max_token_frame * sizeof(int));
    cudaMalloc(&d_q_to_state, max_token_frame * sizeof(int));

    cudaMalloc(&d_q_from_cost, max_token_frame * sizeof(CostT));
    cudaMalloc(&d_q_to_cost, max_token_frame * sizeof(CostT));
 
    cudaMalloc(&d_q_from_info, max_token_frame * sizeof(InfoToken));
    cudaMalloc(&d_q_to_info, max_token_frame * sizeof(InfoToken));

    int *bufi4;
    cudaMalloc(&bufi4, 6*sizeof(int));
    
    d_q_from_local_offset = &bufi4[0];
    d_q_from_global_offset = &bufi4[1];
    d_q_from_end = &bufi4[2];
    d_q_to_end = &bufi4[3];

    d_q_token_from_narcs = &bufi4[4];
    d_cutoff = &bufi4[5];

    
    cudaMallocHost(&h_q_from_info, max_token_frame * sizeof(InfoToken));
    cudaMallocHost(&h_q_token_from_size, sizeof(int));  

    // we could use same pointer
    cudaMalloc(&d_degrees_scan, max_token_frame * sizeof(int));
    cudaMalloc(&d_block_sums_scan, (max_token_frame / 256 + 2)* sizeof(int)); // TODO remove hardcoded
    cudaMalloc(&d_q_arc_offset, max_token_frame * sizeof(int));

    cudaMalloc(&loglikelihoods_d, sizeof(BaseFloat)*(fst_.max_ilabel+1));  
    cudaMalloc(&next_loglikelihoods_d, sizeof(BaseFloat)*(fst_.max_ilabel+1));  
    cudaMallocHost(&loglikelihoods_h, sizeof(BaseFloat)*(fst_.max_ilabel+1));  

    cudaMalloc(&d_state_cost,sizeof(CostT)*fst_.numStates);

    cudaMallocHost(&h_reached_final, sizeof(int));
    cudaMallocHost(&h_q_token_from_narcs, sizeof(int));
    
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
    it_init.prev_token = INT_MIN;
    it_init.arc_idx = -1;

    CostT cost = StdWeight::One().Value();

    cudaMemcpy(d_q_from_state, &start_state, sizeof(StateId), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_from_cost, &cost, sizeof(CostT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_from_info, &it_init, sizeof(InfoToken), cudaMemcpyHostToDevice);

    // We simulate a regular execution for the first iteration
    cudaMemcpy(&d_state_cost[start_state], &(it_init.cost), sizeof(BaseFloat), cudaMemcpyHostToDevice);

    cudaMemset(d_q_to_end, 0, sizeof(int));
    cudaMemset(d_q_from_local_offset, 0, sizeof(int));
    cudaMemset(d_q_from_global_offset, 0, sizeof(int));

    d_q_from_local_offset = &bufi4[0];

    // Init state is in queue
    int one = 1;
    cudaMemcpy(d_q_from_end, &one, sizeof(int), cudaMemcpyHostToDevice);
    *h_q_token_from_size = 1;

    CostT cutoff = FLT_MAX;
    cudaMemcpy(d_cutoff, &cutoff, sizeof(CostT), cudaMemcpyHostToDevice);

    cudaMemset(d_n_CTA_done, 0, sizeof(int));
    
    cudaCheckError();

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

    init_lookup_kernel<<<grid,block>>>(d_state_cost, nstates);
}

typedef CudaDecoder::StateId StateId;

// Used to reset lookup table between frames
// Using the queue to reset only the values needed
// Also takes care of resetting cutof
// TODO rename to something like "ResetForNewFrame"
__global__ void reset_lookup_kernel(StateId *d_q, const int *d_q_from_end, int *state_cost, CostT *d_cutoff) {
    int q_from_end = *d_q_from_end; 

    for(int idx = blockIdx.x*blockDim.x + threadIdx.x;
            idx < q_from_end;
            idx += blockDim.x*gridDim.x) {

        StateId state = d_q[idx];

        state_cost[state]  = floatToOrderedInt(FLT_MAX);
    }

    // Avoiding a kernel call just to reset the cutoff
    if(blockIdx.x == 0 && threadIdx.x == 0)
        *d_cutoff = FLT_MAX; 
}

void CudaDecoder::ResetLookup() {
    int size = *h_q_from_size;

    dim3 grid,block;
    block.x = 256;
    grid.x = DIV_ROUND_UP(size, block.x);

    reset_lookup_kernel<<<grid,block,0,compute_st>>>(d_q_from_state, d_q_from_end, d_state_cost, d_cutoff);
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

        cudaEventSynchronize(loglikelihood_evt);
        std::swap(next_loglikelihoods_d, loglikelihoods_d);
        num_frames_decoded_++; 
        ComputeLogLikelihoods(decodable);

        //KALDI_LOG << "Emitting, frame=" << num_frames_decoded_;
        ProcessEmitting();

        //KALDI_LOG << "Non Emitting";
        ProcessNonemitting(); 


        if(num_frames_decoded_ > 3) {
            //KALDI_ASSERT(0); 
        }

        //printf("total to save = %i \n", h_debug_cnt);
        //computes log likelihoods for the next frame - check order
    }   


    printf("AdvanceDecoding Done\n");
    nvtxRangePop();
}


  void CudaDecoder::ComputeLogLikelihoods(DecodableInterface *decodable) {
    nvtxRangePushA("ComputeLogLikelihoods");

    int32 frame = num_frames_decoded_;

    decodable->ComputeLogLikelihoods(loglikelihoods_h,frame,fst_.max_ilabel+1);

    //copying in another stream to overlap transfer with compute
    cudaMemcpyAsync(next_loglikelihoods_d, loglikelihoods_h, sizeof(BaseFloat)*(fst_.max_ilabel+1), cudaMemcpyHostToDevice,
    copy_st);

    cudaEventRecord(loglikelihood_evt, copy_st);

    nvtxRangePop();
  }


// Below that value, we launch the persistent kernel for NonEmitting
#define NONEM_LT_MAX_NARCS 4096
bool CudaDecoder::ProcessToken(unsigned int *d_arc_offsets,
                        bool is_emitting) {

    if(is_emitting) {
        PreprocessInPlace(d_arc_offsets);
        cudaEventRecord(q_token_from_narcs_evt, compute_st);
        ResetLookup();
        FinalizePreprocessInPlace();
    } else {
        ContractAndPreprocess(d_arc_offsets);
    }

    
    // We need h_q_token_from_narcs to be ready
    cudaEventSynchronize(q_token_from_narcs_evt);
    int main_q_narcs = *h_main_q_narcs

    ExpandArcParams params;

    params.d_main_q_state = d_main_q_state;
    params.d_main_q_cost = d_main_q_cost;
    params.d_main_q_info = d_main_q_info;

    params.d_main_q_local_offset = d_main_q_local_offset;
    params.d_main_q_global_offset = d_main_q_global_offset;
    params.d_main_q_end = d_main_q_end;

    params.d_aux_q_state = d_aux_q_state; 
    params.d_aux_q_cost = d_aux_q_cost; 
    params.d_aux_q_info = d_aux_q_info;
    params.d_aux_q_end = d_aux_q_end;

    params.d_degrees_scan = d_degrees_scan; 
    params.d_q_arc_offsets = d_q_arc_offset;
    params.arc_ilabels = fst_.arc_ilabels_d;
    params.is_emitting = is_emitting;
    params.d_main_q_narcs = d_main_q_narcs;
 
    params.arc_weights = fst_.arc_weights_d; 
    params.arc_nextstates = fst_.arc_nextstates_d; 
    params.d_cutoff = d_cutoff;
    params.beam = beam_;
    params.d_loglikelihoods= loglikelihoods_d;
    params.d_lookup = d_state_cost;

    params.d_n_CTA_done = d_n_CTA_done;

    bool done = false;

    if(main_q_narcs) {
        if(!params.is_emitting 
            && main_q_narcs < NONEM_LT_MAX_NARCS) { 
            NonEmittingLongTail(d_arc_offsets, params); 

            cudaCheckError();

            // Persistent kernel finishes the job
            done = true;
        }
        else {
            ExpandArcs(q_from_narcs, params);
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
  __global__ void contract_and_preprocess_kernel(PreprocessParams params) {


       typedef cub::BlockScan<int2, COMPUTE_DEGREES_DIMX> BlockScan;
       __shared__ typename BlockScan::TempStorage temp_storage;

       __shared__ int2 blk_local_offset_i2;

        const int queue_offset = *params.d_q_from_local_offset;
        const int queue_end = *params.d_q_from_end;
        const int queue_size = queue_end - queue_offset;

        BaseFloat cutoff = *params.d_cutoff;

        for(int block_offset = blockDim.x*blockIdx.x;
                block_offset < queue_size;
                block_offset += gridDim.x*blockDim.x) {

            int idx = queue_offset + block_offset + threadIdx.x;
            int degree = 0;
            int arc_start = -1;
            StateId state_idx;
            CostT cost;

            if(idx < queue_end) {
                cost = params.d_q_from_cost[idx];
                state_idx = params.d_q_from_state[idx];

                if(cost < cutoff) {
                    BaseFloat best_cost = orderedIntToFloat(params.d_state_cost[state_idx]);

                    if(cost == best_cost) {
                        arc_start = params.d_arc_offsets[state_idx];
                        int arc_end = params.d_arc_offsets[state_idx+1];
                        degree = arc_end - arc_start;
                    }
                } 
            }

            int is_pruned = (arc_start == -1);
            int2 scan_i2;
            scan_i2.first  =  is_pruned ? 0 : 1;
            scan_i2.second =  degree;

            BlockScan(temp_storage).ExclusiveSum(scan_i2, scan_i2);

            if(threadIdx.x == (COMPUTE_DEGREES_DIMX-1)) {
                // Scan is exclusive
                int2 inclusive_scan;
                inclusive_scan.first  = scan_i2.first + (is_pruned ? 0 : 1);
                inclusive_scan.second = scan_i2.second + degree;

                blk_local_offset_i2 = atomicAddI2(d_q_to_end_i2, inclusive_scan);
            }

            __syncthreads(); // blk_local_offset + temp_storage

            if(!is_pruned) {
                int q_from_idx = blk_local_offset_i2.first + scan_i2.first;

                InfoToken info = d_q_from_info[idx];
                d_q_to_state[q_from_idx] = state_idx;
                d_q_to_cost[q_from_idx] = cost;
                d_q_to_info[q_from_idx] = info;

                d_degrees_scan[q_from_idx] = blk_local_offset_i2.second + scan_i2.second;
                d_q_arc_offset[q_from_idx] = arc_start;
            }
            

        }

        if(threadIdx.x == 0) {
            int old = atomicAdd(params.d_n_CTA_done, 1);
            bool is_last_CTA = (old == (gridDim.x -1));

            if(is_last_CTA) {
                __threadfence();

                // Avoid a mem copy
                *d_n_CTA_done = 0;
                *h_q_token_from_narcs = d_q_from_local_offset_i2.second; // pinned memory
            }
        }

  }


#define COMPUTE_DEGREES_DIMX 256
__global__ void preprocess_in_place(PreprocessParams params) {

    typedef cub::BlockScan<int, COMPUTE_DEGREES_DIMX> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    __shared__ int blk_scan_offset;
    __shared__ int is_last_CTA;


    int queue_offset = *params.d_main_q_offset;
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
                BaseFloat best_cost = orderedIntToFloat(d_state_cost[state_idx]); 
                if(cost == best_cost) {
                    int start = d_offsets[state_idx]; 
                    int end = d_offsets[state_idx+1]; 
                    degree  = end - start;
                    params.d_q_arc_offset[idx] = start;
                }
            }
        }

        int scan;
        BlockScan(temp_storage).ExclusiveSum(degree, scan);
        if(idx < queue_end) 
            params.d_degrees_scan[idx] = scan;


        if(threadIdx.x == (COMPUTE_DEGREES_DIMX-1))
        {
            params.d_block_scan[block_offset/COMPUTE_DEGREES_DIMX] = (scan + degree); 
        }

        if((block_offset + gridDim.x*blockDim.x) < queue_end)
        {
            __syncthreads(); // we'll reuse temp_storage
        }
    }

    if(threadIdx.x == 0) {
        int old = atomicAdd(params.d_n_CTA_done, 1); 
        blk_scan_offset = 0;
        is_last_CTA = (old == (gridDim.x -1));
    }

    // is_last_CTA + temp_storage reuse
    __syncthreads();
    if(is_last_CTA)
    {
        // The last block alive takes care of scan of block sums 
        __threadfence();

        if(threadIdx.x == 0) {
            *d_n_CTA_done = 0;
        }

        // following value can be different than gridDim.x 
        int total_blk_val = (queue_size + COMPUTE_DEGREES_DIMX -1) / COMPUTE_DEGREES_DIMX;

        for(int blk_idx_off = 0; blk_idx_off < total_blk_val; blk_idx_off += blockDim.x) {
            int blk_idx = blk_idx_off + threadIdx.x; 

            int blk_sum = (blk_idx < total_blk_val) ?  params.d_block_scan[blk_idx] : 0; 
            int blk_scan;
            BlockScan(temp_storage).ExclusiveSum(blk_sum, blk_scan);
            blk_scan += blk_scan_offset; 

            if(blk_idx < total_blk_val) {
                params.d_block_scan[blk_idx] = blk_scan;
            }

            if(threadIdx.x == (COMPUTE_DEGREES_DIMX-1)) {
                int total = blk_scan + blk_sum; 
                blk_scan_offset = total;
            }

            __syncthreads();
            // blk_scan_offset + reuse temp_storage
        }

        if(threadIdx.x == 0)
        {
            *params.d_q_to_narcs = blk_scan_offset; 
            *params.h_q_to_narcs = blk_scan_offset; // pinned memory 
        }
    }
}




  void CudaDecoder::ContractAndPreprocess(unsigned int *d_arc_offsets) {
    dim3 grid,block;
    block.x = COMPUTE_DEGREES_DIMX;
    grid.x = DIV_ROUND_UP(*h_aux_q_end, block.x);

    PreprocessParams params;

    params.d_q_from_state = d_aux_q_state; 
    params.d_q_from_cost = d_aux_q_cost;
    params.d_q_from_info = d_aux_q_info; 
    params.d_q_from_end = d_aux_q_end;

    params.d_q_to_state = d_main_q_state; 
    params.d_q_to_cost = d_main_q_cost;
    params.d_q_to_info = d_main_q_info; 
    params.d_q_to_end_i2 = d_main_q_i2; 


    params.d_degrees_scan = d_degrees_scan; 
    params.d_arc_offsets = _fst. ; // TODO 
    params.d_q_arc_offsets = d_q_arc_offsets; // offsets, relative to the queue

    params.d_state_cost = d_state_cost; 
    params.d_cutoff = d_cutoff; 

    params.d_block_scan = d_degrees_block_scan; 

    params.h_q_to_narcs = h_main_q_narcs; 
    params.d_n_CTA_done = d_n_CTA_done;

    contract_and_preprocess_kernel<<<grid,block,0,compute_st>>>(params);
  }


void CudaDecoder::PreprocessInPlace(unsigned int *d_arc_offsets) {
    dim3 grid,block;
    block.x = COMPUTE_DEGREES_DIMX;
    grid.x = DIV_ROUND_UP(*h_todo, block.x);

    PreprocessParams params;

    params.d_q_from_state = d_aux_q_state; 
    params.d_q_from_cost = d_aux_q_cost;
    params.d_q_from_info = d_aux_q_info; 
    params.d_q_from_end = d_aux_q_end;

    params.d_q_to_state = d_main_q_state; 
    params.d_q_to_cost = d_main_q_cost;
    params.d_q_to_info = d_main_q_info; 
    params.d_q_to_end_i2 = d_main_q_i2; 


    params.d_degrees_scan = d_degrees_scan; 
    params.d_arc_offsets = _fst. ; // TODO 
    params.d_q_arc_offsets = d_q_arc_offsets; // offsets, relative to the queue

    params.d_state_cost = d_state_cost; 
    params.d_cutoff = d_cutoff; 

    params.d_block_scan = d_degrees_block_scan; 

    params.h_q_to_narcs = h_main_q_narcs; 
    params.d_n_CTA_done = d_n_CTA_done;

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
 __global__ void finalize_degrees_scan_kernel(int *d_scan, int *d_blk_scan, const int *d_main_q_offset, const int
  *d_main_q_end) {

        int q_off = *d_main_q_offset;
        int q_end = *d_main_q_end;
        int q_size = q_end - q_off;

        for(int idx = blockDim.x*blockIdx.x + threadIdx.x;
                idx < q_size;
                idx += blockDim.x*gridDim.x) {

            int blk_idx = idx / blockDim.x;
            int blk_scan_offset = d_blk_scan[blk_idx]; // we rely on L1 for this one, avoiding syncs

            d_scan[idx] += blk_scan_offset;
        }

 }

  void CudaDecoder::FinalizePreprocessInPlace() {
      dim3 grid,block;
      block.x = COMPUTE_DEGREES_DIMX;
      grid.x = DIV_ROUND_UP(*h_main_q_size, block.x);

      finalize_degrees_scan_kernel<<<grid,block,0,compute_st>>>(d_degrees_scan, d_degrees_block_scan, d_main_q_offset,
      d_main_q_end); 
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

struct CostTInt {
    CostT cost;
    int i;
}

void __global__ expand_arcs_kernel(ExpandArcParams params) {
    typedef cub::BlockScan<CostTInt, EXPAND_ARCS_DIMX> BlockScan;
    
    __shared__ typename BlockScan::TempStorage temp_storage_scan;

    __shared__ int to_q_block_offset;
    __shared__ CostT blk_cutoff;
 
    const int total_narcs = *params.d_main_q_narcs;
    const int main_q_offset = *params.d_main_q_local_offset;
    const int main_q_size = *params.d_main_q_end - from_q_offset;

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
        int from_q_idx;

        if(valid_input) {
            //we can do better than that
            main_q_idx = main_q_offset + binsearch_maxle(params.d_degrees_scan, th_idx, 0, main_q_size-1); 
            
            int lower_bound = params.d_degrees_scan[main_q_idx];
            int arc_offset_start = params.d_q_arc_offsets[main_q_idx];

            arc_idx = arc_offset_start + (block_offset + threadIdx.x - lower_bound);
            arc_next_state = params.arc_nextstates[arc_idx];
            
            BaseFloat total_cost = params.arc_weights[arc_idx];

            int arc_ilabel = params.is_emitting ? params.arc_ilabels[arc_idx] : 0;
            total_cost += (arc_ilabel != 0) ? -params.d_loglikelihoods[arc_ilabel] : 0.0; 
            total_cost += params.d_main_q_cost[main_q_idx];

            if(total_cost >= blk_cutoff)
                valid_input = false;
            else {
                // switch back to red, worst case is bad
                BaseFloat next_state_cost = orderedIntToFloat(
                atomicMin(&params.d_lookup[arc_next_state],
                floatToOrderedInt(total_cost)
                );

                if(total_cost >= next_state_cost)
                    valid_input = false;
            }
        }
        
        int has_successor = valid_input ? 1 : 0;  // Need a spot in the new q

        CostTInt ci;
        ci.cost = valid_input ? (total_cost + params.beam) : FLT_MAX; // new cutoff candidate
        ci.i = has_successor;

        BlockScan(temp_storage_scan).InclusiveScan(ci, ci, CISum());

        if(threadIdx.x == (EXPAND_ARCS_DIMX - 1)) {
            int total_successors_in_block = ci.i;
            to_q_block_offset = atomicAdd(params.d_aux_q_end, total_successors_in_block);
            
            if(ci.cost < blk_cutoff) {
                int new_cutoff = atomicMin(params.d_cutoff, ci.cost);
                blk_cutoff = fmin(ci.cost, new_cutoff);
            }
        }

        __syncthreads(); // to_q_block_offset

        ci.i -= has_successor; // we want the exclusive sum now
        int to_q_index = to_q_block_offset + ci.i;

        // TODO local q in shared

        if(has_successor) {
            params.d_aux_q_cost[to_q_index] = total_cost;

            if(total_cost < blk_cutoff) { // cutoff may have changed
                // We write the rest of the token only if necessary
                // if the cost is higher than cutoff, 
                // the token will be ignored anyway 

                params.d_aux_q_state[to_q_index] = arc_next_state;

                InfoToken new_tok_info;
                new_tok_info.prev_token = from_q_idx;
                new_tok_info.arc_idx = arc_idx;

                params.d_aux_q_info[to_q_index] = new_tok_info;
            }
        }
    }


    // Last block alive sets h_aux_q_end (pinned memory)
    if(threadIdx.x == 0) {
        int old = atomicAdd(params.d_n_CTA_done, 1);
        if(old == (gridDim.x -1)) {
            __threadfence(); // we want last value of d_aux_q_end
            *params.h_aux_q_end = *params.d_aux_q_end;
            *params.d_n_CTA_done = 0;
        }
    }

}

void CudaDecoder::ExpandArcs(int nthreads, const ExpandArcParams &params) {
    dim3 grid,block;
    block.x = 256;
    grid.x = DIV_ROUND_UP(nthreads, block.x);

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

    int old_q_offset = *params.d_main_q_offset;
    int new_q_offset = *params.d_main_q_end;

    if(threadIdx.x == 0) {
        new_q_end = new_q_offset;
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

                StateId state = params.d_main_q_state[global_q_idx];
                BaseFloat cost = params.d_main_q_costglobal_q_idx];

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
                BaseFloat old_tok_cost = params.d_main_q_cost[q_idx];

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
                params.d_main_q_state[new_q_index] = arc_next_state;

                params.d_main_q_cost[new_q_index] = total_cost;

                InfoToken new_tok_info;
                new_tok_info.prev_token = q_idx;
                new_tok_info.arc_idx = arc_idx;
                params.d_main_q_info[new_q_index] = new_tok_info;
 
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
        *params.d_main_q_end = new_q_end; 
        *params.h_main_q_end = new_q_end; 

        *params.d_cutoff = cutoff;
    }

}
  
void CudaDecoder::NonEmittingLongTail(unsigned int *d_arc_offsets, 
                                const ExpandArcParams &params) {

    dim3 grid,block;
    block.x = NONEM_LT_DIMX;
    grid.x = 1; // it is designed for the long tail
    process_nonem_longtail<<<grid,block,0,compute_st>>>(d_arc_offsets, params);
}


/*
    GetBestCost, GetBestPath, IsFinal
    CPU only, called only at the end

*/


void CudaDecoder::GetBestCost(BaseFloat *min, int *arg, bool isfinal) const {
    CostT best_cost = FLT_MAX; // switch to numeric limits std11
    int best_cost_idx;
// we need main q end ready
    cudaMemcpy(h_main_q_cost, d_main_q_cost, h_main_q_end * sizeof(CostT), cudaMemcpyDeviceToHost);

    if(isfinal)
        cudaMemcpy(h_main_q_state, d_main_q_state, h_main_q_end * sizeof(int), cudaMemcpyDeviceToHost);

    // TODO add event main q ready once memcpy becomes async

    for(int i=0; i < h_main_q_end; ++i) {
        CostT cost = h_main_q_cost[i];

        if(isfinal) 
            cost += fst_.final_h[h_main_q_state[i]];
        
        if(cost < best_cost) {
            best_cost = cost;
            best_cost_idx = i;
        }
    }

    best_cost_idx += h_main_q_global_offset;

    *min = best_cost;
    *arg = best_cost_idx;
}


bool CudaDecoder::ReachedFinal() const {
    cudaMemcpy(h_main_q_state, d_main_q_state, h_main_q_end * sizeof(int), cudaMemcpyDeviceToHost);


    for(int i=0; i < h_main_q_end; ++i) {
        if(fst_.final_h[h_main_q_state[i]] != SedWeight::Zero().Value())
            return true;
    }

    return false;
}
// Outputs an FST corresponding to the single best path
  // through the lattice.
  bool CudaDecoder::GetBestPath(Lattice *fst_out, bool use_final_probs) const {
      nvtxRangePushA("GetBestPath");
    
      bool isfinal = ReachedFinal();
      BaseFloat best_cost;
      int arg_best;
      GetBestCost(&best_cost, &arg_best, isfinal);

  
    printf("is final = %i \n", isfinal);
    printf("best cost : %f  with arg = %f \n", best_cost, arg_best);

    int token_idx = arg_best;
    std::vector<int> h_reversed_path;

    while(token_idx != INT_MIN) {
        int arc_idx = h_all_tokens_info[token_idx].arc_idx;
        h_reversed_path.push_back(arc_idx);
        token_idx = h_all_tokens_info[token_idx].prev_token;
    }


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





} // end namespace kaldi.
