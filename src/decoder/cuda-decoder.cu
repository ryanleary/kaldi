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
        return final_h[state];
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

        cudaMemcpy(e_offsets_d,e_offsets_h,sizeof(unsigned int)*(numStates+1),cudaMemcpyHostToDevice);
        cudaMemcpy(ne_offsets_d,ne_offsets_h,sizeof(unsigned int)*(numStates+1),cudaMemcpyHostToDevice);


        //Allocate non-zero arrays
        cudaMallocHost(&arc_weights_h,arc_count*sizeof(BaseFloat));
        cudaMallocHost(&arc_nextstates_h,arc_count*sizeof(StateId));
        cudaMallocHost(&arc_ilabels_h,arc_count*sizeof(int32));
        cudaMallocHost(&arc_olabels_h,arc_count*sizeof(int32));

        cudaMalloc((void**)&arc_weights_d,arc_count*sizeof(BaseFloat));
        cudaMalloc((void**)&arc_nextstates_d,arc_count*sizeof(StateId));
        // Only the ilabels for the e_arc are needed on the device
        cudaMalloc((void**)&arc_ilabels_d,e_count*sizeof(int32)); 

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
        cudaMemcpy(arc_ilabels_d,arc_ilabels_h, e_count*sizeof(int32),cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        cudaCheckError();

        nvtxRangePop();
    }

    void CudaFst::finalize() {
        nvtxRangePushA("CudaFst destructor");
        cudaFreeHost(final_h);
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

    CudaDecoder::CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config): fst_(fst), 
                     beam_(config.beam),
                     bytes_cudaMalloc(0), 
                     max_tokens_(config.max_tokens), 
                     max_tokens_per_frame_(config.max_tokens_per_frame) {

        // Comments about variables are in the .h file

        cudaStreamCreate(&compute_st);
        cudaStreamCreate(&copy_st);

        cudaEventCreate(&q_token_from_narcs_evt);
        cudaEventCreate(&can_write_to_main_q);

        cudaMalloc(&d_main_q_state, max_tokens_per_frame_ * sizeof(int));
        cudaMallocHost(&h_main_q_state, max_tokens_per_frame_ * sizeof(int));
        cudaMalloc(&d_aux_q_state, max_tokens_per_frame_ * sizeof(int));

        cudaMalloc(&d_main_q_cost, max_tokens_per_frame_ * sizeof(CostType));
        cudaMallocHost(&h_main_q_cost, max_tokens_per_frame_ * sizeof(CostType));
        cudaMalloc(&d_aux_q_cost, max_tokens_per_frame_ * sizeof(CostType));

        cudaMalloc(&d_main_q_info, max_tokens_per_frame_ * sizeof(InfoToken));
        cudaMalloc(&d_aux_q_info, max_tokens_per_frame_ * sizeof(InfoToken));

        cudaMalloc(&d_main_q_local_offset, sizeof(int));
        cudaMalloc(&d_aux_q_end, sizeof(int));
        cudaMalloc(&d_n_CTA_done, sizeof(int));

        cudaMalloc(&d_main_q_end_and_narcs_i2, sizeof(QEndAndNarcs));
        d_main_q_narcs = &d_main_q_end_and_narcs_i2->split.narcs;
        d_main_q_end = &d_main_q_end_and_narcs_i2->split.end;

        cudaMalloc(&d_cutoff, sizeof(BaseFloat));

        h_all_tokens_info.SetCudaStream(copy_st);
        h_all_tokens_info.Reserve(max_tokens_);

        cudaMallocHost(&h_main_q_end, sizeof(int));  
        cudaMallocHost(&h_main_q_narcs, sizeof(int));  
        cudaMallocHost(&h_main_q_local_offset, sizeof(int));  
        cudaMallocHost(&h_aux_q_end, sizeof(int));  

        cudaMallocHost(&h_q_overflow, sizeof(int));  

        cudaMalloc(&d_degrees_scan, max_tokens_per_frame_ * sizeof(int));
        cudaMalloc(&d_degrees_block_scan, (max_tokens_per_frame_ / 256 + 2)* sizeof(int)); // TODO remove hardcoded
        cudaMalloc(&d_main_q_arc_offsets, max_tokens_per_frame_ * sizeof(int));

        cudaMalloc(&loglikelihoods_d, sizeof(BaseFloat)*(fst_.max_ilabel+1));  

        cudaMalloc(&d_state_cost,sizeof(CostType)*fst_.numStates);

        cudaCheckError();
    }

    CudaDecoder::~CudaDecoder() {

        cudaStreamDestroy(compute_st);
        cudaStreamDestroy(copy_st);

        cudaEventDestroy(q_token_from_narcs_evt);
        cudaEventDestroy(can_write_to_main_q);

        cudaFree(d_main_q_state);
        cudaFreeHost(h_main_q_state);
        cudaFree(d_aux_q_state);

        cudaFree(d_main_q_cost);
        cudaFreeHost(h_main_q_cost);
        cudaFree(d_aux_q_cost);

        cudaFree(d_main_q_info);
        cudaFree(d_aux_q_info);

        cudaFree(d_main_q_local_offset);
        cudaFree(d_aux_q_end);
        cudaFree(d_n_CTA_done);

        cudaFree(d_main_q_end_and_narcs_i2);

        cudaFree(d_cutoff);


        cudaFreeHost(h_main_q_end);
        cudaFreeHost(h_main_q_narcs);
        cudaFreeHost(h_main_q_local_offset);
        cudaFreeHost(h_aux_q_end);

        cudaFree(d_degrees_scan);
        cudaFree(d_degrees_block_scan);
        cudaFree(d_main_q_arc_offsets);

        cudaFree(loglikelihoods_d);

        cudaFree(d_state_cost);
    }

    void CudaDecoder::InitDecoding() {

        InitLookup();

        StateId start_state = fst_.Start();
        KALDI_ASSERT(start_state != fst::kNoStateId);

        cudaCheckError();
        InfoToken it_init;
        it_init.prev_token = INT_MIN;
        it_init.arc_idx = -1;

        CostType cost = StdWeight::One().Value();

        // We'll call ProcessNonemitting just after,
        // which will move tokens from aux to main
        cudaMemcpy(d_aux_q_state, &start_state, sizeof(StateId), cudaMemcpyHostToDevice);
        cudaMemcpy(d_aux_q_cost, &cost, sizeof(CostType), cudaMemcpyHostToDevice);
        cudaMemcpy(d_aux_q_info, &it_init, sizeof(InfoToken), cudaMemcpyHostToDevice);

        // We simulate a regular execution for the first iteration
        cudaMemcpy(&d_state_cost[start_state], &cost, sizeof(CostType), cudaMemcpyHostToDevice);

        // Init state is in queue
        int one = 1;
        cudaMemcpy(d_aux_q_end, &one, sizeof(int), cudaMemcpyHostToDevice);
        *h_aux_q_end = 1;

        cudaMemset(d_main_q_end, 0, sizeof(int));
        cudaMemset(d_main_q_narcs, 0, sizeof(int));
        *h_main_q_end = 0;
        *h_main_q_narcs = 0;

        *h_q_overflow = 0;

        cudaMemset(d_main_q_local_offset, 0, sizeof(int));
        main_q_global_offset = 0;
        h_all_tokens_info.Reset();

        CostType cutoff = FLT_MAX;
        cudaMemcpy(d_cutoff, &cutoff, sizeof(CostType), cudaMemcpyHostToDevice);

        cudaMemset(d_n_CTA_done, 0, sizeof(int));

        cudaCheckError();

        num_frames_decoded_ = 0;

        ProcessNonemitting();

        int main_q_size = *h_main_q_end;
        h_all_tokens_info.CopyFromDevice(main_q_global_offset, d_main_q_info, main_q_size);
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

        KALDI_ASSERT(nstates > 0);

        dim3 grid,block;
        block.x = 256;
        grid.x = DIV_ROUND_UP(nstates, block.x);

        init_lookup_kernel<<<grid,block>>>(d_state_cost, nstates);
    }

    typedef CudaDecoder::StateId StateId;
    typedef CudaDecoder::QEndAndNarcs QEndAndNarcs;
    typedef CudaDecoder::CostType CostType;

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


    void CudaDecoder::AdvanceDecoding(DecodableInterface *decodable,
            int32 max_num_frames) {

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

        int prev_main_q_size = *h_main_q_end;
        while (num_frames_decoded_ < target_frames_decoded) {
            
            // Computing a new frame

            num_frames_decoded_++; 
            ComputeLogLikelihoods(decodable);

            // Emitting 
            // we will not write in the main q in that step
            // (preprocess is in place)
            // we don't need can_write_to_main_q
            ProcessEmitting();
            // After process emitting we won't need the token
            // associated with the previous frame
            // the main q has been flushed, we update its offset
            main_q_global_offset += prev_main_q_size;
            
            // Non Emitting
            // we will write to the main q 
            // (preprocess is "contract and preprocess")
            cudaEventSynchronize(can_write_to_main_q);
            ProcessNonemitting(); 
            
            prev_main_q_size = *h_main_q_end;
            
            // We are done with the current frame
            // We copy back its pruned tokens to the host
            // We only copy the "info" part (arc_idx + prev_token)
            // because we don't need anything else for the final backtrack
            h_all_tokens_info.CopyFromDevice(main_q_global_offset, d_main_q_info, prev_main_q_size);
            cudaEventRecord(can_write_to_main_q, copy_st);
            
        }   


        nvtxRangePop();
    }


    void CudaDecoder::ComputeLogLikelihoods(DecodableInterface *decodable) {

        int32 frame = num_frames_decoded_;

        decodable->ComputeLogLikelihoods(loglikelihoods_d,frame,fst_.max_ilabel+1, compute_st);
    }

    void CudaDecoder::PrintOverflowWarning() {

        KALDI_WARN << "Preventing overflow of the frame tokens. Pursuing "
            << "execution but the quality of the output may be decreased. "
            << "To prevent this from happening, please increase the parameter --max-tokens-per-frame"
            << " and/or decrease --beam";
    }


    // Below that value, we launch the persistent kernel for NonEmitting
#define NONEM_LT_MAX_NARCS 4096
    bool CudaDecoder::ProcessToken(unsigned int *d_arc_offsets,
            bool is_emitting) {

        PreprocessParams preprocess_params;
        preprocess_params.d_aux_q_state = d_aux_q_state; 
        preprocess_params.d_aux_q_cost = d_aux_q_cost;
        preprocess_params.d_aux_q_info = d_aux_q_info; 
        preprocess_params.d_aux_q_end = d_aux_q_end;
        preprocess_params.h_aux_q_end = h_aux_q_end;
        preprocess_params.d_main_q_state = d_main_q_state; 
        preprocess_params.d_main_q_cost = d_main_q_cost;
        preprocess_params.d_main_q_info = d_main_q_info; 
        preprocess_params.d_main_q_end_and_narcs_i2 = d_main_q_end_and_narcs_i2; 
        preprocess_params.d_main_q_narcs = d_main_q_narcs;
        preprocess_params.d_main_q_end = d_main_q_end;
        preprocess_params.h_main_q_end = h_main_q_end;
        preprocess_params.d_main_q_local_offset = d_main_q_local_offset;
        preprocess_params.d_main_q_end = d_main_q_end;
        preprocess_params.h_main_q_narcs = h_main_q_narcs;
        preprocess_params.q_capacity = max_tokens_per_frame_;
        preprocess_params.h_q_overflow = h_q_overflow;
        preprocess_params.d_degrees_scan = d_degrees_scan; 
        preprocess_params.d_arc_offsets = d_arc_offsets;
        preprocess_params.d_main_q_arc_offsets = d_main_q_arc_offsets;
        preprocess_params.d_state_cost = d_state_cost; 
        preprocess_params.d_cutoff = d_cutoff; 
        preprocess_params.d_degrees_block_scan = d_degrees_block_scan; 
        preprocess_params.d_n_CTA_done = d_n_CTA_done;

        if(is_emitting) {
            PreprocessInPlace(preprocess_params);
            cudaEventRecord(q_token_from_narcs_evt, compute_st);
            ResetLookup();
            FinalizePreprocessInPlace();
        } else {
            ContractAndPreprocess(preprocess_params);
            cudaEventRecord(q_token_from_narcs_evt, compute_st);
        }


        // We need h_q_token_from_narcs to be ready
        cudaEventSynchronize(q_token_from_narcs_evt);
        cudaCheckError();

        int main_q_narcs = *h_main_q_narcs;
        int q_overflow = *h_q_overflow;

        if(q_overflow) {
            // An overflow was prevented in the contract and preprocess kernel
            // The algorithm can still go on but quality of the result can be reduced
            // (less tokens were generated)

            PrintOverflowWarning();

            *h_q_overflow = 0;
        }

        ExpandArcParams expand_params;
        expand_params.d_main_q_state = d_main_q_state;
        expand_params.d_main_q_cost = d_main_q_cost;
        expand_params.d_main_q_info = d_main_q_info;
        expand_params.d_main_q_local_offset = d_main_q_local_offset;
        expand_params.main_q_global_offset = main_q_global_offset;
        expand_params.d_main_q_end = d_main_q_end;
        expand_params.d_main_q_narcs = d_main_q_narcs;
        expand_params.h_main_q_end = h_main_q_end;
        expand_params.h_main_q_narcs = h_main_q_narcs;
        expand_params.d_aux_q_state = d_aux_q_state; 
        expand_params.d_aux_q_cost = d_aux_q_cost; 
        expand_params.d_aux_q_info = d_aux_q_info;
        expand_params.d_aux_q_end = d_aux_q_end;
        expand_params.h_aux_q_end = h_aux_q_end;
        expand_params.q_capacity = max_tokens_per_frame_;
        expand_params.h_q_overflow = h_q_overflow;
        expand_params.d_degrees_scan = d_degrees_scan; 
        expand_params.d_q_arc_offsets = d_main_q_arc_offsets;
        expand_params.arc_ilabels = fst_.arc_ilabels_d;
        expand_params.is_emitting = is_emitting;
        expand_params.arc_weights = fst_.arc_weights_d; 
        expand_params.arc_nextstates = fst_.arc_nextstates_d; 
        expand_params.d_cutoff = d_cutoff;
        expand_params.beam = beam_;
        expand_params.d_loglikelihoods= loglikelihoods_d;
        expand_params.d_lookup = d_state_cost;
        expand_params.d_n_CTA_done = d_n_CTA_done;
    
        bool done = false;

        if(!is_emitting 
                && main_q_narcs < NONEM_LT_MAX_NARCS) { 
            NonEmittingLongTail(d_arc_offsets, expand_params); 

            cudaCheckError();

            // Persistent kernel finishes the job
            done = true;
        }
        else {
            ExpandArcs(main_q_narcs, expand_params);
        }

        cudaStreamSynchronize(compute_st); 
        cudaCheckError();

        q_overflow = *h_q_overflow;

        if(q_overflow) {
            // An overflow was prevented in the contract and preprocess kernel
            // The algorithm can still go on but quality of the result can be reduced
            // (less tokens were generated)

            PrintOverflowWarning();

            *h_q_overflow = 0;
        }
 
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
        // If remaining n_arcs < 4k, 
        // ProcessToken will call a persistent kernel
        while(!ProcessToken(fst_.ne_offsets_d, false));

        cudaCheckError();
        nvtxRangePop();
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

    typedef CudaDecoder::PreprocessParams PreprocessParams; // TODO move
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

#define COMPUTE_DEGREES_DIMX 256
    __global__ void contract_and_preprocess_kernel(PreprocessParams params) {


        typedef cub::BlockScan<int2, COMPUTE_DEGREES_DIMX> BlockScan;
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
            scan_i2.x =  is_pruned ? 0 : 1;
            scan_i2.y =  degree;

            int2 zero_i2;
            zero_i2.x = zero_i2.y = 0;

            BlockScan(temp_storage).ExclusiveScan(scan_i2, scan_i2, zero_i2, F2Sum());

            
            QEndAndNarcs inclusive_scan;
            if(threadIdx.x == (COMPUTE_DEGREES_DIMX-1)) {
                // CUB Scan is exclusive
                inclusive_scan.split.end = scan_i2.x + (is_pruned ? 0 : 1);
                inclusive_scan.split.narcs = scan_i2.y + degree;

                blk_local_offset_i2.both = atomicAdd(&params.d_main_q_end_and_narcs_i2->both, inclusive_scan.both);
                total_in_CTA = inclusive_scan.split.end;
            }

            __syncthreads(); // blk_local_offset + temp_storage

            // main_q overflow
            if((blk_local_offset_i2.split.end + total_in_CTA) >= params.q_capacity) {
                if(threadIdx.x == (COMPUTE_DEGREES_DIMX-1)) {
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
    The tokens are already in the main q (they were placed here by a previous "contract and preprocess"). We implicitly
    prune the non-optimal ones (by setting the degree to 0), and we compute the degrees scan.

    Here we have to do the scan in two passes : the scan will be finished in "finalize_preprocess"

    This preprocess step is used in ProcessEmitting. Tokens were placed in main_q by
    the ProcessNonEmitting of the previous frame. We cannot renumber them (it would break
    the prev_token index). We preprocess in place, leaving things as they are in main_q

*/

#define COMPUTE_DEGREES_DIMX 256
    __global__ void preprocess_in_place_kernel(PreprocessParams params) {
    
        typedef cub::BlockScan<int, COMPUTE_DEGREES_DIMX> BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage;

        __shared__ int blk_scan_offset;
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


            if(threadIdx.x == (COMPUTE_DEGREES_DIMX-1))
                params.d_degrees_block_scan[block_offset/COMPUTE_DEGREES_DIMX] = (scan + degree); 

            if((block_offset + gridDim.x*blockDim.x) < queue_end)
                __syncthreads(); // we'll reuse temp_storage
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
                *params.d_n_CTA_done = 0;
            }

            // following value can be different than gridDim.x 
            int total_blk_val = (queue_size + COMPUTE_DEGREES_DIMX -1) / COMPUTE_DEGREES_DIMX;

            for(int blk_idx_off = 0; blk_idx_off < total_blk_val; blk_idx_off += blockDim.x) {
                int blk_idx = blk_idx_off + threadIdx.x; 

                int blk_sum = (blk_idx < total_blk_val) ?  params.d_degrees_block_scan[blk_idx] : 0; 
                int blk_scan;
                BlockScan(temp_storage).ExclusiveSum(blk_sum, blk_scan);
                blk_scan += blk_scan_offset; 

                if(blk_idx < total_blk_val) {
                    params.d_degrees_block_scan[blk_idx] = blk_scan;
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
                *params.d_main_q_narcs = blk_scan_offset; 
                *params.h_main_q_narcs = blk_scan_offset; // pinned memory
            }
        }
    }


    void CudaDecoder::ContractAndPreprocess(PreprocessParams &params) {
        dim3 grid,block;
        block.x = COMPUTE_DEGREES_DIMX;
        grid.x = DIV_ROUND_UP(*h_aux_q_end, block.x);

        // We can have grid.x == 0 and still have a valid execution
        if(grid.x)
            contract_and_preprocess_kernel<<<grid,block,0,compute_st>>>(params);
    }


    void CudaDecoder::PreprocessInPlace(PreprocessParams &params) {
        dim3 grid,block;
        block.x = COMPUTE_DEGREES_DIMX;
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

            int blk_idx = (idx - q_off) / COMPUTE_DEGREES_DIMX;
            int blk_scan_offset = d_blk_scan[blk_idx]; // we rely on L1 for this one, avoiding syncs

            d_scan[idx] += blk_scan_offset;
        }

    }

    void CudaDecoder::FinalizePreprocessInPlace() {
        dim3 grid,block;
        block.x = COMPUTE_DEGREES_DIMX;
        int main_q_size = *h_main_q_end - *h_main_q_local_offset;
        grid.x = DIV_ROUND_UP(main_q_size, block.x);

        // If the main_q is empty, we will not be able to continue
        KALDI_ASSERT(grid.x > 0);

        finalize_degrees_scan_kernel<<<grid,block,0,compute_st>>>(d_degrees_scan, d_degrees_block_scan, d_main_q_local_offset,
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

    typedef CudaDecoder::ExpandArcParams ExpandArcParams; // TODO move

#define EXPAND_ARCS_DIMX 256

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
        typedef cub::BlockScan<CostTInt, EXPAND_ARCS_DIMX> BlockScan;

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

                            if(threadIdx.x == (EXPAND_ARCS_DIMX - 1)) {
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
                                if(threadIdx.x == (EXPAND_ARCS_DIMX - 1)) {
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
                                
                                atomicMin(&params.d_lookup[arc_next_state],
                                floatToOrderedInt(total_cost)
                                );

                                //printf("cost = %f, cutoff = %f, beam=%f \n", total_cost, blk_cutoff, params.beam);
                                if(total_cost < blk_cutoff) { // cutoff may have changed
                                    // We write the rest of the token only if necessary
                                    // if the cost is higher than cutoff, 
                                    // the token will be ignored anyway 

                                    params.d_aux_q_state[to_q_index] = arc_next_state;

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
                    *params.d_main_q_local_offset = 0;
                    *params.d_main_q_end = 0;
                    *params.h_main_q_end = 0;
                } else {
                    *params.d_main_q_local_offset = main_q_end;
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


#define NONEM_LT_DIMX 1024
    __launch_bounds__(NONEM_LT_DIMX, 1)
        __global__ void process_nonem_longtail(unsigned int *d_arc_offsets, 
                ExpandArcParams params) {

            typedef cub::BlockScan<int, NONEM_LT_DIMX> BlockScan;
            typedef cub::BlockReduce<float, NONEM_LT_DIMX> BlockReduce;

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
        
        CostType best_cost = FLT_MAX; // switch to numeric limits std11
        int best_cost_idx;
        // we need main q end ready
        int main_q_size = *h_main_q_end;

        cudaMemcpy(h_main_q_cost, d_main_q_cost, main_q_size * sizeof(CostType), cudaMemcpyDeviceToHost);

        if(isfinal)
            cudaMemcpy(h_main_q_state, d_main_q_state, main_q_size * sizeof(int), cudaMemcpyDeviceToHost);

        // TODO add event main q ready once memcpy becomes async

        for(int i=0; i < main_q_size; ++i) {
            CostType cost = h_main_q_cost[i];

            if(isfinal) 
                cost += fst_.final_h[h_main_q_state[i]];

            if(cost < best_cost) {
                best_cost = cost;
                best_cost_idx = i;
            }
        }

        //printf("global_offset=%i \n", main_q_global_offset);
        best_cost_idx += main_q_global_offset; 

        *min = best_cost;
        *arg = best_cost_idx;
    }


    bool CudaDecoder::ReachedFinal() const {
        int main_q_size = *h_main_q_end;
        cudaMemcpy(h_main_q_state, d_main_q_state, main_q_size * sizeof(int), cudaMemcpyDeviceToHost);


        for(int i=0; i < main_q_size; ++i) {
            if(fst_.final_h[h_main_q_state[i]] != StdWeight::Zero().Value())
                return true;
        }

        return false;
    }
    // Outputs an FST corresponding to the single best path
    // through the lattice.
    bool CudaDecoder::GetBestPath(Lattice *fst_out, bool use_final_probs) const {
        nvtxRangePushA("GetBestPath");

        cudaEventSynchronize(can_write_to_main_q); // We want the copy to the host to be done

        bool isfinal = ReachedFinal();
        BaseFloat best_cost;
        int arg_best;
        GetBestCost(&best_cost, &arg_best, isfinal);

        //printf("is final = %i \n", isfinal);
        //printf("best cost : %f  with arg = %i \n", best_cost, arg_best);

        int token_idx = arg_best;
        std::vector<int> reversed_path;

        while(token_idx != INT_MIN) {
            int arc_idx = h_all_tokens_info.GetRawPointer()[token_idx].arc_idx;
            reversed_path.push_back(arc_idx);
            token_idx = h_all_tokens_info.GetRawPointer()[token_idx].prev_token;
        }


        fst_out->DeleteStates();

        // We can assert first state equals to root

        StateId cur_state = fst_out->AddState();
        fst_out->SetStart(cur_state);

        reversed_path.pop_back(); // dummy first arc

        for (int i = reversed_path.size()-1; i >= 1; i--) {
            int arc_idx = reversed_path[i];
            LatticeArc arc(fst_.arc_ilabels_h[arc_idx], fst_.arc_olabels_h[arc_idx], LatticeWeight(fst_.arc_weights_h[arc_idx], 0), fst_.arc_nextstates_h[arc_idx]);

            arc.nextstate = fst_out->AddState();
            fst_out->AddArc(cur_state, arc);
            cur_state = arc.nextstate;
        }

        if (isfinal && use_final_probs)
            fst_out->SetFinal(cur_state,
                    LatticeWeight(fst_.Final(fst_.arc_nextstates_h[reversed_path[0]]), 0.0));
        else
            fst_out->SetFinal(cur_state, LatticeWeight::One());

        fst::RemoveEpsLocal(fst_out);

        nvtxRangePop();
        return true;
    }


    CudaDecoder::TokenVector::TokenVector() {
        capacity = 16; // Not important, we're going to call Reserve anyway

        cudaMallocHost(&h_data, capacity * sizeof(InfoToken)); 
        SetCudaStream(0);

        Reset();
    }

    void CudaDecoder::TokenVector::Reset() {
        size = 0;
    };

    void CudaDecoder::TokenVector::SetCudaStream(cudaStream_t st) {
        copy_st = st;
    }

    void CudaDecoder::TokenVector::CopyFromDevice(size_t offset, InfoToken *d_ptr, size_t count) {
        Reserve(size+count); // making sure we have the space

        cudaMemcpyAsync(&h_data[offset], d_ptr, count*sizeof(InfoToken), cudaMemcpyDeviceToHost, copy_st);
        size += count;
    }
    
    void CudaDecoder::TokenVector::Reserve(size_t min_capacity) {
        if(min_capacity <= capacity)
            return;

        while(capacity < min_capacity)
            capacity *= 2;

        KALDI_LOG << "Reallocating TokenVector on host (new capacity = " << capacity << " tokens)";

        InfoToken *h_old_data = h_data;
        cudaMallocHost(&h_data, capacity * sizeof(InfoToken)); 

        if(!h_data)
            KALDI_ERR << "Host ran out of memory to store tokens. Exiting.";
        
        if(size)
            cudaMemcpyAsync(h_data, h_old_data, size * sizeof(InfoToken), cudaMemcpyHostToHost, copy_st);

        cudaFreeHost(h_old_data);
    }

    InfoToken * CudaDecoder::TokenVector::GetRawPointer() const {
        return h_data;
    }

    CudaDecoder::TokenVector::~TokenVector() {
        cudaFreeHost(h_data);
    }
} // end namespace kaldi.
