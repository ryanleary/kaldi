// decoder/cuda-decoder.cu

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
#include <algorithm>
#include <nvToolsExt.h>
#include <cuda_runtime_api.h>
#include <float.h>
#include <math.h>
#include <cub/cub.cuh>


#define MEMADVISE

namespace kaldi {

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
        cudaMalloc(&d_degrees_block_scan, (max_tokens_per_frame_ / KERNEL_PREPROCESS_DIMX + 1 + 1)* sizeof(int));
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
        *h_main_q_local_offset = 0;
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
        preprocess_params.h_main_q_local_offset = h_main_q_local_offset;
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
        expand_params.h_main_q_local_offset = h_main_q_local_offset;
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
                    LatticeWeight(fst_.final_h[fst_.arc_nextstates_h[reversed_path[0]]], 0.0));
        else
            fst_out->SetFinal(cur_state, LatticeWeight::One());

        fst::RemoveEpsLocal(fst_out);

        nvtxRangePop();
        return true;
    }


} // end namespace kaldi.
