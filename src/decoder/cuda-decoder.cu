// decoder/cuda-decoder.cu

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

#include "decoder/cuda-decoder.h"
#include <algorithm>
#include <nvToolsExt.h>
#include <cuda_runtime_api.h>
#include <float.h>
#include <math.h>
#include <cub/cub.cuh>


#define MEMADVISE

#define DIV_ROUND_UP(a,b) ((a+b-1)/b)

namespace kaldi {

    CudaDecoder::CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config): fst_(fst), 
                     beam_(config.beam),
                     bytes_cudaMalloc(0), 
                     max_tokens_(config.max_tokens), 
                     max_tokens_per_frame_(config.max_tokens_per_frame),
                     h_all_tokens_info_(config.max_tokens) {

        //
        // For a description of the class members, please refer to the cuda-decoder.h file
        //

        cudaStreamCreate(&compute_st_);
        cudaStreamCreate(&copy_st_); // TODO COPY IS NOT INIT in h_all_tokens_info

        cudaEventCreate(&can_read_h_main_q_narcs_);
        cudaEventCreate(&can_write_to_main_q_);

        cudaMalloc(&d_main_q_state_, max_tokens_per_frame_ * sizeof(int32));
        cudaMallocHost(&h_main_q_state_, max_tokens_per_frame_ * sizeof(int32));
        cudaMalloc(&d_aux_q_state_, max_tokens_per_frame_ * sizeof(int32));

        cudaMalloc(&d_main_q_cost_, max_tokens_per_frame_ * sizeof(CostType));
        cudaMallocHost(&h_main_q_cost_, max_tokens_per_frame_ * sizeof(CostType));
        cudaMalloc(&d_aux_q_cost_, max_tokens_per_frame_ * sizeof(CostType));

        cudaMalloc(&d_main_q_info_, max_tokens_per_frame_ * sizeof(InfoToken));
        cudaMalloc(&d_aux_q_info_, max_tokens_per_frame_ * sizeof(InfoToken));

        cudaMalloc(&d_main_q_local_offset_, sizeof(int32));
        cudaMalloc(&d_aux_q_end_, sizeof(int32));
        cudaMalloc(&d_n_CTA_done_, sizeof(int32));

        cudaMalloc(&d_main_q_end_and_narcs_i2_, sizeof(TokenAndArcCountUnion));
        d_main_q_narcs_ = &d_main_q_end_and_narcs_i2_->split.narcs;
        d_main_q_end_ = &d_main_q_end_and_narcs_i2_->split.ntokens;

        cudaMalloc(&d_cutoff_, sizeof(IntegerCostType));

        cudaMallocHost(&h_main_q_end_, sizeof(int32));  
        cudaMallocHost(&h_main_q_narcs_, sizeof(int32));  
        cudaMallocHost(&h_main_q_local_offset_, sizeof(int32));  
        cudaMallocHost(&h_aux_q_end_, sizeof(int32));  

        cudaMallocHost(&h_q_overflow_, sizeof(int32));  

        cudaMalloc(&d_main_q_degrees_prefix_sum_, max_tokens_per_frame_ * sizeof(int32));

        // d_main_q_degrees_block_sums_prefix_sum_ is the prefix sum of the "block sums"
        // a block sum is, for each CUDA block, the sum of the arc degrees of all tokens associated to that CUDA block
        // we add +1 because we want the last element of the prefix sum (a prefix sum of n elements generates (n+1) values)
        cudaMalloc(&d_main_q_degrees_block_sums_prefix_sum_, 
                (DIV_ROUND_UP(max_tokens_per_frame_, KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX) + 1)* sizeof(int32));

        cudaMalloc(&d_main_q_arc_offsets_, max_tokens_per_frame_ * sizeof(int32));

        cudaMalloc(&d_loglikelihoods_, sizeof(BaseFloat)*(fst_.max_ilabel+1));  

        cudaMalloc(&d_state_best_cost_, sizeof(IntegerCostType)*fst_.numStates);

        h_all_tokens_info_.SetCudaStream(copy_st_);

        cudaCheckError();
    }

    CudaDecoder::~CudaDecoder() {
        cudaStreamDestroy(compute_st_);
        cudaStreamDestroy(copy_st_);

        cudaEventDestroy(can_read_h_main_q_narcs_);
        cudaEventDestroy(can_write_to_main_q_);

        cudaFree(d_main_q_state_);
        cudaFree(d_aux_q_state_);
        cudaFree(d_main_q_cost_);
        cudaFree(d_aux_q_cost_);
        cudaFree(d_main_q_info_);
        cudaFree(d_aux_q_info_);
        cudaFree(d_main_q_local_offset_);
        cudaFree(d_aux_q_end_);
        cudaFree(d_n_CTA_done_);
        cudaFree(d_main_q_end_and_narcs_i2_);
        cudaFree(d_cutoff_);
        cudaFree(d_main_q_degrees_prefix_sum_);
        cudaFree(d_main_q_degrees_block_sums_prefix_sum_);
        cudaFree(d_main_q_arc_offsets_);
        cudaFree(d_loglikelihoods_);
        cudaFree(d_state_best_cost_);

        cudaFreeHost(h_main_q_end_);
        cudaFreeHost(h_main_q_narcs_);
        cudaFreeHost(h_main_q_local_offset_);
        cudaFreeHost(h_aux_q_end_);
        cudaFreeHost(h_main_q_cost_);
        cudaFreeHost(h_main_q_state_);
    }

    void CudaDecoder::InitDecoding() {
        cudaStreamSynchronize(compute_st_);

        // Filling the best state cost lookup table with +INF
        InitStateCostLookup();


        // Adding the start state to the initial token queue
        StateId first_token_state;
        CostType first_token_cost;
        InfoToken first_token_info;

        first_token_state = fst_.Start();
        first_token_cost = StdWeight::One().Value();
        first_token_info.prev_token = INT_MIN;
        first_token_info.arc_idx = -1;

        KALDI_ASSERT(first_token_state != fst::kNoStateId);

        //
        // We add that initial token to the aux_q
        // it will be moved to the main_q during the ProcessNonemitting phase 
        // that will be called in a few lines
        //
        // Note : we launch copies in the compute stream here
        // It means that we want them to be in the main pipeline
        // compute_st_ is just a name - it's a generic CUDA stream
        //
        cudaMemcpyAsync(d_aux_q_state_, &first_token_state, sizeof(StateId), cudaMemcpyHostToDevice, compute_st_);
        cudaMemcpyAsync(d_aux_q_cost_, &first_token_cost, sizeof(CostType), cudaMemcpyHostToDevice, compute_st_);
        cudaMemcpyAsync(d_aux_q_info_, &first_token_info, sizeof(InfoToken), cudaMemcpyHostToDevice, compute_st_);

        // Updating the best state cost lookup table for the initial token state
        cudaMemcpyAsync(&d_state_best_cost_[first_token_state], &first_token_cost, sizeof(IntegerCostType),
                        cudaMemcpyHostToDevice, compute_st_);

        // We have one token is the aux_q
        int32 one = 1;
        cudaMemcpyAsync(d_aux_q_end_, &one, sizeof(int32), cudaMemcpyHostToDevice, compute_st_);
        *h_aux_q_end_ = 1;

        // The main_q is empty
        cudaMemsetAsync(d_main_q_end_, 0, sizeof(int32), compute_st_);
        *h_main_q_end_ = 0;
        cudaMemsetAsync(d_main_q_narcs_, 0, sizeof(int32), compute_st_);
        *h_main_q_narcs_ = 0;
        cudaMemsetAsync(d_main_q_local_offset_, 0, sizeof(int32), compute_st_);
        *h_main_q_local_offset_ = 0;
        
        // Resetting the TokenInfoVector in the CPU host
        h_all_tokens_info_.Reset();
        main_q_global_offset_ = 0;

        // Initializing flag
        *h_q_overflow_ = 0;

        cudaMemsetAsync(d_n_CTA_done_, 0, sizeof(int32), compute_st_);

        cudaStreamSynchronize(compute_st_);
        cudaCheckError();

        num_frames_decoded_ = 0;

        ProcessNonemitting();

        int32 main_q_size = *h_main_q_end_;
        h_all_tokens_info_.CopyFromDevice(main_q_global_offset_, d_main_q_info_, main_q_size);
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

        int32 prev_main_q_size = *h_main_q_end_;
        while (num_frames_decoded_ < target_frames_decoded) {
            
            // Computing a new frame

            // Loglikelihoods from the acoustic model
            ComputeLogLikelihoods(decodable);

            // ProcessEmitting 
            // 
            // Before executing ProcessEmitting, we have :
            // - The main_q contains tokens from the last frame
            // - The aux_q is empty
            //
            // ProcessEmitting will do the operation :
            //
            // read tokens from main_q ----FST---> create new tokens in the aux_q
            //
            // We will not write in the main q in that step
            // The input tokens are already in the main_q
            // (they were put there by the ProcessNonemittings 
            // from the previous frame)
            // We don't need can_write_to_main_q_
            // because we won't write to the main_q
            // The output tokens will go to aux_q
            ProcessEmitting();

            // After ProcessEmitting we won't need the token
            // associated with the previous frame anymore
            // At the end of ProcessEmitting the main_q was flushed 
            // (by setting main_q_end == 0)
            // Tokens that were flushed at that step have been previously 
            // moved to the host memory 
            // We update the global offset of the main_q
            // the global offset takes into account all tokens that have been moved
            // to the host memory

            main_q_global_offset_ += prev_main_q_size;
            
            // ProcessNonemitting
            //
            // Processing non emitting arcs
            //
            // The operation is :
            //
            // read input tokens from aux_q 
            //     ---contract (prune)--->
            // write non-pruned input tokens to main_q (append at the end of the queue)
            // read input tokens from main_q 
            //     ---FST--->
            // create new tokens in the aux_q
            //
            // We then iterate those operations until no new tokens are created 

            // We will write to main_q. We need it to be ready
            cudaEventSynchronize(can_write_to_main_q_);
            ProcessNonemitting(); 
            
            prev_main_q_size = *h_main_q_end_;
            
            // We are done with the current frame
            // We copy back its  tokens to the host
            // We only copy the "info" part (arc_idx + prev_token)
            // because we don't need anything else for the final backtrack
            h_all_tokens_info_.CopyFromDevice(main_q_global_offset_, d_main_q_info_, prev_main_q_size);
            cudaEventRecord(can_write_to_main_q_, copy_st_);
            
            num_frames_decoded_++; 
        }   


        nvtxRangePop();
    }


    void CudaDecoder::ComputeLogLikelihoods(DecodableInterface *decodable) {

        int32 frame = num_frames_decoded_;

        decodable->ComputeLogLikelihoods(d_loglikelihoods_,frame,fst_.max_ilabel+1, compute_st_);
    }

    void CudaDecoder::PrintOverflowWarning() {

        KALDI_WARN << "Preventing overflow of the frame tokens. Pursuing "
            << "execution but the quality of the output may be decreased. "
            << "To prevent this from happening, please increase the parameter --max-tokens-per-frame"
            << " and/or decrease --beam";
    }


    //
    // ProcessToken does the heavy work of ProcessEmitting and ProcessNonemitting
    // The only differences between emitting and non emitting case are the CSR offsets
    // and the boolean params.is_emitting
    //
    // It first calls the appropriate Preprocess kernel
    // and then ExpandArc to traverse the arcs
    //
    bool CudaDecoder::ProcessToken(bool is_emitting) {

        // Traversing either emitting or nonemitting arcs of the FST
        unsigned int *d_arc_offsets = is_emitting ? fst_.d_e_offsets : fst_.d_ne_offsets;

        PreprocessParams preprocess_params;
        preprocess_params.d_aux_q_state = d_aux_q_state_; 
        preprocess_params.d_aux_q_cost = d_aux_q_cost_;
        preprocess_params.d_aux_q_info = d_aux_q_info_; 
        preprocess_params.d_aux_q_end = d_aux_q_end_;
        preprocess_params.h_aux_q_end = h_aux_q_end_;
        preprocess_params.d_main_q_state = d_main_q_state_; 
        preprocess_params.d_main_q_cost = d_main_q_cost_;
        preprocess_params.d_main_q_info = d_main_q_info_; 
        preprocess_params.d_main_q_end_and_narcs_i2 = d_main_q_end_and_narcs_i2_; 
        preprocess_params.d_main_q_narcs = d_main_q_narcs_;
        preprocess_params.d_main_q_end = d_main_q_end_;
        preprocess_params.h_main_q_end = h_main_q_end_;
        preprocess_params.d_main_q_local_offset = d_main_q_local_offset_;
        preprocess_params.h_main_q_local_offset = h_main_q_local_offset_;
        preprocess_params.h_main_q_narcs = h_main_q_narcs_;
        preprocess_params.q_capacity = max_tokens_per_frame_;
        preprocess_params.h_q_overflow = h_q_overflow_;
        preprocess_params.d_main_q_degrees_prefix_sum = d_main_q_degrees_prefix_sum_; 
        preprocess_params.d_arc_offsets = d_arc_offsets;
        preprocess_params.d_main_q_arc_offsets = d_main_q_arc_offsets_;
        preprocess_params.d_state_best_cost = d_state_best_cost_; 
        preprocess_params.d_cutoff = d_cutoff_; 
        preprocess_params.d_main_q_degrees_block_sums_prefix_sum = d_main_q_degrees_block_sums_prefix_sum_; 
        preprocess_params.d_n_CTA_done = d_n_CTA_done_;

        if(is_emitting) {
            // If we are emitting, all tokens from previous frame are already
            // in the main_q. They were put there by PreprocessAndContracts,
            // in the ProcessNonEmittings from previous frames 
            // Those tokens already have children (through the ProcessNonemittings)
            // we cannot prune them - it would break the prev_token indexes
            // We do the PreprocessInPlace, we read from main_q and do not modify it
            // 
            PreprocessInPlace(preprocess_params);
            // Preprocess sets h_main_q_narcs (pinned memory) at the end of its executing
            cudaEventRecord(can_read_h_main_q_narcs_, compute_st_);

            // Resetting the lookup table for the new frame
            ResetStateCostLookupAndFinalizePreprocessInPlace();
        } else {
            PreprocessAndContract(preprocess_params);
            
            // Preprocess sets h_main_q_narcs (pinned memory) at the end of its executing
            cudaEventRecord(can_read_h_main_q_narcs_, compute_st_);

            // No need to call FinalizePreprocessInPlace - the prefix sum is done in one pass in PreprocessAndContract
        }


        // We need h_main_q_narcs to be ready
        cudaEventSynchronize(can_read_h_main_q_narcs_);
        cudaCheckError();

        int32 main_q_narcs = *h_main_q_narcs_;
        int32 q_overflow = *h_q_overflow_;

        if(q_overflow) {
            // An overflow was prevented in the contract and preprocess kernel
            // The algorithm can still go on but quality of the result can be reduced
            // (less tokens were generated)

            PrintOverflowWarning();

            *h_q_overflow_ = 0;
        }

        ExpandArcParams expand_params;
        expand_params.d_main_q_state = d_main_q_state_;
        expand_params.d_main_q_cost = d_main_q_cost_;
        expand_params.d_main_q_info= d_main_q_info_;
        expand_params.d_main_q_local_offset = d_main_q_local_offset_;
        expand_params.h_main_q_local_offset = h_main_q_local_offset_;
        expand_params.main_q_global_offset = main_q_global_offset_;
        expand_params.d_main_q_end = d_main_q_end_;
        expand_params.d_main_q_narcs = d_main_q_narcs_;
        expand_params.h_main_q_end = h_main_q_end_;
        expand_params.h_main_q_narcs = h_main_q_narcs_;
        expand_params.d_aux_q_state = d_aux_q_state_; 
        expand_params.d_aux_q_cost = d_aux_q_cost_; 
        expand_params.d_aux_q_info = d_aux_q_info_;
        expand_params.d_aux_q_end = d_aux_q_end_;
        expand_params.h_aux_q_end = h_aux_q_end_;
        expand_params.q_capacity = max_tokens_per_frame_;
        expand_params.h_q_overflow = h_q_overflow_;
        expand_params.d_main_q_degrees_prefix_sum = d_main_q_degrees_prefix_sum_; 
        expand_params.d_q_arc_offsets = d_main_q_arc_offsets_;
        expand_params.arc_ilabels = fst_.d_arc_ilabels;
        expand_params.is_emitting = is_emitting;
        expand_params.arc_weights = fst_.d_arc_weights; 
        expand_params.arc_nextstates = fst_.d_arc_nextstates; 
        expand_params.d_cutoff = d_cutoff_;
        expand_params.beam = beam_;
        expand_params.d_loglikelihoods = d_loglikelihoods_;
        expand_params.d_state_best_cost = d_state_best_cost_;
        expand_params.d_n_CTA_done = d_n_CTA_done_;
    

        // If we are non emitting and have a small number of arcs to process,
        // we use FinalizeProcessNonemitting
        // it is a persistent kernel which will iterate on Preprocess/ExpandArc
        // internally, using in-block syncs
        // it avoids calling too many kernels during the multiple iterations of NonEmitting
        // In any cases, we need to call FinalizeProcessNonemitting to finalize ProcessNonemitting
        // it takes care of finalizing our work for the frame (moving offsets, etc.)
        bool done = false;
        if(!is_emitting 
                && main_q_narcs < KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS) { 
            FinalizeProcessNonemitting(d_arc_offsets, expand_params); 

            cudaCheckError();

            // Persistent kernel finishes the job
            done = true;
        }
        else {
            ExpandArcs(expand_params, main_q_narcs);
        }

        cudaStreamSynchronize(compute_st_); 
        cudaCheckError();

        q_overflow = *h_q_overflow_;

        if(q_overflow) {
            // An overflow was prevented in the contract and preprocess kernel
            // The algorithm can still go on but quality of the result can be reduced
            // (less tokens were generated)

            PrintOverflowWarning();

            *h_q_overflow_ = 0;
        }
 
        return done;
    }


    void CudaDecoder::ProcessEmitting() {
        nvtxRangePushA("ProcessEmitting");

        // true => use emitting arcs
        ProcessToken(true); 

        cudaCheckError();
        nvtxRangePop();
    }

    void CudaDecoder::ProcessNonemitting() {
        nvtxRangePushA("ProcessNonemitting");

        // While not done, call it
        // If remaining n_arcs < KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS, 
        // ProcessToken will call a persistent kernel
        // false => use non emitting arcs
        while(!ProcessToken(false));

        cudaCheckError();
        nvtxRangePop();
    }

    // GetBestCost
    // CPU-only code
    // returns the minimum cost among all tokens cost in the current frame
    // also returns the index of one token with that min cost
    //
    // Only called at the end of the computation of one audio file
    // not optimized
    //
    void CudaDecoder::GetBestCost(bool isfinal, BaseFloat *min, int32 *argmin) const {
        
        CostType best_cost = std::numeric_limits<CostType>::max();
        int32 min_cost_token_index;

        // we need h_main_q_end_ ready
        cudaStreamSynchronize(compute_st_);

        // Copying the costs from current frame back to host memory
        // h_main_q_cost_ is never filled automatically 
        // when moving the tokens back to the host, we only move the { arc_idx, prev_token } part
        int32 main_q_size = *h_main_q_end_;
        cudaMemcpyAsync(h_main_q_cost_, 
                        d_main_q_cost_, 
                        main_q_size * sizeof(CostType), 
                        cudaMemcpyDeviceToHost,
                        compute_st_);

        if(isfinal)
            cudaMemcpyAsync(h_main_q_state_,     
                            d_main_q_state_, 
                            main_q_size * sizeof(int32), 
                            cudaMemcpyDeviceToHost,
                            compute_st_);

        // Waiting for data
        cudaStreamSynchronize(compute_st_);


        // Finding best cost
        for(int32 i=0; i < main_q_size; ++i) {
            CostType cost = h_main_q_cost_[i];

            if(isfinal) 
                cost += fst_.h_final[h_main_q_state_[i]];

            if(cost < best_cost) {
                best_cost = cost;
                min_cost_token_index = i;
            }
        }

        // The main_q always has a main_q_global_offset_
        min_cost_token_index += main_q_global_offset_; 

        // Saving result
        *min = best_cost;
        *argmin = min_cost_token_index;
    }


    //
    // ReachedFinal() returns true if the main_q contains a final state 
    // CPU-only code
    //
    // Only called at the end of the computation of one audio file
    // not optimized
    //
    bool CudaDecoder::ReachedFinal() const {
        // we need h_main_q_end_ ready
        cudaStreamSynchronize(compute_st_);

        int32 main_q_size = *h_main_q_end_;
        
        // Copying the states from current frame back to host memory
        // h_main_q_state_ is never filled automatically 
        // when moving the tokens back to the host, we only move the { arc_idx, prev_token } part
        cudaMemcpyAsync(h_main_q_state_,     
                d_main_q_state_, 
                main_q_size * sizeof(int32), 
                cudaMemcpyDeviceToHost,
                compute_st_);

        // Waiting for data
        cudaStreamSynchronize(compute_st_);

        // Looking for a final state
        for(int32 i=0; i < main_q_size; ++i) {
            if(fst_.h_final[h_main_q_state_[i]] != StdWeight::Zero().Value())
                return true;
        }

        return false;
    }



    //
    // GetBestPath is called at the end of the computation
    // It chooses the best token from the last frame, 
    // and backtracks all the path to the beginning (StartState)
    // from there
    // It then returns that path
    //
    bool CudaDecoder::GetBestPath(Lattice *fst_out, bool use_final_probs) const {
        nvtxRangePushA("GetBestPath");

        // We want the copy to host of the last tokens to be done
        cudaEventSynchronize(can_write_to_main_q_);

        bool isfinal = ReachedFinal();

        // Finding the best token from the last frame
        // ie the token with min cost
        BaseFloat best_cost;
        int32 token_with_best_cost;
        GetBestCost(isfinal, &best_cost, &token_with_best_cost);


        // Backtracking
        // Going all the way from the token with best cost
        // to the beginning (StartState)
        int32 token_idx = token_with_best_cost;
        int32 last_valid_token_idx;
        std::vector<int32> reversed_path;

        // The first token was inserted at the beginning of the queue
        // it always has index 0
        // We backtrack until that first token
        while(token_idx != 0) {
            last_valid_token_idx = token_idx;
            int32 arc_idx = h_all_tokens_info_.GetRawPointer()[token_idx].arc_idx;
            reversed_path.push_back(arc_idx);
            token_idx = h_all_tokens_info_.GetRawPointer()[token_idx].prev_token;
        }


        // Reset the fst_out
        fst_out->DeleteStates();

        // Building the output Lattice
        StateId cur_state = fst_out->AddState();
        fst_out->SetStart(cur_state);

        for (int32 i = reversed_path.size()-1; i >= 1; i--) {
            int32 arc_idx = reversed_path[i];

            LatticeArc arc(fst_.h_arc_ilabels[arc_idx], 
                           fst_.h_arc_olabels[arc_idx],
                           LatticeWeight(fst_.h_arc_weights[arc_idx], 0), 
                           fst_.h_arc_nextstates[arc_idx]);

            arc.nextstate = fst_out->AddState();
            fst_out->AddArc(cur_state, arc);
            cur_state = arc.nextstate;
        }

        // Adding final cost to final state
        if (isfinal && use_final_probs)
            fst_out->SetFinal(cur_state,
                    LatticeWeight(fst_.h_final[fst_.h_arc_nextstates[reversed_path[0]]], 0.0));
        else
            fst_out->SetFinal(cur_state, LatticeWeight::One());

        fst::RemoveEpsLocal(fst_out);

        nvtxRangePop();
        return true;
    }


} // end namespace kaldi.
