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
#include <algorithm>
#include <cub/cub.cuh>


#define MEMADVISE

#define KALDI_CUDA_DECODER_DIV_ROUND_UP(a,b) ((a+b-1)/b)

namespace kaldi {

    CudaDecoder::CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config): fst_(fst), 
                     default_beam_(config.default_beam),
                     max_tokens_(config.max_tokens), 
                     max_tokens_per_frame_(config.max_tokens_per_frame),
                     h_all_tokens_info_(config.max_tokens) {
        //
        // For a description of the class members, please refer to the cuda-decoder.h file
        //
        cudaStreamCreate(&compute_st_);
        cudaStreamCreate(&copy_st_); 

        cudaEventCreate(&can_read_h_main_q_narcs_);
        cudaEventCreate(&can_write_to_main_q_);
        cudaEventCreate(&can_read_final_h_main_q_end_);
        cudaEventCreate(&before_finalize_nonemitting_kernel_);

        cudaMalloc(&d_main_q_state_, max_tokens_per_frame_ * sizeof(*d_main_q_state_));
        cudaMallocHost(&h_main_q_state_, max_tokens_per_frame_ * sizeof(*h_main_q_state_));
        cudaMalloc(&d_aux_q_state_, max_tokens_per_frame_ * sizeof(*d_aux_q_state_));

        cudaMalloc(&d_main_q_cost_, max_tokens_per_frame_ * sizeof(*d_main_q_cost_));
        cudaMallocHost(&h_main_q_cost_, max_tokens_per_frame_ * sizeof(*h_main_q_cost_));
        cudaMalloc(&d_aux_q_cost_, max_tokens_per_frame_ * sizeof(*d_aux_q_cost_));

        cudaMalloc(&d_main_q_info_, max_tokens_per_frame_ * sizeof(*d_main_q_info_));
        cudaMalloc(&d_aux_q_info_, max_tokens_per_frame_ * sizeof(*d_aux_q_info_));

        cudaMalloc(&d_main_q_local_offset_, sizeof(*d_main_q_local_offset_));
        cudaMalloc(&d_aux_q_end_, sizeof(*d_aux_q_end_));
        cudaMalloc(&d_n_CTA_done_, sizeof(*d_n_CTA_done_));

        cudaMalloc(&d_main_q_end_and_narcs_i2_, sizeof(*d_main_q_end_and_narcs_i2_));
        d_main_q_narcs_ = &d_main_q_end_and_narcs_i2_->split.narcs;
        d_main_q_end_ = &d_main_q_end_and_narcs_i2_->split.ntokens;

        cudaMalloc(&d_global_min_cost_and_beam_, sizeof(*d_global_min_cost_and_beam_));

        // TODO alloc once, for all small vals, better data locality
        cudaMallocHost(&h_main_q_end_, sizeof(*h_main_q_end_));  
        cudaMallocHost(&h_main_q_narcs_, sizeof(*h_main_q_narcs_));  
        cudaMallocHost(&h_main_q_local_offset_, sizeof(*h_main_q_local_offset_));  
        cudaMallocHost(&h_aux_q_end_, sizeof(*h_aux_q_end_));  
        cudaMallocHost(&h_main_q_end_before_finalize_nonemitting_kernel_, sizeof(*h_main_q_end_before_finalize_nonemitting_kernel_));  

        cudaMallocHost(&h_q_overflow_, sizeof(h_q_overflow_));  

        cudaMalloc(&d_main_q_degrees_prefix_sum_, max_tokens_per_frame_ * sizeof(*d_main_q_degrees_prefix_sum_));

        // d_main_q_degrees_block_sums_prefix_sum_ is the prefix sum of the "block sums"
        // a block sum is, for each CUDA block, the sum of the arc degrees of all tokens associated to that CUDA block
        // we add +1 because we want the last element of the prefix sum (a prefix sum of n elements generates (n+1) values)
        cudaMalloc(&d_main_q_degrees_block_sums_prefix_sum_, 
                (KALDI_CUDA_DECODER_DIV_ROUND_UP(max_tokens_per_frame_, KALDI_CUDA_DECODER_KERNEL_PREPROCESS_DIMX) + 1)
                 * sizeof(*d_main_q_degrees_block_sums_prefix_sum_));

        cudaMalloc(&d_main_q_arc_offsets_, (max_tokens_per_frame_+1) * sizeof(*d_main_q_arc_offsets_));

        cudaMalloc(&d_loglikelihoods_, (fst_.max_ilabel_+1)*sizeof(*d_loglikelihoods_));  

        cudaMalloc(&d_state_best_cost_, fst_.num_states_*sizeof(*d_state_best_cost_));

        h_all_tokens_info_.SetCudaStream(copy_st_);

        if(KALDI_CUDA_DECODER_DEBUG_LEVEL > 0) {
            KALDI_LOG << "Running the decoder in debug level " << KALDI_CUDA_DECODER_DEBUG_LEVEL;
                     
            uint32_t debug_buffer_queue_size = max_tokens_per_frame_ + 1;
            cudaMallocHost(&h_debug_buf1_, std::max(fst_.num_states_, debug_buffer_queue_size) * sizeof(h_debug_buf1_));
            cudaMallocHost(&h_debug_buf2_, debug_buffer_queue_size * sizeof(h_debug_buf2_));
        }

        KALDI_DECODER_CUDA_CHECK_ERROR();


        // Used as +INF for min_cost and d_state_cost
        // we will compute min_cost + beam during computation
        // if min_cost == FLT_MAX, we have an overflow
        // avoiding that by removing the beam from infinite
        // (2* the beam in case of rounding error)
        infinite_cost_ = FLT_MAX - 2*config.default_beam;
        // Building the parameters structs
        // Used to launch the ExpandArc and Preprocess kernels

        preprocess_params_.d_aux_q_state = d_aux_q_state_; 
        preprocess_params_.d_aux_q_cost = d_aux_q_cost_;
        preprocess_params_.d_aux_q_info = d_aux_q_info_; 
        preprocess_params_.d_aux_q_end = d_aux_q_end_;
        preprocess_params_.h_aux_q_end = h_aux_q_end_;
        preprocess_params_.d_main_q_state = d_main_q_state_; 
        preprocess_params_.d_main_q_cost = d_main_q_cost_;
        preprocess_params_.d_main_q_info = d_main_q_info_; 
        preprocess_params_.d_main_q_end_and_narcs_i2 = d_main_q_end_and_narcs_i2_; 
        preprocess_params_.d_main_q_narcs = d_main_q_narcs_;
        preprocess_params_.d_main_q_end = d_main_q_end_;
        preprocess_params_.h_main_q_end = h_main_q_end_;
        preprocess_params_.h_main_q_end_before_finalize_nonemitting_kernel = h_main_q_end_before_finalize_nonemitting_kernel_;
        preprocess_params_.d_main_q_local_offset = d_main_q_local_offset_;
        preprocess_params_.h_main_q_local_offset = h_main_q_local_offset_;
        preprocess_params_.h_main_q_narcs = h_main_q_narcs_;
        preprocess_params_.q_capacity = max_tokens_per_frame_;
        preprocess_params_.h_q_overflow = h_q_overflow_;
        preprocess_params_.d_main_q_degrees_prefix_sum = d_main_q_degrees_prefix_sum_; 
        preprocess_params_.d_main_q_arc_offsets = d_main_q_arc_offsets_;
        preprocess_params_.d_state_best_cost = d_state_best_cost_; 
        preprocess_params_.d_global_min_cost_and_beam = d_global_min_cost_and_beam_; 
        preprocess_params_.d_main_q_degrees_block_sums_prefix_sum = d_main_q_degrees_block_sums_prefix_sum_; 
        preprocess_params_.d_n_CTA_done = d_n_CTA_done_;
        preprocess_params_.infinite_cost = infinite_cost_;

        expand_params_.d_main_q_state = d_main_q_state_;
        expand_params_.d_main_q_cost = d_main_q_cost_;
        expand_params_.d_main_q_info= d_main_q_info_;
        expand_params_.d_main_q_local_offset = d_main_q_local_offset_;
        expand_params_.h_main_q_local_offset = h_main_q_local_offset_;
        expand_params_.d_main_q_end = d_main_q_end_;
        expand_params_.d_main_q_narcs = d_main_q_narcs_;
        expand_params_.h_main_q_end = h_main_q_end_;
        expand_params_.h_main_q_narcs = h_main_q_narcs_;
        expand_params_.d_aux_q_state = d_aux_q_state_; 
        expand_params_.d_aux_q_cost = d_aux_q_cost_; 
        expand_params_.d_aux_q_info = d_aux_q_info_;
        expand_params_.d_aux_q_end = d_aux_q_end_;
        expand_params_.h_aux_q_end = h_aux_q_end_;
        expand_params_.q_capacity = max_tokens_per_frame_;
        expand_params_.h_q_overflow = h_q_overflow_;
        expand_params_.d_main_q_degrees_prefix_sum = d_main_q_degrees_prefix_sum_; 
        expand_params_.d_q_arc_offsets = d_main_q_arc_offsets_;
        expand_params_.arc_ilabels = fst_.d_arc_ilabels_;
        expand_params_.arc_weights = fst_.d_arc_weights_; 
        expand_params_.arc_nextstates = fst_.d_arc_nextstates_; 
        expand_params_.d_global_min_cost_and_beam = d_global_min_cost_and_beam_; 
        expand_params_.default_beam = default_beam_;
        expand_params_.d_loglikelihoods = d_loglikelihoods_;
        expand_params_.d_state_best_cost = d_state_best_cost_;
        expand_params_.d_n_CTA_done = d_n_CTA_done_;

    }

    CudaDecoder::~CudaDecoder() {
        cudaStreamDestroy(compute_st_);
        cudaStreamDestroy(copy_st_);

        cudaEventDestroy(can_read_h_main_q_narcs_);
        cudaEventDestroy(can_write_to_main_q_);
        cudaEventDestroy(can_read_final_h_main_q_end_);
        cudaEventDestroy(before_finalize_nonemitting_kernel_);

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
        cudaFree(d_main_q_degrees_prefix_sum_);
        cudaFree(d_main_q_degrees_block_sums_prefix_sum_);
        cudaFree(d_main_q_arc_offsets_);
        cudaFree(d_loglikelihoods_);
        cudaFree(d_state_best_cost_);
        cudaFree(d_global_min_cost_and_beam_);

        cudaFreeHost(h_main_q_end_);
        cudaFreeHost(h_main_q_narcs_);
        cudaFreeHost(h_main_q_local_offset_);
        cudaFreeHost(h_aux_q_end_);
        cudaFreeHost(h_main_q_cost_);
        cudaFreeHost(h_main_q_state_);

        if(KALDI_CUDA_DECODER_DEBUG_LEVEL > 0) {
            cudaFreeHost(h_debug_buf1_);
            cudaFreeHost(h_debug_buf2_);
        }
        KALDI_DECODER_CUDA_CHECK_ERROR();
    }

    void CudaDecoder::InitDecoding() {
        cudaStreamSynchronize(compute_st_);

        // Filling the best state cost lookup table with +INF
        InitStateBestCostLookup();

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
        cudaMemcpyAsync(&d_state_best_cost_[first_token_state], 
                        &first_token_cost, 
                        sizeof(IntegerCostType),
                        cudaMemcpyHostToDevice, 
                        compute_st_);

        // We have one token is the aux_q
        int32 aux_q_end = 1;
        cudaMemcpyAsync(d_aux_q_end_, &aux_q_end, sizeof(*d_aux_q_end_), cudaMemcpyHostToDevice, compute_st_);
        *h_aux_q_end_ = aux_q_end;

        // The main_q is empty
        cudaMemsetAsync(d_main_q_end_, 0, sizeof(*d_main_q_end_), compute_st_);
        *h_main_q_end_ = 0;
        cudaMemsetAsync(d_main_q_narcs_, 0, sizeof(*d_main_q_narcs_), compute_st_);
        *h_main_q_narcs_ = 0;
        cudaMemsetAsync(d_main_q_local_offset_, 0, sizeof(*d_main_q_local_offset_), compute_st_);
        *h_main_q_local_offset_ = 0;
        
        // Resetting the TokenInfoVector on host
        h_all_tokens_info_.Reset();
        main_q_global_offset_ = 0;

        // Initializing flag
        *h_q_overflow_ = 0;

        cudaMemsetAsync(d_n_CTA_done_, 0, sizeof(*d_n_CTA_done_), compute_st_);

        KALDI_DECODER_CUDA_CHECK_ERROR();

        num_frames_decoded_ = 0;

        // Initial ProcessNonEmitting
        PreprocessAndContract(aux_q_end);
        cudaStreamSynchronize(compute_st_);
        int main_q_narcs = *h_main_q_narcs_;
        
        while(main_q_narcs > KALDI_CUDA_DECODER_KERNEL_NONEM_LT_DIMX) {
            ExpandArcs(false, main_q_narcs);
            PreprocessAndContract(main_q_narcs);

            cudaStreamSynchronize(compute_st_);
            main_q_narcs = *h_main_q_narcs_;
        }

        FinalizeProcessNonemitting(); 
    
        cudaStreamSynchronize(compute_st_);
        PreprocessInPlace(*h_main_q_end_);
        cudaEventRecord(can_read_h_main_q_narcs_, compute_st_);
        // Resetting the lookup table for the new frame
        ResetStateBestCostLookupAndFinalizePreprocessInPlace(*h_main_q_end_);

        // Saving initial queue to host
        int32 main_q_size = *h_main_q_end_;
        h_all_tokens_info_.CopyFromDevice(main_q_global_offset_, d_main_q_info_, main_q_size);
        cudaEventRecord(can_write_to_main_q_, copy_st_);
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

        // Loglikelihoods from the acoustic model
        ComputeLogLikelihoods(decodable);

        int32 prev_main_q_size = *h_main_q_end_;
        nvtxRangePushA("Decoding");
        while (num_frames_decoded_ < target_frames_decoded) {
            // Computing a new frame
        
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

            // ProcessEmitting generates tokens associated with the new frame i
            // When we call ProcessEmitting, the main_q contains the tokens associated
            // with the previous frame (i-1). Using d_main_q_state and the emitting arcs from the FST graph,
            // we create a new tokens queue, which will be stored in the aux_q

            cudaEventSynchronize(can_read_h_main_q_narcs_);
            int32 main_q_n_e_arcs = *h_main_q_narcs_;

            if(KALDI_CUDA_DECODER_DEBUG_LEVEL > 1)
                DebugAssertsNewFrame();

            if(KALDI_CUDA_DECODER_DEBUG_LEVEL > 0)
                DebugAssertsBeforeExpand(true);

            ExpandArcs(true, main_q_n_e_arcs);

            // Loglikelihoods from the acoustic model
            // We are done using loglikelihoods for current frame
            // Launching kernel for next frame now if there is one
            
            nvtxRangePop();
            if ((num_frames_decoded_+1) < target_frames_decoded) 
                ComputeLogLikelihoods(decodable);
            nvtxRangePushA("Decoding");

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
            // PreprocessAndContract:
            // read input tokens from aux_q 
            //     ---contract (prune)--->
            // write non-pruned input tokens to main_q (append at the end of the queue)
            //
            // ExpandArc:
            // read input tokens from main_q 
            //     ---FST--->
            // create new tokens in the aux_q
            //
            // We then iterate those operations until no new tokens are created 
            //

            // We will write to main_q. We need it to be ready
            // for next kernels on compute_st_ 
            cudaStreamWaitEvent(compute_st_, can_write_to_main_q_, 0);

            // Using that value as estimate for how many threads to launch
            int main_q_size_narcs_estimate = main_q_n_e_arcs;

            bool finalize_nonemitting_was_executed = false;
            PreprocessAndContract(main_q_size_narcs_estimate);
            bool first_attempt = true;
            while(!finalize_nonemitting_was_executed) {
                // We push to the pipeline a fixed number of NonEmitting iterations
                // If that's not enough, we'll detect it and push some more
                int32 to_launch = first_attempt 
                                  ? KALDI_CUDA_DECODER_NONEM_NEXPAND_PIPELINE_FIRST
                                  : KALDI_CUDA_DECODER_NONEM_NEXPAND_PIPELINE_RELAUNCH;

                for(int32 i=0; i<to_launch; ++i) {
                    if(KALDI_CUDA_DECODER_DEBUG_LEVEL > 0)
                        DebugAssertsBeforeExpand(false);

                    ExpandArcs(false, main_q_size_narcs_estimate);
                    PreprocessAndContract(main_q_size_narcs_estimate);
                }

                cudaEventRecord(before_finalize_nonemitting_kernel_, compute_st_);
                // Try to compute the final iterations of NonEmitting
                // if too many arcs still need to be processed,
                // this kernel stops and we'll run more "heavy load" iterations
                FinalizeProcessNonemitting(); 
               
                cudaEventSynchronize(before_finalize_nonemitting_kernel_);

                // If the number of arcs was above that value, 
                // the finalize nonemitting kernel killed itself 
                // and we must relaunch heavy load kernels (ExpandArcs)
                int main_q_narcs = *h_main_q_narcs_;
                finalize_nonemitting_was_executed = (main_q_narcs < KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS);
                first_attempt = false;
            }

            // The final h_main_q_end_ for this frame is written in finalize nonemitting
            cudaEventRecord(can_read_final_h_main_q_end_, compute_st_);


            // Before FinalizeProcessNonEmitting most of the main_q has been built
            // We use value of main_q_end before this kernel as an estimate,
            // Note : This is not an upper bound, FinalizeProcessNonEmitting can generates
            // more than KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS tokens
            int main_q_size_estimate = *h_main_q_end_before_finalize_nonemitting_kernel_
                                        + KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS;
            
            // PreprocessInPlace for next ProcessEmitting
            // We do it here (and not at the beginning of the loop) for two reasons :
            // - We can pipeline it in the CUDA queue
            // - The lookup table is back to its original state (full of +INF) at 
            //   the end of each frame. We can possibly share it between decoders
            //   (not implemented yet)
            PreprocessInPlace(main_q_size_estimate);
            cudaEventRecord(can_read_h_main_q_narcs_, compute_st_);

            // Resetting the lookup table for the next frame + FinalizePreprocessInPlace
            ResetStateBestCostLookupAndFinalizePreprocessInPlace(main_q_size_estimate);
 
            // We need the exact value of h_main_q_end_ for the copy to host
            cudaEventSynchronize(can_read_final_h_main_q_end_);
            prev_main_q_size = *h_main_q_end_;
            
            // We are done with the current frame
            // We copy back its  tokens to the host
            // We only copy the "info" part (arc_idx + prev_token)
            // because we don't need anything else for the final backtrack
            h_all_tokens_info_.CopyFromDevice(main_q_global_offset_, d_main_q_info_, prev_main_q_size);
            cudaEventRecord(can_write_to_main_q_, copy_st_);
            
            CheckOverflow();
            KALDI_DECODER_CUDA_CHECK_ERROR();
            num_frames_decoded_++; 
        }   
    
        nvtxRangePop();
    }


    void CudaDecoder::ComputeLogLikelihoods(DecodableInterface *decodable) {
        int32 frame = num_frames_decoded_;

        decodable->ComputeLogLikelihoods(d_loglikelihoods_,frame,fst_.max_ilabel_+1, compute_st_);
    }

    void CudaDecoder::CheckOverflow() {
            int32 q_overflow = *h_q_overflow_;
            if(q_overflow) {
                // An overflow was prevented in a kernel
                // The algorithm can still go on but quality of the result can be reduced
                // (less tokens were generated)
                KALDI_WARN << "Preventing overflow of the frame tokens. Pursuing "
                    << "execution but the quality of the output may be decreased. "
                    << "To prevent this from happening, please increase the parameter --max-tokens-per-frame"
                    << " and/or decrease --beam";

                *h_q_overflow_ = 0;
            }

    }


    // GetBestCost
    // CPU-only code
    // returns the minimum cost among all tokens cost in the current frame
    // also returns the index of one token with that min cost
    //
    // Only called at the end of the computation of one audio file
    // not optimized
    //
    void CudaDecoder::GetBestCost(bool isfinal, CostType *min, int32 *argmin) const {
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
                        main_q_size * sizeof(*d_main_q_cost_), 
                        cudaMemcpyDeviceToHost,
                        compute_st_);

        if(isfinal)
            cudaMemcpyAsync(h_main_q_state_,     
                            d_main_q_state_, 
                            main_q_size * sizeof(*d_main_q_state_), 
                            cudaMemcpyDeviceToHost,
                            compute_st_);

        // Waiting for data
        cudaStreamSynchronize(compute_st_);


        // Finding best cost
        for(int32 i=0; i < main_q_size; ++i) {
            CostType cost = h_main_q_cost_[i];

            if(isfinal) 
                cost += fst_.h_final_[h_main_q_state_[i]];

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
                main_q_size * sizeof(*d_main_q_state_), 
                cudaMemcpyDeviceToHost,
                compute_st_);

        // Waiting for data
        cudaStreamSynchronize(compute_st_);

        // Looking for a final state
        for(int32 i=0; i < main_q_size; ++i) {
            if(fst_.h_final_[h_main_q_state_[i]] != StdWeight::Zero().Value())
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
        CostType best_cost;
        int32 token_with_best_cost;
        GetBestCost(isfinal, &best_cost, &token_with_best_cost);


        // Backtracking
        // Going all the way from the token with best cost
        // to the beginning (StartState)
        int32 token_idx = token_with_best_cost;
        std::vector<int32> reversed_path;

        // The first token was inserted at the beginning of the queue
        // it always has index 0
        // We backtrack until that first token
        while(token_idx != 0) {
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

            LatticeArc arc(fst_.h_arc_ilabels_[arc_idx], 
                           fst_.h_arc_olabels_[arc_idx],
                           LatticeWeight(fst_.h_arc_weights_[arc_idx], 0), 
                           fst_.h_arc_nextstates_[arc_idx]);

            arc.nextstate = fst_out->AddState();
            fst_out->AddArc(cur_state, arc);
            cur_state = arc.nextstate;
        }

        // Adding final cost to final state
        if (isfinal && use_final_probs)
            fst_out->SetFinal(cur_state,
                    LatticeWeight(fst_.h_final_[fst_.h_arc_nextstates_[reversed_path[0]]], 0.0));
        else
            fst_out->SetFinal(cur_state, LatticeWeight::One());

        fst::RemoveEpsLocal(fst_out);

        nvtxRangePop();
        return true;
    }

    //
    // Debug functions
    // Called to verify that intermediate values are valid 
    //

    void CudaDecoder::DebugAssertsBeforeExpand(bool is_emitting) {
        cudaStreamSynchronize(compute_st_);

        int32 main_q_end = *h_main_q_end_;
        int32 main_q_offset = *h_main_q_local_offset_;

        cudaMemcpyAsync(h_main_q_state_,     
                d_main_q_state_,
                main_q_end * sizeof(*d_main_q_state_), 
                cudaMemcpyDeviceToHost,
                compute_st_);

        unsigned int *h_arc_offsets = is_emitting ? fst_.h_e_offsets_ : fst_.h_ne_offsets_;

        int32 * h_prefix_sum = h_debug_buf1_;
        cudaMemcpyAsync(h_prefix_sum,     
                d_main_q_degrees_prefix_sum_, 
                (main_q_end+1) * sizeof(*d_main_q_degrees_prefix_sum_), 
                cudaMemcpyDeviceToHost,
                compute_st_);

        int32 * h_q_arc_offsets = h_debug_buf2_;
        cudaMemcpyAsync(h_q_arc_offsets,     
                d_main_q_arc_offsets_,
                main_q_end * sizeof(*d_main_q_arc_offsets_), 
                cudaMemcpyDeviceToHost,
                compute_st_);

        // Waiting for the copies
        cudaStreamSynchronize(compute_st_);

        for(int32 i = main_q_offset; i < main_q_end; ++i) {
            int32 state = h_main_q_state_[i];
            KALDI_ASSERT(state >= 0);
            KALDI_ASSERT(state < fst_.num_states_);


            KALDI_ASSERT(h_prefix_sum[i] >= 0);
            KALDI_ASSERT(h_prefix_sum[i] <= h_prefix_sum[i+1]); 
            int32 degree_in_prefix_sum = h_prefix_sum[i+1] - h_prefix_sum[i];
            int32 degree_in_fst = h_arc_offsets[state+1] - h_arc_offsets[state];

            // Testing for degree == 0, which is possible in preprocessinplace
            // only possible if is_emitting, nonemitting uses contractandpreprocess
            if(is_emitting) {
                KALDI_ASSERT(degree_in_prefix_sum == 0 || degree_in_prefix_sum == degree_in_fst);
                // if degree == 0 arc_offsets may not be valid, but we won't use it
                KALDI_ASSERT(degree_in_prefix_sum == 0 || h_arc_offsets[state] == h_q_arc_offsets[i]); 
            } else {
                KALDI_ASSERT(degree_in_prefix_sum == degree_in_fst);
                KALDI_ASSERT(h_arc_offsets[state] == h_q_arc_offsets[i]); 
            }
        }
    }
    
    void CudaDecoder::DebugAssertsNewFrame() {
        cudaStreamSynchronize(compute_st_);

        int32 float_inf_as_int = 2139095039; // TODO 

        int32 nstates = fst_.num_states_;

        int *h_state_best_cost = h_debug_buf1_;
        cudaMemcpyAsync(h_state_best_cost,     
                d_state_best_cost_,
                nstates * sizeof(*d_state_best_cost_), 
                cudaMemcpyDeviceToHost,
                compute_st_);
        cudaStreamSynchronize(compute_st_);

        for(int i=0; i<nstates; ++i)
            KALDI_ASSERT(h_state_best_cost[i] == float_inf_as_int);
    }

} // end namespace kaldi.
