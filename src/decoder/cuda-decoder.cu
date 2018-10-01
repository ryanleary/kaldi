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
#include "decoder/cuda-decoder-kernels.cu"
#include <algorithm>
#include <nvToolsExt.h>
#include <cuda_runtime_api.h>
#include <float.h>
#include <algorithm>
#include <cub/cub.cuh>

#define MEMADVISE

namespace kaldi {

	CudaDecoder::CudaDecoder(const CudaFst &fst, 
			const CudaDecoderConfig &config,
			int32 nlanes,
			int32 nchannels): fst_(fst), 
	default_beam_(config.default_beam),
	max_tokens_(config.max_tokens), 
	max_tokens_per_frame_(config.max_tokens_per_frame),
	nlanes_(nlanes),
	nchannels_(nchannels) {
		//
		// For a description of the class members, please refer to the cuda-decoder.h file
		//
		cudaStreamCreate(&compute_st_);
		cudaStreamCreate(&copy_st_); 

		cudaEventCreate(&can_read_h_main_q_narcs_);
		cudaEventCreate(&can_write_to_main_q_);
		cudaEventCreate(&can_read_final_h_main_q_end_);
		cudaEventCreate(&before_finalize_nonemitting_kernel_);

		KALDI_ASSERT(nlanes > 0);
		KALDI_ASSERT(nchannels > 0);

		++nchannels_; // allocating init_channel_params at the same time

		// Setting Kernel Params
		// sent to kernels by copy
		// Making sure we'll be able to send it to the kernels
		//KALDI_STATIC_ASSERT(sizeof(KernelParams) < KALDI_CUDA_DECODER_MAX_KERNEL_ARGUMENTS_BYTE_SIZE); TODO find include

		h_kernel_params_ = new KernelParams(nlanes_, 
				nchannels_, 
				max_tokens_per_frame_,
				fst_.max_ilabel_,
				fst_.num_states_);

		cudaMemsetAsync(h_kernel_params_->d_channels_counters.data(), 0, nchannels_*sizeof(h_kernel_params_->d_channels_counters.data()));
		cudaMemsetAsync(h_kernel_params_->d_lanes_counters.data(), 0, nlanes_*sizeof(h_kernel_params_->d_lanes_counters.data()));
		cudaMallocHost(&h_lanes_counters_, nlanes_ * sizeof(*h_lanes_counters_));
		cudaMallocHost(&h_channels_counters_, nchannels_ * sizeof(*h_channels_counters_));

		// Using last one as init_channel_params
		init_channel_id_ = nchannels_-1;
		// Initialize host tokens memory pools
		for(int ichannel=0; ichannel<nchannels_; ++ichannel)
			h_all_tokens_info_.emplace_back(max_tokens_, copy_st_);

		ComputeInitialChannel();
		--nchannels_; // removing the init_channel_params from general list

		KALDI_DECODER_CUDA_CHECK_ERROR();
		num_frames_decoded_.resize(nchannels_, 0);

		// Filling all best_state_cost with +INF
		int32 nstates = fst_.NumStates();
		init_state_best_cost_lookup_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(nstates, nlanes_),
						KALDI_CUDA_DECODER_1D_BLOCK,
						0,
						compute_st_>>>(*h_kernel_params_);

		// Making sure that everything is ready to use
		cudaStreamSynchronize(compute_st_);

		// TODO move to constructor
		h_kernel_params_->d_arc_offsets = fst_.d_arc_offsets_;
		h_kernel_params_->d_arc_ilabels = fst_.d_arc_ilabels_;
		h_kernel_params_->d_arc_weights = fst_.d_arc_weights_;
		h_kernel_params_->d_arc_nextstates = fst_.d_arc_nextstates_;
		h_kernel_params_->default_beam = default_beam_;
		h_kernel_params_->q_capacity = max_tokens_per_frame_; 
		h_kernel_params_->init_channel_id = init_channel_id_; 
		h_kernel_params_->max_nlanes = nlanes_; 
		h_kernel_params_->nstates = fst_.num_states; 

		if(KALDI_CUDA_DECODER_DEBUG_LEVEL > 0) {
			KALDI_LOG << "Running the decoder in debug level " << KALDI_CUDA_DECODER_DEBUG_LEVEL;

			uint32_t debug_buffer_queue_size = max_tokens_per_frame_ + 1;
			cudaMallocHost(&h_debug_buf1_, std::max(fst_.num_states_, debug_buffer_queue_size) * sizeof(h_debug_buf1_));
			cudaMallocHost(&h_debug_buf2_, debug_buffer_queue_size * sizeof(h_debug_buf2_));
		} 
	}

	CudaDecoder::~CudaDecoder() {
		cudaStreamDestroy(compute_st_);
		cudaStreamDestroy(copy_st_);

		cudaEventDestroy(can_read_h_main_q_narcs_);
		cudaEventDestroy(can_write_to_main_q_);
		cudaEventDestroy(can_read_final_h_main_q_end_);
		cudaEventDestroy(before_finalize_nonemitting_kernel_);

		cudaFreeHost(h_lane_params);
		cudaFreeHost(h_channel_params);

		// Will call the cudaFrees inside destructors 
		delete h_kernel_params_;

		if(KALDI_CUDA_DECODER_DEBUG_LEVEL > 0) {
			cudaFreeHost(h_debug_buf1_);
			cudaFreeHost(h_debug_buf2_);
		}

		KALDI_DECODER_CUDA_CHECK_ERROR();
	}

	void CudaDecoder::ComputeInitialChannel() {
		// Lane used to compute init_channel_id_
		int32 lane_id = 0;

		// Filling the best state cost lookup table with +INF
		InitStateBestCostLookup(lane_id);

		// Adding the start state to the initial token queue
		StateId first_token_state;
		CostType first_token_cost;
		InfoToken first_token_info;

		first_token_state = fst_.Start();
		first_token_cost = StdWeight::One().Value();
		first_token_info.prev_token = INT_MIN;
		first_token_info.arc_idx = -1;

		KALDI_ASSERT(first_token_state != fst::kNoStateId);

		// We add that initial token to the aux_q
		// it will be moved to the main_q during the ProcessNonemitting phase 
		// that will be called in a few lines
		//
		// Note : we launch copies in the compute stream here
		// It means that we want them to be in the main pipeline
		// compute_st_ is just a name - it's a generic CUDA stream
		cudaMemcpy(h_lane_params[lane_id].d_aux_q_state, &first_token_state, sizeof(StateId), cudaMemcpyHostToDevice);
		cudaMemcpy(h_lane_params[lane_id].d_aux_q_cost, &first_token_cost, sizeof(CostType), cudaMemcpyHostToDevice);
		cudaMemcpy(h_lane_params[lane_id].d_aux_q_info, &first_token_info, sizeof(InfoToken), cudaMemcpyHostToDevice);

		// Updating the best state cost lookup table for the initial token state
		cudaMemcpy(&h_lane_params[lane_id].d_state_best_cost[first_token_state], 
				&first_token_cost, 
				sizeof(IntegerCostType),
				cudaMemcpyHostToDevice);

		// We have one token is the aux_q
		int32 aux_q_end = 1;
		cudaMemcpy(&d_lane_params[lane_id].aux_q_end, &aux_q_end, sizeof(*d_aux_q_end_), cudaMemcpyHostToDevice);

		// Following kernels working channel_id
		h_kernel_params_->channel_to_compute[lane_id] = init_channel_id_;
		h_kernel_params_->nchannels = 1;

		// Initial ProcessNonEmitting
		PreprocessAndContract(aux_q_end);
		FinalizeProcessNonemitting(); 

		// Preparing for first frame + reverting back to init state (lookup table, etc.)
		int main_q_end;
		cudaMemcpy(&main_q_end, &d_channel_params[init_channel_id_].frame_final_main_q_end, sizeof(int32), cudaMemcpyDeviceToHost);
		PreprocessInPlace(main_q_end);
		ResetStateBestCostLookupAndFinalizePreprocessInPlace(main_q_end);

		// Saving init params on host
		cudaMemcpy(h_channel_params[init_channel_id_], d_channel_params[init_channel_id_], sizeof(ChannelParams), cudaMemcpyDeviceToHost);

		// Saving initial queue to host
		h_all_tokens_info_[init_channel_id_].CopyFromDevice(h_channel_params[init_channel_id_].d_main_q_info, main_q_size);

		// Waiting for copy to be done
		cudaStreamSynchronize(copy_st_);

		KALDI_DECODER_CUDA_CHECK_ERROR();
	}

	void CudaDecoder::InitDecoding(const std::vector<ChannelId> &channels) {
		KALDI_ASSERT(channels.size() < n_lanes_);

		// Size of the initial main_q_size
		int init_main_q_size = h_channel_params_[init_channel_id_].final_frame_main_q_end;
		dim3 grid,block;
		block.x = KALDI_CUDA_DECODER_GENERIC_DIMX;
		grid.x = KALDI_CUDA_DECODER_DIV_ROUND_UP(init_main_q_size, block.x);
		grid.z = channels.size(); 

		// Getting *h_kernel_params ready to use
		SetChannelsInKernelParams(channels);

		// Initializing the main_q_end and everything else needed
		// to get the channels ready to compute new utterances
		init_decoding_on_device_kernel_<<<grid,block>>>(*h_kernel_params);

		// Tokens from initial main_q needed on host
		for(ChannelId channel_id : channels)
			h_all_tokens_info_[channel_id].Clone(h_all_tokens_info_[init_channel_id_]);
	}

	void CudaDecoder::LoadChannelsStatesToLanesCPU() {
		for(const LaneId ilane=0; ilane<h_kernel_params->nlanes_used; ++ilane) {
			const ChannelId ichannel = h_kernel_params->channel_to_compute[ilane];
			h_lane_counters[ilane].main_q_narcs_and_end = h_channel_counters[ichannel].prev_main_q_narcs_and_end;
		}	
	}

	void CudaDecoder::SaveChannelsStatesFromLanesCPU() {
		for(const LaneId ilane=0; ilane<h_kernel_params->nlanes_used; ++ilane) {
			const ChannelId ichannel = h_kernel_params->channel_to_compute[ilane];
			h_channel_counters[ichannel].prev_main_q_narcs_and_end = h_lane_counters[ilane].main_q_narcs_and_end;
			h_channel_counters[ichannel].prev_main_q_global_offset = h_lane_counters[ilane].main_q_global_offset;
		}
	}

	void CudaDecoder::AdvanceDecoding(DecodableInterface *decodable,
			const std::vector<ChannelId> &channels,
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

		int32 nlanes_used = channels.size();
		// We can process at most n_lanes_ channels at the same time
		KALDI_ASSERT(nlanes_used < nlanes_);

		// Setting up the  *kernel_params
		SetChannelsInKernelParams(channels);

		// Getting the lanes ready to work with those channels  
		load_channels_state_in_lanes_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
						KALDI_CUDA_DECODER_ONE_WARP_BLOCK,
						0,
						compute_st_>>>(*kernel_params);

		LoadChannelsStateToLanesCPU();
		nvtxRangePushA("Decoding");

		while (num_frames_decoded_ < target_frames_decoded) {
			// Computing a new frame

			// Loglikelihoods from the acoustic model
			// FIXME for now we duplicate the loglikelihoods 
			// to all channels for perf. measurement. 
			// We must decide which design to adopt
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

			// ProcessEmitting generates tokens associated with the new frame i
			// When we call ProcessEmitting, the main_q contains the tokens associated
			// with the previous frame (i-1). Using d_main_q_state and the emitting arcs from the FST graph,
			// we create a new tokens queue, which will be stored in the aux_q

			// Process emitting, expanding arcs
			// Looking for the channel with max numbers of arcs
			int32 max_main_q_narcs = 0;
			for(LaneId ilane = 0; ilane<nlanes_usd; ++ilane)
				max_main_q_narcs = std::max(max_main_q_narcs, h_lane_counters_[ilane].main_q_narcs_and_end.x);

			// true is for IS_EMITTING
			expand_arcs_kernel<true><<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_narcs, nlanes_used),
				KALDI_CUDA_DECODER_1D_BLOCK,
				0,
				compute_st_>>>(*kernel_params_);

			// Copying the counters to host,
			// we need the aux_q_end values
			cudaMemcpyAsync(h_lanes_counters,     
					d_lanes_counters, 
					nlanes_used*sizeof(LaneParams), 
					cudaMemcpyDeviceToHost,
					compute_st_);

			// Updating a few counters, like resetting aux_q_end to 0...
			// true is for IS_EMITTING
			post_expand_kernel<true><<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
						KALDI_CUDA_DECODER_ONE_WARP_BLOCK,
						0,
						compute_st_>>>(*kernel_params_);
			cudaStreamSynchronize(compute_st_);

			// Loglikelihoods from the acoustic model
			// We are done using loglikelihoods for current frame
			// Launching kernel for next frame now if there is one
			nvtxRangePop(); // Decoding
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
			while(true) {
				int32 max_aux_q_end = 0;
				for(LaneId ilane=0;ilane < nlanes_used;++ilane) {
					const int32 aux_q_end = h_lane_params[ilane].aux_q_end;
					max_aux_q_end = std::max(max_aux_q_end, aux_q_end);
				}
				preprocess_and_contract_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_aux_q_end, nlanes_used),
								KALDI_CUDA_DECODER_1D_BLOCK,
								0,
								compute_st_>>>(*kernel_params);

				// Moving the lanes_params to host,
				// to have the main_q_narcs values
				cudaMemcpyAsync(h_lanes_counters,     
						d_lanes_counters, 
						nlanes_used*sizeof(LaneCounters), 
						cudaMemcpyDeviceToHost,
						compute_st_);
				// Waiting for the copy
				cudaStreamSynchronize(compute_st_);

				for(LaneId ilane=0; ilane<nlanes_used; ++lane_id) {
					const int32 main_q_narcs = h_lane_counters[lane_id].main_q_narcs;
					max_main_q_narcs = std::max(max_aux_q_end, main_q_narcs);
				}

				// If we have only a few arcs, jumping to the one-CTA per lane persistent version
				if(max_main_q_narcs < KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS)
					break;

				// false is for non emitting
				expand_arcs_kernel<false><<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_narcs, nlanes_used),
						KALDI_CUDA_DECODER_1D_BLOCK,
						0,
						compute_st_>>>(*kernel_params_);


				// Moving the lanes_params to host,
				// to have the aux_q_end values
				cudaMemcpyAsync(h_lanes_counters,     
						d_lanes_counters, 
						n_lanes_*sizeof(LaneCounters), 
						cudaMemcpyDeviceToHost,
						compute_st_);

				// false is for non emitting
				post_expand_kernel<false><<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
							KALDI_CUDA_DECODER_ONE_WARP_BLOCK,
							0,
							compute_st_>>>(*kernel_params);

				// Waiting for the copy
				cudaStreamSynchronize(compute_st_);
			}

			// Finalizing process non emitting. Takes care of the long tail, 
			// the final iterations with a small numbers of arcs. Do the work inside a single CTA (per lane),
			// using local __syncthreads() 
			finalize_process_non_emitting<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
							KALDI_CUDA_DECODER_LARGEST_1D_BLOCK,
							0,
							compute_st_>>>(*kernel_params);

			// Moving back to host the final (for this frame) values of :
			// - main_q_end
			// - main_q_narcs
			cudaMemcpyAsync(h_lanes_counters,     
					d_lanes_counters, 
					n_lanes_*sizeof(LaneCounters), 
					cudaMemcpyDeviceToHost,
					compute_st_);

			// Waiting for the copy
			cudaStreamSynchronize(compute_st_);

			int32 max_main_q_end = 0;
			for(LaneId ilane=0; ilane<nlanes_used; ++lane_id) {
				max_main_q_end = std::max(max_main_q_end,
						h_lane_counters_[ilane].main_q_narcs);
			}

			// PreprocessInPlace for next ProcessEmitting
			// We do it here (and not at the beginning of the loop) to 
			// return the lane back to its original state after this frame computation
			// (preprocess in place is the last one to use the state_best_cost lookup)
			// TODO rename
			preprocess_in_place_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_end, nlanes_used),
						KALDI_CUDA_DECODER_1D_BLOCK,
						0,
						compute_st_>>>(*kernel_params);

			exclusive_sum_batched_step2_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCK(1, nlanes_used),
							KALDI_CUDA_DECODER_ONE_WARP_BLOCK,
							0,
							compute_st_>>>(*kernel_params);

			exclusive_sum_batched_step3_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCK(1, nlanes_used),
							KALDI_CUDA_DECODER_ONE_WARP_BLOCK,
							0,
							compute_st_>>>(*kernel_params);

			for(LaneId ilane=0; ilane<nlanes_used; ++lane_id) {
				const ChannelId ichannel = h_kernel_params->channel_to_compute[ilane];
				const int32 main_q_end = h_lane_counters[ilane].main_q_narcs_and_end.y;
				h_all_tokens_info_[ichannel].CopyFromDevice(h_lane_params[ilane].d_main_q_info, main_q_end);
				num_frames_decoded_[ichannel]++; 
			}

			// We cannot write to the lanes.d_main_q_info 
			// until the copy is done
			cudaEventRecord(can_write_to_main_q_, copy_st_);

			CheckOverflow();
			KALDI_DECODER_CUDA_CHECK_ERROR();
		}   

		// Context switch : saving channels states
		save_channels_state_from_lanes_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
						KALDI_CUDA_DECODER_ONE_WARP_BLOCK,
						0,
						compute_st_>>>(*kernel_params);
		SaveChannelsStatesFromLanesCPU();

		nvtxRangePop();
	}


	void CudaDecoder::ComputeLogLikelihoods(DecodableInterface *decodable) {
		/* TODO
		int32 frame = num_frames_decoded_;

		decodable->ComputeLogLikelihoods(d_loglikelihoods_,frame,fst_.max_ilabel_+1, compute_st_);
		*/
	}

	void CudaDecoder::CheckOverflow() {
		for(const LaneId ilane=0; ilane<h_kernel_params_->nlanes_used; ++ilane) {
			bool q_overflow = h_lane_counters_[ilane].q_overflow;
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

		int32 float_inf_as_int = 2139095039; // FIXME use real infinite_cost_

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

	void CudaDecoder::SetChannelsInKernelParams(const std::vector<ChannelId> &channels) {
		KALDI_ASSERT(channels.size() < n_lanes_);
		for(LaneId lane_id=0; lane_id<channels.size(); ++lane_id)
			h_kernel_params_->channel_to_compute[lane_id] = channels[lane_id];
		h_kernel_params_->nchannels = channels.size();
	}
} // end namespace kaldi.
