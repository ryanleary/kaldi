// decoder/cuda-decoder.cu
// TODO nvidia apache2
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
#include "cuda-decoder-kernels.h"

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
		KALDI_ASSERT(nlanes_ < KALDI_CUDA_DECODER_MAX_N_LANES);
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
		init_channel_id_ = nchannels_-1; // Using last one as init_channel_params

		const int32 num_states = fst_.num_states_;
		d_channels_counters_.Resize(nchannels_, 1);
		d_lanes_counters_.Resize(nlanes, 1);
		d_main_q_state_and_cost_.Resize(nchannels_, max_tokens_per_frame_);
		d_main_q_info_.Resize(nlanes, max_tokens_per_frame_);
		d_aux_q_state_and_cost_.Resize(nlanes, max_tokens_per_frame_);
		d_aux_q_info_.Resize(nlanes, max_tokens_per_frame_);
		d_main_q_degrees_prefix_sum_.Resize(nchannels_, max_tokens_per_frame_);
		d_main_q_degrees_block_sums_prefix_sum_.Resize(nlanes, 
				KALDI_CUDA_DECODER_DIV_ROUND_UP(max_tokens_per_frame_, KALDI_CUDA_DECODER_1D_BLOCK) + 1);
		d_main_q_arc_offsets_.Resize(nchannels_,  max_tokens_per_frame_);
		d_state_best_int_cost_.Resize(nlanes, num_states);
		d_loglikelihoods_.Resize(nlanes, fst_.max_ilabel_+1);

		// Setting Kernel Params
		// sent to kernels by copy
		// Making sure we'll be able to send it to the kernels
		//KALDI_STATIC_ASSERT(sizeof(KernelParams) < KALDI_CUDA_DECODER_MAX_KERNEL_ARGUMENTS_BYTE_SIZE); TODO find include

		cudaMemsetAsync(d_channels_counters_.MutableData(), 0, nchannels_*sizeof(d_channels_counters_.MutableData()));
		cudaMemsetAsync(d_lanes_counters_.MutableData(), 0, nlanes_*sizeof(d_lanes_counters_.MutableData()));
		cudaMallocHost(&h_lanes_counters_, nlanes_ * sizeof(*h_lanes_counters_));
		cudaMallocHost(&h_channels_counters_, nchannels_ * sizeof(*h_channels_counters_));

		h_device_params_ = new DeviceParams();
		h_device_params_->d_channels_counters = d_channels_counters_.GetInterface();
		h_device_params_->d_lanes_counters =d_lanes_counters_.GetInterface();
		h_device_params_->d_main_q_state_and_cost = d_main_q_state_and_cost_.GetInterface();
		h_device_params_->d_main_q_info = d_main_q_info_.GetInterface();
		h_device_params_->d_aux_q_state_and_cost = d_aux_q_state_and_cost_.GetInterface();
		h_device_params_->d_aux_q_info = d_aux_q_info_.GetInterface();
		h_device_params_->d_main_q_degrees_prefix_sum = d_main_q_degrees_prefix_sum_.GetInterface();
		h_device_params_->d_main_q_degrees_block_sums_prefix_sum = d_main_q_degrees_block_sums_prefix_sum_.GetInterface();
		h_device_params_->d_main_q_arc_offsets = d_main_q_arc_offsets_.GetInterface();
		h_device_params_->d_loglikelihoods = d_loglikelihoods_.GetInterface();
		h_device_params_->d_state_best_int_cost = d_state_best_int_cost_.GetInterface();
		h_device_params_->d_arc_e_offsets = fst_.d_e_offsets_;
		h_device_params_->d_arc_ne_offsets = fst_.d_ne_offsets_;
		h_device_params_->d_arc_ilabels = fst_.d_arc_ilabels_;
		h_device_params_->d_arc_weights = fst_.d_arc_weights_;
		h_device_params_->d_arc_nextstates = fst_.d_arc_nextstates_;
		h_device_params_->d_fst_final_costs = fst_.d_final_;
		h_device_params_->default_beam = default_beam_;
		h_device_params_->q_capacity = max_tokens_per_frame_; 
		h_device_params_->init_channel_id = init_channel_id_; 
		h_device_params_->max_nlanes = nlanes_; 
		h_device_params_->nstates = fst_.num_states_; 
		h_device_params_->init_state = fst_.Start();
		KALDI_ASSERT(h_device_params_->init_state != fst::kNoStateId);
		h_device_params_->init_cost = StdWeight::One().Value();

		h_kernel_params_ = new KernelParams();

		// Initialize host tokens memory pools
		for(int ichannel=0; ichannel<nchannels_; ++ichannel)
			h_all_tokens_info_.emplace_back(max_tokens_, copy_st_);

		// Filling all best_state_cost with +INF
		init_state_best_cost_lookup_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(num_states, nlanes_),
						KALDI_CUDA_DECODER_1D_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);
	
		ComputeInitialChannel();
		--nchannels_; // removing the init_channel_params from general list

		KALDI_DECODER_CUDA_CHECK_ERROR();
		num_frames_decoded_.resize(nchannels_, 0);

		// Making sure that everything is ready to use
		cudaStreamSynchronize(compute_st_);
	}

	CudaDecoder::~CudaDecoder() {
		cudaStreamDestroy(compute_st_);
		cudaStreamDestroy(copy_st_);

		cudaEventDestroy(can_read_h_main_q_narcs_);
		cudaEventDestroy(can_write_to_main_q_);
		cudaEventDestroy(can_read_final_h_main_q_end_);
		cudaEventDestroy(before_finalize_nonemitting_kernel_);

		cudaFreeHost(h_lanes_counters_);
		cudaFreeHost(h_channels_counters_);

		// Will call the cudaFrees inside destructors 
		delete h_kernel_params_;
		delete h_device_params_;
		cudaFree(d_device_params_);

		KALDI_DECODER_CUDA_CHECK_ERROR();
	}

	void CudaDecoder::ComputeInitialChannel() {
		KALDI_ASSERT(nlanes_ > 0);
		const int32 ilane = 0;
		KALDI_ASSERT(ilane == 0);
		// Following kernels working channel_id
		h_kernel_params_->channel_to_compute[ilane] = init_channel_id_;
		h_kernel_params_->nlanes_used = 1;

		// Adding the start state to the initial token queue
		initialize_initial_lane_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, 1),
			KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
			0,
			compute_st_>>>(*h_device_params_);

		// Initial ProcessNonEmitting
		preprocess_and_contract_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, 1),
			KALDI_CUDA_DECODER_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);
		finalize_process_non_emitting_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, 1),
			KALDI_CUDA_DECODER_LARGEST_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);

		int32 main_q_end;
		cudaMemcpyAsync(&main_q_end, 
				&d_lanes_counters_.lane(ilane)->main_q_narcs_and_end.y, 
				sizeof(int32), 
				cudaMemcpyDeviceToHost, 
				compute_st_);
		cudaStreamSynchronize(compute_st_);

		KALDI_ASSERT(main_q_end > 0);

		// Preparing for first frame + reverting back to init state (lookup table, etc.)
		preprocess_in_place_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(main_q_end, 1),
			KALDI_CUDA_DECODER_1D_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);

		exclusive_sum_batched_step2_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(main_q_end, 1),
			KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);

		exclusive_sum_batched_step3_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(main_q_end, 1),
			KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);

			// Saving initial queue to host
		h_all_tokens_info_[init_channel_id_].CopyFromDevice(d_main_q_info_.lane(ilane), main_q_end);
		// Waiting for copy to be done
		cudaStreamSynchronize(copy_st_);

		// Context switch : saving channel state
		save_channels_state_from_lanes_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, 1),
						KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);
	
		// Saving init params on host
		cudaMemcpyAsync(h_lanes_counters_, 
				d_lanes_counters_.MutableData(), 
				1*sizeof(*h_lanes_counters_), 
				cudaMemcpyDeviceToHost,
				compute_st_);

		// Waiting for compute to be done 
		cudaStreamSynchronize(compute_st_);

		SaveChannelsStateFromLanesCPU();

		KALDI_DECODER_CUDA_CHECK_ERROR();
	}

	void CudaDecoder::InitDecoding() {
		std::vector<ChannelId> channels = {0};	
		InitDecoding(channels);
	}

	void CudaDecoder::InitDecoding(const std::vector<ChannelId> &channels) {
		const int nlanes_used = channels.size();
		// Getting *h_kernel_params ready to use
		SetChannelsInKernelParams(channels);
		KALDI_ASSERT(nlanes_used == h_kernel_params_->nlanes_used);

		// Size of the initial main_q_size
		const int32 init_main_q_size = h_channels_counters_[init_channel_id_].prev_main_q_narcs_and_end.y;

		KALDI_ASSERT(init_main_q_size > 0);
		// Getting the channels ready to compute new utterances
		init_decoding_on_device_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(init_main_q_size, nlanes_used),
						KALDI_CUDA_DECODER_1D_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);

		cudaStreamSynchronize(compute_st_);
		KALDI_DECODER_CUDA_CHECK_ERROR();
		for(ChannelId ichannel : channels) {
			// Tokens from initial main_q needed on host
			h_all_tokens_info_[ichannel].Clone(h_all_tokens_info_[init_channel_id_]);
			h_channels_counters_[ichannel] = h_channels_counters_[init_channel_id_];
			num_frames_decoded_[ichannel] = 0;
		}
	}

	// Context-switch : Load and Store
	void CudaDecoder::LoadChannelsStateToLanesCPU() {
		for(LaneId ilane=0; ilane<h_kernel_params_->nlanes_used; ++ilane) {
			const ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
			h_lanes_counters_[ilane].main_q_narcs_and_end = h_channels_counters_[ichannel].prev_main_q_narcs_and_end;
		}	
	}

	void CudaDecoder::SaveChannelsStateFromLanesCPU() {
		for(LaneId ilane=0; ilane<h_kernel_params_->nlanes_used; ++ilane) {
			const ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
			h_channels_counters_[ichannel].prev_main_q_narcs_and_end = h_lanes_counters_[ilane].main_q_narcs_and_end;
			h_channels_counters_[ichannel].prev_main_q_global_offset = h_lanes_counters_[ilane].main_q_global_offset;
		}
	}

	void CudaDecoder::AdvanceDecoding(DecodableInterface *decodable,
			int32 max_num_frames) {
		std::vector<ChannelId> channels = {0};	
		std::vector<DecodableInterface*> decodables = {decodable};	
		AdvanceDecoding(channels, decodables, max_num_frames);
	}

	void CudaDecoder::AdvanceDecoding(const std::vector<ChannelId> &channels,
			std::vector<DecodableInterface*> &decodables,
			int32 max_num_frames) {
		const int nlanes_used = channels.size();
		if(nlanes_used <= 0)
			return;
		
		// How many frames should we decode ?
		int32 nframes_to_decode = INT_MAX;
		for(int32 ilane=0; ilane<nlanes_used; ++ilane) {
			const ChannelId ichannel = channels[ilane];
			const int32 num_frames_decoded = num_frames_decoded_[ichannel];
			KALDI_ASSERT(num_frames_decoded >= 0 &&
					"You must call InitDecoding() before AdvanceDecoding()");
			int32 num_frames_ready = decodables[ilane]->NumFramesReady(); // FIXME plug the right one
			// num_frames_ready must be >= num_frames_decoded, or else
			// the number of frames ready must have decreased (which doesn't
			// make sense) or the decodable object changed between calls
			// (which isn't allowed).
			KALDI_ASSERT(num_frames_ready >= num_frames_decoded);
			int32 channel_nframes_to_decode = num_frames_ready - num_frames_decoded;
			nframes_to_decode = std::min(nframes_to_decode,
						channel_nframes_to_decode);
		}	
		if(max_num_frames >= 0)
			nframes_to_decode = std::min(nframes_to_decode, max_num_frames);

		// Getting *h_kernel_params ready to use
		SetChannelsInKernelParams(channels);
		KALDI_ASSERT(nlanes_used == h_kernel_params_->nlanes_used);

		// Getting the lanes ready to work with those channels  
		load_channels_state_in_lanes_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
						KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);

		LoadChannelsStateToLanesCPU();
		nvtxRangePushA("Decoding");
		for(int32 iframe=0; iframe<nframes_to_decode; ++iframe)  {
			// Computing a new frame

			// Loglikelihoods from the acoustic model
			nvtxRangePop(); // Decoding
			ComputeLogLikelihoods(decodables);
			nvtxRangePushA("Decoding");

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
			int32 max_main_q_narcs = 0; // TODO some kind of strided iterator
			for(LaneId ilane = 0; ilane<nlanes_used; ++ilane) {
				const int32 main_q_narcs = h_lanes_counters_[ilane].main_q_narcs_and_end.x;
				max_main_q_narcs = std::max(max_main_q_narcs, main_q_narcs); 
			}
			KALDI_ASSERT(max_main_q_narcs > 0);
			// true is for IS_EMITTING
			expand_arcs_kernel<true><<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_narcs, nlanes_used),
				KALDI_CUDA_DECODER_1D_BLOCK,
				0,
				compute_st_>>>(*h_device_params_,*h_kernel_params_);

			// Updating a few counters, like resetting aux_q_end to 0...
			// true is for IS_EMITTING
			post_expand_kernel<true><<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
						KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);

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
				// Moving the lanes_params to host,
				// to have the aux_q_end values
				cudaMemcpyAsync(h_lanes_counters_,     
						d_lanes_counters_.MutableData(), 
						nlanes_used*sizeof(LaneCounters), 
						cudaMemcpyDeviceToHost,
						compute_st_);
				cudaStreamSynchronize(compute_st_);
				int32 max_aux_q_end = 0;
				for(LaneId ilane=0;ilane < nlanes_used;++ilane) {
					const int32 aux_q_end = h_lanes_counters_[ilane].post_expand_aux_q_end;
					//printf("ne aux_q_end=%i, lane=%i \n", aux_q_end, ilane);
					max_aux_q_end = std::max(max_aux_q_end, aux_q_end);
				}
				if(max_aux_q_end == 0) // not likely, but possible
					break; 

				preprocess_and_contract_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_aux_q_end, nlanes_used),
								KALDI_CUDA_DECODER_1D_BLOCK,
								0,
								compute_st_>>>(*h_device_params_,*h_kernel_params_);

				// Moving the lanes_params to host,
				// to have the main_q_narcs values
				cudaMemcpyAsync(h_lanes_counters_,     
						d_lanes_counters_.MutableData(), 
						nlanes_used*sizeof(LaneCounters), 
						cudaMemcpyDeviceToHost,
						compute_st_);
				// Waiting for the copy
				cudaStreamSynchronize(compute_st_);

				max_main_q_narcs = 0;
				for(LaneId ilane=0; ilane<nlanes_used; ++ilane) {
					const int32 main_q_narcs = h_lanes_counters_[ilane].main_q_narcs_and_end.x;
					max_main_q_narcs = std::max(max_main_q_narcs, main_q_narcs);
					//printf("ne arcs=%i lane=%i \n",  main_q_narcs, ilane);
				}

				// If we have only a few arcs, jumping to the one-CTA per lane persistent version
				//printf("%i<%i ? \n", max_main_q_narcs, KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS);
				KALDI_ASSERT(KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS > 0); 
				if(max_main_q_narcs < KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS) {
					break;
				}

				// false is for non emitting
				expand_arcs_kernel<false><<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_narcs, nlanes_used),
						KALDI_CUDA_DECODER_1D_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);



				// false is for non emitting
				post_expand_kernel<false><<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
							KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
							0,
							compute_st_>>>(*h_device_params_,*h_kernel_params_);

				KALDI_DECODER_CUDA_CHECK_ERROR();
			}

			// Finalizing process non emitting. Takes care of the long tail, 
			// the final iterations with a small numbers of arcs. Do the work inside a single CTA (per lane),
			// using local __syncthreads() 
			finalize_process_non_emitting_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
							KALDI_CUDA_DECODER_LARGEST_1D_BLOCK,
							0,
							compute_st_>>>(*h_device_params_,*h_kernel_params_);

			// Moving back to host the final (for this frame) values of :
			// - main_q_end
			// - main_q_narcs
			cudaMemcpyAsync(h_lanes_counters_,     
					d_lanes_counters_.MutableData(), 
					nlanes_used*sizeof(LaneCounters), 
					cudaMemcpyDeviceToHost,
					compute_st_);

			// Waiting for the copy
			cudaStreamSynchronize(compute_st_);
			KALDI_DECODER_CUDA_CHECK_ERROR();

			int32 max_main_q_end = 0;
			for(LaneId ilane=0; ilane<nlanes_used; ++ilane) {
				const int32 main_q_end = h_lanes_counters_[ilane].main_q_narcs_and_end.y;
				KALDI_ASSERT(main_q_end > 0);
				max_main_q_end = std::max(max_main_q_end, main_q_end);
			}
			// PreprocessInPlace for next ProcessEmitting
			// We do it here (and not at the beginning of the loop) to 
			// return the lane back to its original state after this frame computation
			// (preprocess in place is the last one to use the state_best_cost lookup)
			// TODO rename
			preprocess_in_place_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_end, nlanes_used),
						KALDI_CUDA_DECODER_1D_BLOCK,
						0,
						compute_st_>>>(*h_device_params_,*h_kernel_params_);

			exclusive_sum_batched_step2_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
							KALDI_CUDA_DECODER_1D_BLOCK,
							0,
							compute_st_>>>(*h_device_params_,*h_kernel_params_);

			exclusive_sum_batched_step3_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_end, nlanes_used),
							KALDI_CUDA_DECODER_1D_BLOCK,
							0,
							compute_st_>>>(*h_device_params_,*h_kernel_params_);
			KALDI_DECODER_CUDA_CHECK_ERROR();

			for(LaneId ilane=0; ilane<nlanes_used; ++ilane) {
				const ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
				const int32 main_q_end = h_lanes_counters_[ilane].main_q_narcs_and_end.y;
				h_all_tokens_info_[ichannel].CopyFromDevice(d_main_q_info_.lane(ilane), main_q_end);
			}

			// We cannot write to the lanes.d_main_q_info 
			// until the copy is done
			cudaEventRecord(can_write_to_main_q_, copy_st_);

			CheckOverflow();
			KALDI_DECODER_CUDA_CHECK_ERROR();
			
			for(ChannelId ichannel : channels)
				++num_frames_decoded_[ichannel];

			cudaMemcpyAsync(h_lanes_counters_,     
					d_lanes_counters_.MutableData(), 
					nlanes_used*sizeof(LaneCounters), 
					cudaMemcpyDeviceToHost,
					compute_st_);

			// Waiting for the copy
			cudaStreamSynchronize(compute_st_);
		}   

		// Context switch : saving channels states
		save_channels_state_from_lanes_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(1, nlanes_used),
			KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
			0,
			compute_st_>>>(*h_device_params_,*h_kernel_params_);
		SaveChannelsStateFromLanesCPU();

		nvtxRangePop();
	}


	void CudaDecoder::ComputeLogLikelihoods(std::vector<DecodableInterface*> &decodables) {
		KALDI_ASSERT(decodables.size() == h_kernel_params_->nlanes_used);
		for(LaneId ilane=0; ilane<h_kernel_params_->nlanes_used; ++ilane) {
			ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
			int32 frame = num_frames_decoded_[ichannel];
			decodables[ilane]->ComputeLogLikelihoods(d_loglikelihoods_.lane(ilane), frame, fst_.max_ilabel_+1, compute_st_); // TODO batch
		}
	}

	void CudaDecoder::CheckOverflow() {
		for(LaneId ilane=0; ilane<h_kernel_params_->nlanes_used; ++ilane) {
			bool q_overflow = h_lanes_counters_[ilane].q_overflow;
			if(q_overflow) {
				// An overflow was prevented in a kernel
				// The algorithm can still go on but quality of the result can be reduced
				// (less tokens were generated)
				KALDI_WARN << "Preventing overflow of the frame tokens. Pursuing "
					<< "execution but the quality of the output may be decreased. "
					<< "To prevent this from happening, please increase the parameter --max-tokens-per-frame"
					<< " and/or decrease --beam";
			}
		}
	}


	// GetBestCost
	// returns the minimum cost among all tokens cost in the current frame
	// also returns the index of one token with that min cost
	//
	// Only called at the end of the computation of one audio file
	// not optimized
	void CudaDecoder::GetBestCost(const std::vector<ChannelId> &channels, bool use_final_costs, std::vector<std::pair<int32,CostType>> *argmins, std::vector<bool> *has_reached_final) {
		const int nlanes_used = channels.size();
		if(nlanes_used <= 0)
			return;
		// Getting *h_kernel_params ready to use
		SetChannelsInKernelParams(channels);
		KALDI_ASSERT(nlanes_used == h_kernel_params_->nlanes_used);
		int32 max_main_q_end = 0;
		for(ChannelId ichannel : channels)
			max_main_q_end = std::max(max_main_q_end, h_channels_counters_[ichannel].prev_main_q_narcs_and_end.y); 

		// TODO reset counters->reached_final to 0

		// We already know what's the best cost, because we needed it for the cutoff
		// it was saved in channel_counters.prev_min_cost
		// we just need to find its index	
		get_best_cost_kernel<<<KALDI_CUDA_DECODER_NUM_BLOCKS(max_main_q_end, nlanes_used),
				KALDI_CUDA_DECODER_1D_BLOCK,
				0,
				compute_st_>>>(*h_device_params_,*h_kernel_params_, use_final_costs, StdWeight::Zero().Value());
		cudaMemcpyAsync(h_lanes_counters_,     
				d_lanes_counters_.MutableData(), 
				nlanes_used*sizeof(*h_lanes_counters_), 
				cudaMemcpyDeviceToHost,
				compute_st_);

		argmins->clear();
		has_reached_final->clear();
		cudaStreamSynchronize(compute_st_);
		for(int32 ilane=0; ilane<nlanes_used; ++ilane) {
			int2 minarg = h_lanes_counters_[ilane].min_int_cost_and_arg_with_final;
			CostType min_cost = 0.0f; // FIXME intToFloat host 
			int32 arg = minarg.y;
			argmins->push_back({arg,min_cost});
			has_reached_final->push_back(h_lanes_counters_[ilane].reached_final);
		}
		cudaStreamSynchronize(compute_st_);
	}

	//
	// GetBestPath is called at the end of the computation
	// It chooses the best token from the last frame, 
	// and backtracks all the path to the beginning (StartState)
	// from there
	// It then returns that path
	//
	bool CudaDecoder::GetBestPath(Lattice* fst_out, bool use_final_probs) {
		std::vector<ChannelId> channels = {0};	
		std::vector<Lattice*> fst_out_vec = {fst_out};	
		return GetBestPath(channels, fst_out_vec, use_final_probs); 
	}
	bool CudaDecoder::GetBestPath(const std::vector<ChannelId> &channels, std::vector<Lattice*> &fst_out_vec, bool use_final_probs) {
		KALDI_ASSERT(channels.size() == fst_out_vec.size());
		KALDI_ASSERT(channels.size() <= nchannels_);

		nvtxRangePushA("GetBestPath");
		std::vector<std::pair<int32,CostType>> argmins;
		std::vector<bool> has_reached_final;
		GetBestCost(channels, use_final_probs, &argmins, &has_reached_final);
		// TODO handle if a final state was not found

		// We want the copy to host of the last tokens to be done
		// we're going to read h_all_tokens_info
		cudaEventSynchronize(can_write_to_main_q_);
		for(int32 i=0; i<channels.size(); ++i) {
			const ChannelId ichannel = channels[i];
			const int32 token_with_best_cost = argmins[i].first;
			const bool isfinal = has_reached_final[i];
			int32 token_idx = token_with_best_cost;

			// Backtracking
			// Going all the way from the token with best cost
			// to the beginning (StartState)
			std::vector<int32> reversed_path;

			// The first token was inserted at the beginning of the queue
			// it always has index 0
			// We backtrack until that first token
			while(token_idx != 0) {
				int32 arc_idx = h_all_tokens_info_[ichannel].GetRawPointer()[token_idx].arc_idx;
				reversed_path.push_back(arc_idx);
				token_idx = h_all_tokens_info_[ichannel].GetRawPointer()[token_idx].prev_token;
			}
			
			Lattice *fst_out = fst_out_vec[i];

			// Reset the fst_out
			fst_out->DeleteStates();

			// Building the output Lattice
			StateId cur_state = fst_out->AddState();
			fst_out->SetStart(cur_state);

			for (int32 i=reversed_path.size()-1; i>=1; i--) {
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


		}
		nvtxRangePop();
		return true;
	}
	void CudaDecoder::SetChannelsInKernelParams(const std::vector<ChannelId> &channels) {
		KALDI_ASSERT(channels.size() <= nchannels_);
		KALDI_ASSERT(channels.size() <= nlanes_);
		for(LaneId lane_id=0; lane_id<channels.size(); ++lane_id)
			h_kernel_params_->channel_to_compute[lane_id] = channels[lane_id];
		h_kernel_params_->nlanes_used = channels.size();
	}

	int32 CudaDecoder::NumFramesDecoded(ChannelId ichannel) const {
		KALDI_ASSERT(ichannel < nchannels_);
		return num_frames_decoded_[ichannel];	
	}
/*
	int32 CudaDecoder::NumFramesDecoded() const {
		return NumFramesDecoded(0);
	}
*/
} // end namespace kaldi.
