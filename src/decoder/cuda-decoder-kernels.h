#ifndef KALDI_DECODER_CUDA_DECODER_KERNELS_H_
#define KALDI_DECODER_CUDA_DECODER_KERNELS_H_
#include "cuda-decoder.h"
namespace kaldi {
	typedef CudaDecoder::CostType CostType;
	typedef CudaDecoder::StateId StateId;

	__global__ void init_state_best_cost_lookup_kernel(KernelParams params);
	__global__ void get_best_cost_kernel(KernelParams params, bool isfinal, CostType fst_zero);
	__global__ void finalize_process_non_emitting_kernel(KernelParams params);
	__global__ void exclusive_sum_batched_step3_kernel(KernelParams params);
	__global__ void exclusive_sum_batched_step2_kernel(KernelParams params);
	__global__ void save_channels_state_from_lanes_kernel(KernelParams params);
	__global__ void load_channels_state_in_lanes_kernel(KernelParams params);
	__global__ void init_decoding_on_device_kernel(KernelParams params);
	__global__ void initialize_initial_lane_kernel(KernelParams params, StateId init_state, CostType init_cost);
	template<bool IS_EMITTING>
		__global__ void expand_arcs_kernel(KernelParams params);
	template<bool IS_EMITTING>
		__global__ void post_expand_kernel(KernelParams params);
	__global__ void preprocess_in_place_kernel(KernelParams params);
	__global__ void preprocess_and_contract_kernel(KernelParams params);

	struct LaneCounters {
		// Contains both main_q_end and narcs
		// End index of the main queue
		// only tokens at index i with i < main_q_end 
		// are valid tokens
		// Each valid token the subqueue main_q[main_q_offset, main_q_end[ has 
		// a number of outgoing arcs (out-degree)
		// main_q_narcs is the sum of those numbers 
		//
		// We sometime need to update both end and narcs at the same time,
		// which is why they're packed together
		int2 main_q_narcs_and_end;

		// Some kernels need to perform some operations before exiting
		// n_CTA_done is a counter that we increment when a CTA (CUDA blocks)
		// is done
		// Each CTA then tests the value for n_CTA_done to detect if it's the last to exit
		// If that's the cast, it does what it has to do, and sets n_CTA_done back to 0
		int32 aux_q_end;
		int32 post_expand_aux_q_end; // used for double buffering

		// Depending on the value of the parameter "max_tokens_per_frame"
		// we can end up with an overflow when generating the tokens for a frame
		// We try to prevent this from happening using an adaptive beam
		// if an overflow is about to happen, the kernels revert all data
		// to the last valid state, and set that flag to true
		// Even if that flag is set, we can continue the execution (quality
		// of the output can be lowered)
		// We use that flag to display a warning to stderr
		int32 q_overflow;

		// ExpandArcs does not use at its input the complete main queue
		// It only reads from the index range [main_q_local_offset, end[
		int32 main_q_local_offset;
		int32 main_q_global_offset;            

		IntegerCostType min_int_cost;
		IntegerCostType int_beam;
		IntegerCostType int_cutoff; // min_cost + beam (if min_cost < INF, otherwise INF)

		// Only valid after calling GetBestCost
		// different than min_int_cost : we include the "final" cost
		int2 min_int_cost_and_arg_with_final;
		int32 reached_final;
	};

	// 
	// Parameters used by a decoder channel
	// Their job is to save the state of the decoding 
	// channel between frames
	//
	struct ChannelCounters {
		// Cutoff for the current frame
		// Contains both the global min cost (min cost for that frame)
		// And the current beam
		// We use an adaptive beam, so the beam might change during computation
		CostType prev_beam;

		// main_q_end and main_q_narcs at the end of the previous frame
		int2 prev_main_q_narcs_and_end;

		// The token at index i in the main queue has in reality 
		// a global index of (i + main_q_global_offset)
		// This global index is unique and takes into account that 
		// we've flushed the main_q back to the host. We need unique indexes 
		// for each token in order to have valid token.prev_token data members
		// and be able to backtrack at the end
		int32 prev_main_q_global_offset;            
	};


	template<typename T> 
		struct LaneMatrixInterface  {
			T *data_;	
			int32 ld_;	 // leading dimension - may use a log2 at some point
			__host__ __device__ T *lane(const int32 ilane) {
				return &data_[ilane*ld_];
			}
		};

	template<typename T> 
		struct ChannelMatrixInterface {
			T *data_;	
			int32 ld_;	 // leading dimension
			__host__ __device__ T *channel(const int32 ichannel) {
				return &data_[ichannel*ld_];
			}
		};

	template<typename T>
		// if necessary, make a version that always use ld_ as the next power of 2
		class DeviceMatrix {
			T *data_;	
			void Allocate() {
				KALDI_ASSERT(nrows_ > 0);
				KALDI_ASSERT(ld_ > 0);
				KALDI_ASSERT(!data_);
				cudaMalloc(&data_, nrows_*ld_*sizeof(*data_));
			}
			void Free() {
				KALDI_ASSERT(data_);
				cudaFree(data_);
			}
			protected:
			int32 ld_;	 // leading dimension
			int32 nrows_;	 // leading dimension
			public:
			DeviceMatrix() : data_(NULL), ld_(0), nrows_(0) {}

			virtual ~DeviceMatrix() {
				if(data_)
					Free();
			}

			void Resize(int32 nrows, int32 ld) {
				KALDI_ASSERT(nrows > 0);
				KALDI_ASSERT(ld > 0);
				nrows_ = nrows;
				ld_ = ld;
			}

			T *MutableData() {
				if(!data_)
					Allocate();
				return data_;
			}
			// abstract getInterface... 
		};

	public:
	template<typename T>
		class DeviceLaneMatrix : public DeviceMatrix<T>  {
			public:
				LaneMatrixInterface<T> GetInterface() {	
					return {this->MutableData(), this->ld_};
				}
		};

	template<typename T>
		class DeviceChannelMatrix : public DeviceMatrix<T> {
			public:
				ChannelMatrixInterface<T> GetInterface() {	
					return {this->MutableData(), this->ld_};
				}
		};


	struct KernelParams {
		// In AdvanceDecoding,
		// the lane lane_id will compute the channel
		// with channel_id = channel_to_compute[lane_id]
		ChannelId channel_to_compute[KALDI_CUDA_DECODER_MAX_N_LANES];
		int32 nlanes_used;
		int32 max_nlanes;

		ChannelMatrixInterface<ChannelCounters> d_channels_counters; 
		LaneMatrixInterface<LaneCounters> d_lanes_counters; 

		ChannelMatrixInterface<int2> d_main_q_state_and_cost; 
		LaneMatrixInterface<InfoToken> d_main_q_info; 

		LaneMatrixInterface<int2> d_aux_q_state_and_cost; // TODO int_cost
		LaneMatrixInterface<InfoToken> d_aux_q_info; 
		ChannelMatrixInterface<int32> d_main_q_degrees_prefix_sum; 
		LaneMatrixInterface<int32> d_main_q_degrees_block_sums_prefix_sum; 
		ChannelMatrixInterface<int32> d_main_q_arc_offsets; 
		ChannelMatrixInterface<CostType> d_loglikelihoods;
		LaneMatrixInterface<IntegerCostType> d_state_best_int_cost; 

		// TODO use the CudaFst struct
		int32 q_capacity;
		CostType *d_arc_weights;
		int32 *d_arc_nextstates;
		int32 *d_arc_ilabels;
		uint32 *d_arc_e_offsets;
		uint32 *d_arc_ne_offsets;
		CostType *d_fst_final_costs;
		int32 nstates;
		CostType default_beam;
		int32 init_channel_id;
	};

} // namespace kaldi
#endif
