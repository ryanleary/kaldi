// decoder/cuda-decoder-utils.cu
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


#include "decoder/cuda-decoder-utils.h"
#include <nvToolsExt.h>

namespace kaldi {

	/***************************************CudaFst Implementation*****************************************/

	void CudaFst::Initialize(const fst::Fst<StdArc> &fst) {
		nvtxRangePushA("CudaFst constructor");
		//count states since Fst doesn't provide this functionality
		num_states_=0;
		for( fst::StateIterator<fst::Fst<StdArc> > iter(fst); !iter.Done(); iter.Next()) {
			num_states_++;
		}
		start_=fst.Start();
		cudaMallocHost(&h_final_,sizeof(*h_final_)*num_states_);

		//allocate and initialize offset arrays
		cudaMallocHost(&h_e_offsets_, (num_states_+1)*sizeof(*h_e_offsets_));
		cudaMallocHost(&h_ne_offsets_, (num_states_+1)*sizeof(*h_ne_offsets_));

		cudaMalloc((void**)&d_e_offsets_,(num_states_+1)*sizeof(*d_e_offsets_));
		cudaMalloc((void**)&d_ne_offsets_,(num_states_+1)*sizeof(*d_ne_offsets_));
		cudaMalloc((void**)&d_final_,(num_states_)*sizeof(*d_final_));

		//iterate through states and arcs and count number of arcs per state
		e_count_=0;
		ne_count_=0;
		max_ilabel_=0;

		// Init first offsets
		h_ne_offsets_[0] = 0; 
		h_e_offsets_[0] = 0; 

		for(int i=0;i<num_states_;i++) {
			h_final_[i]=fst.Final(i).Value();

			//count emiting and non_emitting arcs
			for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done(); aiter.Next()) {
				StdArc arc = aiter.Value();
				int32 ilabel = arc.ilabel;

				if(ilabel>max_ilabel_) {
					max_ilabel_ = ilabel;
				}

				if(ilabel!=0) { //emitting
					e_count_++;
				} else { //non-emitting
					ne_count_++;
				}
			}
			h_ne_offsets_[i+1] = ne_count_;
			h_e_offsets_[i+1] = e_count_;
		}

		// We put the emitting arcs before the nonemitting arcs in the arc list
		// adding offset to the non emitting arcs
		// we go to num_states_+1 to take into account the last offset
		for(int i=0;i<num_states_+1;i++) 
			h_ne_offsets_[i]+=e_count_;   //e_arcs before

		arc_count_=e_count_+ne_count_;

		cudaMemcpy(d_e_offsets_,h_e_offsets_,(num_states_+1)*sizeof(*d_e_offsets_),cudaMemcpyHostToDevice);
		cudaMemcpy(d_ne_offsets_,h_ne_offsets_,(num_states_+1)*sizeof(*d_ne_offsets_),cudaMemcpyHostToDevice);
		cudaMemcpy(d_final_,h_final_,num_states_*sizeof(*d_final_),cudaMemcpyHostToDevice);

		cudaMallocHost(&h_arc_weights_,arc_count_*sizeof(*h_arc_weights_));
		cudaMallocHost(&h_arc_nextstates_,arc_count_*sizeof(*h_arc_nextstates_));
		cudaMallocHost(&h_arc_ilabels_,arc_count_*sizeof(*h_arc_ilabels_));
		cudaMallocHost(&h_arc_olabels_,arc_count_*sizeof(*h_arc_olabels_));

		cudaMalloc(&d_arc_weights_,arc_count_*sizeof(*d_arc_weights_));
		cudaMalloc(&d_arc_nextstates_,arc_count_*sizeof(*d_arc_nextstates_));

		// Only the ilabels for the e_arc are needed on the device
		cudaMalloc(&d_arc_ilabels_,e_count_*sizeof(*d_arc_ilabels_)); 
		// We do not need the olabels on the device - GetBestPath is on CPU

		//now populate arc data
		int e_idx=0;
		int ne_idx=e_count_; //starts where e_offsets_ ends

		for(int i=0;i<num_states_;i++) {
			for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done(); aiter.Next()) {
				StdArc arc = aiter.Value();
				int idx;
				if(arc.ilabel!=0) { //emitting
					idx=e_idx++;
				} else {
					idx=ne_idx++;
				}
				h_arc_weights_[idx]=arc.weight.Value();
				h_arc_nextstates_[idx]=arc.nextstate;
				h_arc_ilabels_[idx]=arc.ilabel;
				h_arc_olabels_[idx]=arc.olabel;
			}
		}

		cudaMemcpy(d_arc_weights_,h_arc_weights_,arc_count_*sizeof(*d_arc_weights_),cudaMemcpyHostToDevice);
		cudaMemcpy(d_arc_nextstates_,h_arc_nextstates_,arc_count_*sizeof(*d_arc_nextstates_),cudaMemcpyHostToDevice);
		cudaMemcpy(d_arc_ilabels_,h_arc_ilabels_, e_count_*sizeof(*d_arc_ilabels_),cudaMemcpyHostToDevice);

		// Making sure the graph is ready
		cudaDeviceSynchronize();
		KALDI_DECODER_CUDA_CHECK_ERROR();
		nvtxRangePop();
	}

	void CudaFst::Finalize() {
		nvtxRangePushA("CudaFst destructor");
		cudaFreeHost(h_final_);
		cudaFreeHost(h_e_offsets_);
		cudaFreeHost(h_ne_offsets_);

		cudaFree(d_e_offsets_);
		cudaFree(d_ne_offsets_);
		cudaFree(d_final_);

		cudaFreeHost(h_arc_weights_);
		cudaFreeHost(h_arc_nextstates_);
		cudaFreeHost(h_arc_ilabels_);
		cudaFreeHost(h_arc_olabels_);

		cudaFree(d_arc_weights_);
		cudaFree(d_arc_nextstates_);
		cudaFree(d_arc_ilabels_);
		nvtxRangePop();
	}


	/***************************************End CudaFst****************************************************/


	// Constructor always takes an initial capacity for the vector
	// even if the vector can grow if necessary, it damages performance
	// we need to have an appropriate initial capacity (is set using a parameter in CudaDecoderConfig)
	InfoTokenVector::InfoTokenVector(int32 capacity, cudaStream_t copy_st) : capacity_(capacity), copy_st_(copy_st) {
		KALDI_LOG << "Allocating InfoTokenVector with capacity = " << capacity_ << " tokens";
		cudaMallocHost(&h_data_, capacity_ * sizeof(*h_data_)); 
		Reset();
	}

        InfoTokenVector::InfoTokenVector(const InfoTokenVector &other) : InfoTokenVector(other.capacity_, other.copy_st_) {}

	void InfoTokenVector::Reset() {
		size_ = 0;
	};

	void InfoTokenVector::CopyFromDevice(InfoToken *d_ptr, int32 count) { // TODO add the Append keyword 
		Reserve(size_+count); // making sure we have the space

		cudaMemcpyAsync(&h_data_[size_], d_ptr, count*sizeof(*h_data_), cudaMemcpyDeviceToHost, copy_st_);
		size_ += count;
	}

	void InfoTokenVector::Clone(const InfoTokenVector &other) {
		Reserve(other.Size());
		size_ = other.Size();
		if(size_ == 0)
			return;
		const InfoToken *h_data_other = other.GetRawPointer();
		cudaMemcpyAsync(h_data_, h_data_other, size_ * sizeof(*h_data_), cudaMemcpyHostToHost, copy_st_);
		cudaStreamSynchronize(copy_st_); // after host2host?
	};

	void InfoTokenVector::Reserve(int32 min_capacity) {
		if(min_capacity <= capacity_)
			return;

		while(capacity_ < min_capacity)
			capacity_ *= 2;

		KALDI_LOG << "Reallocating InfoTokenVector on host (new capacity = " << capacity_ << " tokens).";

		cudaStreamSynchronize(copy_st_);
		InfoToken *h_old_data = h_data_;
		cudaMallocHost(&h_data_, capacity_ * sizeof(*h_data_)); 

		if(!h_data_)
			KALDI_ERR << "Host ran out of memory to store tokens. Exiting.";

		if(size_ > 0)
			cudaMemcpyAsync(h_data_, h_old_data, size_ * sizeof(*h_data_), cudaMemcpyHostToHost, copy_st_);

		cudaStreamSynchronize(copy_st_);
		cudaFreeHost(h_old_data);
	}

	InfoToken * InfoTokenVector::GetRawPointer() const {
		return h_data_;
	}

	InfoTokenVector::~InfoTokenVector() {
		cudaFreeHost(h_data_);
	}

} // end namespace kaldi
