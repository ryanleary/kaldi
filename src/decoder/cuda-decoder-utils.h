// decoder/cuda-decoder-utils.h

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

#ifndef KALDI_DECODER_CUDA_DECODER_UTILS_H_
#define KALDI_DECODER_CUDA_DECODER_UTILS_H_

//Macro for checking cuda errors following a cuda launch or api call
#define KALDI_DECODER_CUDA_CHECK_ERROR() do {                        \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
        KALDI_ERR << "Cuda failure " << __FILE__ << ":" << __LINE__ << ": '" << cudaGetErrorString(e); \
    }                                                                 \
} while(0);

#define KALDI_CUDA_DECODER_1D_KERNEL_LOOP(i, n)                      \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
			i += blockDim.x * gridDim.x)

#define KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(offset, th_idx, n)         \
	for (int offset = blockIdx.x * blockDim.x, th_idx = threadIdx.x; offset < (n);     \
			offset += blockDim.x * gridDim.x)

#define IS_LAST_1D_THREAD() \
	(threadIdx.x == (blockDim.x-1))

#define KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(i, n)                   \
	for (int i = blockIdx.y; i < (n);  i += gridDim.y)                     



#include "util/stl-utils.h"
#include "fst/fstlib.h"

namespace kaldi {

    class CudaFst {
        public:
            typedef fst::StdArc StdArc;
            typedef StdArc::Weight StdWeight;
            typedef StdArc::Label Label;
            typedef StdArc::StateId StateId;

            CudaFst() {};
            // Creates a CSR representation of the FST,
            // then copies it to the GPU
            void Initialize(const fst::Fst<StdArc> &fst);
            void Finalize();

            inline uint32_t NumStates() const {  return num_states_; }
            inline StateId Start() const { return start_; }    
        private:
            friend class CudaDecoder;

            // Total number of states
            unsigned int num_states_; 

            // Starting state of the FST
            // Computation should start from state start_
            StateId  start_;

            // We have all ilabel <= max_ilabel_
            unsigned int max_ilabel_;              
            
            // Number of emitting, non-emitting, and total number of arcs
            unsigned int e_count_, ne_count_, arc_count_;       

            // This data structure is similar to a CSR matrix format 
            // with 2 offsets matrices (one emitting one non-emitting).

            // Offset arrays are num_states_+1 in size (last state needs 
            // its +1 arc_offset)
            // Arc values for state i are stored in the range of [offset[i],offset[i+1][
            unsigned int *h_e_offsets_,*d_e_offsets_;               //Emitting offset arrays 
            unsigned int *h_ne_offsets_, *d_ne_offsets_;            //Non-emitting offset arrays

            // These are the values for each arc. 
            // Arcs belonging to state i are found in the range of [offsets[i], offsets[i+1][
            // Use e_offsets or ne_offsets depending on what you need (emitting/nonemitting)
            // The ilabels arrays are of size e_count_, not arc_count_

            BaseFloat *h_arc_weights_, *d_arc_weights_; // TODO define CostType here
            StateId *h_arc_nextstates_, *d_arc_nextstates_;
            int32 *h_arc_ilabels_, *d_arc_ilabels_;
            int32 *h_arc_olabels_;

            // Final costs
            // final cost of state i is h_final_[i]
            float *h_final_;
    };

    // InfoToken contains data that needs to be saved for the backtrack
    // in GetBestPath
    // It will be moved back to CPU memory using a InfoTokenVector
    struct InfoToken {
        int prev_token;
        int arc_idx;
    };

    //
    // InfoTokenVector
    // Vector for InfoToken that uses CPU pinned memory
    // We use it to transfer the relevant parts of the tokens
    // back to the CPU memory
    //
    class InfoTokenVector {
        int32 capacity_, size_;
        // Stream used for the async copies device->host
        cudaStream_t copy_st_;
        InfoToken *h_data_;

        public:
        InfoTokenVector(int initial_capacity, cudaStream_t copy_st_);
        void Clone(const InfoTokenVector &other);
        void Reset();
        void CopyFromDevice(size_t offset, InfoToken *d_ptr, size_t count);    
        int32 Size() { return size_ };
        void Reserve(size_t min_capacity);    
        InfoToken *GetRawPointer() const;
        virtual ~InfoTokenVector();
    };

    template<typename T>
	    class DeviceMatrix {
		    T *data_;	
		    // TODO ideally we'd want ld_ to be templated, 
		    // and be a power of 2, to avoid having a lot of int multiplication
		    int ld_;	 // leading dimension
		    public:
		    DeviceMatrix(int nrows, int ld) : ld_(ld) {
			    cudaMalloc(&data, nrows*ld*sizeof(*data));
		    }

		    virtual ~DeviceMatrix() {
			    cudaFree(data);
		    }

		    __host__ __device__ T *row(int r) {
			    return &data[r*ld];
		    }

		    __host__ __device__ T *data() {
			    return data_;
		    }
	    };
} // end namespace kaldi

#endif
