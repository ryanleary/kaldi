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
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

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
            void initialize(const fst::Fst<StdArc> &fst);
            void finalize();

            inline uint32_t NumStates() const {  return numStates; }
            inline StateId Start() const { return start; }    
            size_t getCudaMallocBytes() const { return bytes_cudaMalloc; }
        private:
            friend class CudaDecoder;

            unsigned int numStates;               //total number of states
            StateId  start;

            unsigned int max_ilabel;              //the largest ilabel
            unsigned int e_count, ne_count, arc_count;       //number of emitting and non-emitting states

            //This data structure is similar to a CSR matrix format 
            //where I have 2 matrices (one emitting one non-emitting).

            //Offset arrays are numStates+1 in size. 
            //Arc values for state i are stored in the range of [i,i+1)
            //size numStates+1
            unsigned int *h_e_offsets,*d_e_offsets;               //Emitting offset arrays 
            unsigned int *h_ne_offsets, *d_ne_offsets;            //Non-emitting offset arrays

            //These are the values for each arc. Arcs belonging to state i are found in the range of [offsets[i], offsets[i+1]) 
            //non-zeros (Size arc_count+1)
            BaseFloat *h_arc_weights, *d_arc_weights;
            StateId *h_arc_nextstates, *d_arc_nextstates;
            int32 *h_arc_ilabels, *d_arc_ilabels;
            int32 *h_arc_olabels;

            //final costs
            float *h_final;
            //allocation size
            size_t bytes_cudaMalloc;
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
        size_t capacity_, size_;
        // Stream used for the async copies device->host
        cudaStream_t copy_st_;
        InfoToken *h_data_;

        public:
        InfoTokenVector(int initial_capacity, cudaStream_t st);
        void Reset();
        void SetCudaStream(cudaStream_t st);
        void CopyFromDevice(size_t offset, InfoToken *d_ptr, size_t count);    
        void Reserve(size_t min_capacity);    
        InfoToken *GetRawPointer() const;
        virtual ~InfoTokenVector();
    };

} // end namespace kaldi

#endif
