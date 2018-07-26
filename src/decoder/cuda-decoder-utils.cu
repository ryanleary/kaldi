// decoder/cuda-decoder-utils.cu

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


#include "decoder/cuda-decoder-utils.h"
#include <nvToolsExt.h>

namespace kaldi {

    /***************************************CudaFst Implementation*****************************************/

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


    TokenVector::TokenVector() {
        capacity = 16; // Not important, we're going to call Reserve anyway

        cudaMallocHost(&h_data, capacity * sizeof(InfoToken)); 
        SetCudaStream(0);

        Reset();
    }

    void TokenVector::Reset() {
        size = 0;
    };

    void TokenVector::SetCudaStream(cudaStream_t st) {
        copy_st = st;
    }

    void TokenVector::CopyFromDevice(size_t offset, InfoToken *d_ptr, size_t count) {
        Reserve(size+count); // making sure we have the space

        cudaMemcpyAsync(&h_data[offset], d_ptr, count*sizeof(InfoToken), cudaMemcpyDeviceToHost, copy_st);
        size += count;
    }

    void TokenVector::Reserve(size_t min_capacity) {
        if(min_capacity <= capacity)
            return;

        while(capacity < min_capacity)
            capacity *= 2;

        KALDI_LOG << "Reallocating TokenVector on host (new capacity = " << capacity << " tokens)";

        cudaStreamSynchronize(copy_st);
        InfoToken *h_old_data = h_data;
        cudaMallocHost(&h_data, capacity * sizeof(InfoToken)); 

        if(!h_data)
            KALDI_ERR << "Host ran out of memory to store tokens. Exiting.";

        if(size)
            cudaMemcpyAsync(h_data, h_old_data, size * sizeof(InfoToken), cudaMemcpyHostToHost, copy_st);

        cudaStreamSynchronize(copy_st);
        cudaFreeHost(h_old_data);
    }

    InfoToken * TokenVector::GetRawPointer() const {
        return h_data;
    }

    TokenVector::~TokenVector() {
        cudaFreeHost(h_data);
    }

} // end namespace kaldi
