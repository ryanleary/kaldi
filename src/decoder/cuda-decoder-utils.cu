// decoder/cuda-decoder-utils.cu

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


#include "decoder/cuda-decoder-utils.h"
#include <nvToolsExt.h>

namespace kaldi {

    /***************************************CudaFst Implementation*****************************************/

    void CudaFst::initialize(const fst::Fst<StdArc> &fst) {
        nvtxRangePushA("CudaFst constructor");

        //count states since Fst doesn't provide this functionality
        numStates=0;
        for( fst::StateIterator<fst::Fst<StdArc> > iter(fst); !iter.Done(); iter.Next()) {
            numStates++;
        }
        start=fst.Start();
        cudaMallocHost(&h_final,sizeof(float)*numStates);

        //allocate and initialize offset arrays
        cudaMallocHost(&h_e_offsets, (numStates+1)*sizeof(unsigned int));
        cudaMallocHost(&h_ne_offsets, (numStates+1)*sizeof(unsigned int));

        cudaMalloc((void**)&d_e_offsets,sizeof(unsigned int)*(numStates+1));
        cudaMalloc((void**)&d_ne_offsets,sizeof(unsigned int)*(numStates+1));
 
       //iterate through states and arcs and count number of arcs per state
        e_count=0;
        ne_count=0;
        max_ilabel=0;
       
        // Init first offsets
        h_ne_offsets[0] = 0; 
        h_e_offsets[0] = 0; 

        for(int i=0;i<numStates;i++) {
            h_final[i]=fst.Final(i).Value();

            //count emiting and non_emitting arcs
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
            h_ne_offsets[i+1]=ne_count;
            h_e_offsets[i+1]=e_count;
        }

        // We put the emitting arcs before the nonemitting arcs in the arc list
        // adding offset to the non emitting arcs
        // we go to numStates+1 to take into account the last offset
        for(int i=0;i<numStates+1;i++) 
            h_ne_offsets[i]+=e_count;   //e_arcs before

        arc_count=e_count+ne_count;

        cudaMemcpy(d_e_offsets,h_e_offsets,sizeof(unsigned int)*(numStates+1),cudaMemcpyHostToDevice);
        cudaMemcpy(d_ne_offsets,h_ne_offsets,sizeof(unsigned int)*(numStates+1),cudaMemcpyHostToDevice);

        cudaMallocHost(&h_arc_weights,arc_count*sizeof(BaseFloat));
        cudaMallocHost(&h_arc_nextstates,arc_count*sizeof(StateId));
        cudaMallocHost(&h_arc_ilabels,arc_count*sizeof(int32));
        cudaMallocHost(&h_arc_olabels,arc_count*sizeof(int32));

        cudaMalloc(&d_arc_weights,arc_count*sizeof(BaseFloat));
        cudaMalloc(&d_arc_nextstates,arc_count*sizeof(StateId));

        // Only the ilabels for the e_arc are needed on the device
        cudaMalloc(&d_arc_ilabels,e_count*sizeof(int32)); 
        // We do not need the olabels on the device - GetBestPath is on CPU

        //now populate arc data
        int e_idx=0;
        int ne_idx=e_count; //starts where e_offsets ends

        for(int i=0;i<numStates;i++) {
            for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done(); aiter.Next()) {
                StdArc arc = aiter.Value();
                int idx;
                if(arc.ilabel!=0) { //emitting
                    idx=e_idx++;
                } else {
                    idx=ne_idx++;
                }
                h_arc_weights[idx]=arc.weight.Value();
                h_arc_nextstates[idx]=arc.nextstate;
                h_arc_ilabels[idx]=arc.ilabel;
                h_arc_olabels[idx]=arc.olabel;
            }
        }

        cudaMemcpy(d_arc_weights,h_arc_weights,arc_count*sizeof(BaseFloat),cudaMemcpyHostToDevice);
        cudaMemcpy(d_arc_nextstates,h_arc_nextstates,arc_count*sizeof(StateId),cudaMemcpyHostToDevice);
        cudaMemcpy(d_arc_ilabels,h_arc_ilabels, e_count*sizeof(int32),cudaMemcpyHostToDevice);
        
        // Making sure the graph is ready
        cudaDeviceSynchronize();

        cudaCheckError();

        nvtxRangePop();
    }

    void CudaFst::finalize() {
        nvtxRangePushA("CudaFst destructor");
        cudaFreeHost(h_final);
        cudaFreeHost(h_e_offsets);
        cudaFreeHost(h_ne_offsets);

        cudaFree(d_e_offsets);
        cudaFree(d_ne_offsets);

        cudaFreeHost(h_arc_weights);
        cudaFreeHost(h_arc_nextstates);
        cudaFreeHost(h_arc_ilabels);
        cudaFreeHost(h_arc_olabels);

        cudaFree(d_arc_weights);
        cudaFree(d_arc_nextstates);
        cudaFree(d_arc_ilabels);
        nvtxRangePop();
    }


    /***************************************End CudaFst****************************************************/


    // Constructor always takes an initial capacity for the vector
    // even if the vector can grow if necessary, it damages performance
    // we need to have an appropriate initial capacity (is set using a parameter in CudaDecoderConfig)
    InfoTokenVector::InfoTokenVector(int capacity) {
        capacity_ = capacity;
        KALDI_LOG << "Allocating InfoTokenVector with capacity = " << capacity_ << " tokens";
        cudaMallocHost(&h_data_, capacity_ * sizeof(InfoToken)); 
        SetCudaStream(0); // using default stream
        Reset();
    }

    void InfoTokenVector::Reset() {
        size_ = 0;
    };

    void InfoTokenVector::SetCudaStream(cudaStream_t st) {
        copy_st_ = st;
    }

    void InfoTokenVector::CopyFromDevice(size_t offset, InfoToken *d_ptr, size_t count) {
        Reserve(size_+count); // making sure we have the space

        cudaMemcpyAsync(&h_data_[offset], d_ptr, count*sizeof(InfoToken), cudaMemcpyDeviceToHost, copy_st_);
        size_ += count;
    }

    void InfoTokenVector::Reserve(size_t min_capacity) {
        if(min_capacity <= capacity_)
            return;

        while(capacity_ < min_capacity)
            capacity_ *= 2;

        KALDI_LOG << "Reallocating InfoTokenVector on host (new capacity = " << capacity_ << " tokens).";

        cudaStreamSynchronize(copy_st_);
        InfoToken *h_old_data = h_data_;
        cudaMallocHost(&h_data_, capacity_ * sizeof(InfoToken)); 

        if(!h_data_)
            KALDI_ERR << "Host ran out of memory to store tokens. Exiting.";

        if(size_ > 0)
            cudaMemcpyAsync(h_data_, h_old_data, size_ * sizeof(InfoToken), cudaMemcpyHostToHost, copy_st_);

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
