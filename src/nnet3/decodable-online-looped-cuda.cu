// nnet3/decodable-online-looped.cc

// Copyright  2017  Johns Hopkins University (author: Daniel Povey)

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

#include <nnet3/decodable-online-looped.h>
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

#if 0
//call cuda kernel...
computeLogLikelihoodsKernel<<<blocks,threads>>>(out,count,current_log_post_.Data()+(subsampled_frame-current_log_post_subsampled_offset_)*current_log_post_.Stride());
//Copy trans_model_ to device...
//current_log_post_.Data();
//current_log_post_.Stride();
//simple kernel
//out[i]=data[row,trans_model_.TransitionIdToPdf(i)];

#endif

__global__ void computeLogLikelihoodsKernel( BaseFloat *out, int32 count, BaseFloat *data, int32 *trans) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
    out[i]=data[trans[i]];
  }
}
DecodableAmNnetLoopedOnlineCuda::DecodableAmNnetLoopedOnlineCuda(
      const TransitionModel &trans_model,
      const DecodableNnetSimpleLoopedInfo &info,
      OnlineFeatureInterface *input_features,
      OnlineFeatureInterface *ivector_features):
      DecodableNnetLoopedOnlineBase(info, input_features, ivector_features),
      trans_model_(trans_model) {
        
      int size=trans_model_.id2pdf_id_.size()*sizeof(int32_t);
      cudaMalloc(&trans_d_,size);
      cudaMemcpy(trans_d_,&trans_model_.id2pdf_id_[0],size,cudaMemcpyHostToDevice);
 };

DecodableAmNnetLoopedOnlineCuda::~DecodableAmNnetLoopedOnlineCuda() {
    cudaFree(trans_d_);
}

void DecodableAmNnetLoopedOnlineCuda::ComputeLogLikelihoods(BaseFloat *out, int32 subsampled_frame, int32 count, void *stream) {
//critical section to avoid illegal access errrors in ensure frame is computed.  Not sure what the cause is.  We should root cause and fix properly.  TODO
#pragma omp critical 
  {
  EnsureFrameIsComputed(subsampled_frame);
  cudaStreamSynchronize(cudaStreamPerThread);      
  }
  int threads=128;
  int blocks=(count+threads-1)/threads;
  computeLogLikelihoodsKernel<<<blocks,threads,0,stream>>>(out,count,current_log_post_.Data()+(subsampled_frame-current_log_post_subsampled_offset_)*current_log_post_.Stride(),trans_d_);
#if 0
  for(int i=0;i<count;i++) {
    BaseFloat val = current_log_post_(
      subsampled_frame - current_log_post_subsampled_offset_,
      trans_model_.TransitionIdToPdf(i));
    out[i]=val;
  }
#endif
}

} // namespace nnet3
} // namespace kaldi
