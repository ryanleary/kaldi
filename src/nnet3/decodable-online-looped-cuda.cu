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

__global__ void compute_loglikelihoogs_kernel_( BaseFloat *out, int32 count, BaseFloat *data, int32 *trans) {
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
  EnsureFrameIsComputed(subsampled_frame);
  cudaStreamSynchronize(cudaStreamPerThread);      
  int threads=128;
  int blocks=(count+threads-1)/threads;
  compute_loglikelihoogs_kernel_<<<blocks,threads,0,stream>>>(out,count,current_log_post_.Data()+(subsampled_frame-current_log_post_subsampled_offset_)*current_log_post_.Stride(),trans_d_);
}

} // namespace nnet3
} // namespace kaldi
