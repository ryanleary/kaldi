// online2/online-nnet3-decoding.cc

// Copyright    2013-2014  Johns Hopkins University (author: Daniel Povey)
//              2016  Api.ai (Author: Ilya Platonov)

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

#include "online2/online-nnet3-cuda-decoding.h"
#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"

#include <numeric>

namespace kaldi {

SingleUtteranceNnet3CudaDecoder::SingleUtteranceNnet3CudaDecoder(
    const TransitionModel &trans_model,
    const nnet3::DecodableNnetSimpleLoopedInfo &info,
    CudaDecoder &cuda_decoder,
    OnlineNnet2FeaturePipeline *features) :
    input_feature_frame_shift_in_seconds_(features->FrameShiftInSeconds()),
    trans_model_(trans_model),
    decodable_(trans_model_, info,
               features->InputFeature(), features->IvectorFeature()),
    decoder_(cuda_decoder) {
    KALDI_ASSERT(DECODER_NDUPLICATES == 1); // FIXME we have only one decodable to use
    const int32 nchannels = DECODER_NDUPLICATES;
    channels_.resize(nchannels);
    std::iota(channels_.begin(), channels_.end(), 0); // we will compute channels 0, 1, 2...
  decoder_.InitDecoding(channels_);
}


void SingleUtteranceNnet3CudaDecoder::AdvanceDecoding() {
  std::vector<DecodableInterface*> decodables = {&decodable_};
  decoder_.AdvanceDecoding(channels_, decodables);
}

int32 SingleUtteranceNnet3CudaDecoder::NumFramesDecoded() const {
  return decoder_.NumFramesDecoded(0); // FIXME 0 hardcoded
}

void SingleUtteranceNnet3CudaDecoder::GetBestPath(bool end_of_utterance,
                                              std::vector<Lattice*> &best_paths) const {
  decoder_.GetBestPath(channels_, best_paths, end_of_utterance);
}

}  // namespace kaldi
