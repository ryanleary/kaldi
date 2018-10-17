// online2/online-nnet3-decoding.h

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)

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


#ifndef KALDI_ONLINE2_ONLINE_NNET3_CUDA_DECODING_H_
#define KALDI_ONLINE2_ONLINE_NNET3_CUDA_DECODING_H_

#include <string>
#include <vector>
#include <deque>

#include "nnet3/decodable-online-looped.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "itf/online-feature-itf.h"
#include "online2/online-endpoint.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "decoder/lattice-faster-online-decoder.h"
#include "decoder/cuda-decoder.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include <vector>

#define DECODER_NDUPLICATES 1 //FIXME used for testing

namespace kaldi {
/// @addtogroup  onlinedecoding OnlineDecoding
/// @{


/**
   You will instantiate this class when you want to decode a single
   utterance using the online-decoding setup for neural nets.
*/
class SingleUtteranceNnet3CudaDecoder {
 std::vector<ChannelId> channels_;  //FIXME temp
 public:

  // Constructor. The pointer 'features' is not being given to this class to own
  // and deallocate, it is owned externally.
  SingleUtteranceNnet3CudaDecoder(
                              const TransitionModel &trans_model,
                              const nnet3::DecodableNnetSimpleLoopedInfo &info,
                              CudaDecoder &cuda_decoder,
                              OnlineNnet2FeaturePipeline *features);

  /// advance the decoding as far as we can.
  void AdvanceDecoding();

  int32 NumFramesDecoded() const;

  /// Outputs an FST corresponding to the single best path through the current
  /// lattice. If "use_final_probs" is true AND we reached the final-state of
  /// the graph then it will include those as final-probs, else it will treat
  /// all final-probs as one.
  void GetBestPath(bool end_of_utterance,
                   std::vector<Lattice*> &best_paths) const;

  const CudaDecoder &Decoder() const { return decoder_; }

  ~SingleUtteranceNnet3CudaDecoder() { }
 private:

  // this is remembered from the constructor; it's ultimately
  // derived from calling FrameShiftInSeconds() on the feature pipeline.
  BaseFloat input_feature_frame_shift_in_seconds_;

  // we need to keep a reference to the transition model around only because
  // it's needed by the endpointing code.
  const TransitionModel &trans_model_;

  nnet3::DecodableAmNnetLoopedOnlineCuda decodable_;

  CudaDecoder &decoder_;
};


/// @} End of "addtogroup onlinedecoding"

}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_NNET3_CUDA_DECODING_H_
