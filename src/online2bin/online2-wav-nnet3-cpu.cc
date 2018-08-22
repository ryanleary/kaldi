// online2bin/online2-wav-nnet3-latgen-faster.cc

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

#include "feat/wave-reader.h"
#include "online2/online-nnet3-faster-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"
#include <atomic>
#include <thread>
#include <chrono>

#define MAX_THREADS 100

using namespace kaldi;

void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
  KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                << (likelihood / num_frames) << " over " << num_frames
                << " frames.";

  if (word_syms != NULL) {
    std::cerr << utt << ' ';
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "")
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      std::cerr << s << ' ';
    }
    std::cerr << std::endl;
  }
}


struct DecodeParams {
  bool replicate;
  int num_threads;
  BaseFloat chunk_length_secs;
  bool online;
  bool write_lattice;
 
  OnlineNnet2FeaturePipelineConfig  feature_opts;
  nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
  FasterDecoderOptions decoder_opts;
  CompactLatticeWriter *clat_writer;
  fst::Fst<fst::StdArc> *decode_fst;
  
  std::string nnet3_rxfilename;
  std::string wav_rspecifier;
  std::string word_syms_rxfilename;
  std::string spk2utt_rspecifier;

  std::atomic<int> utt_idx;
  std::mutex lock;
  std::vector<double> total_time;
  std::vector<double> total_audio;
};

void decode_function(DecodeParams &params, int th_idx) {
   auto start = std::chrono::high_resolution_clock::now();
  // feature_opts includes configuration for the iVector adaptation,
  // as well as the basic features.
  KALDI_LOG << "Thread " << th_idx << " of " << params.num_threads << std::endl;
  OnlineNnet2FeaturePipelineInfo feature_info(params.feature_opts);

  if (!params.online) {
    feature_info.ivector_extractor_info.use_most_recent_ivector = true;
    feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
  }

  TransitionModel trans_model;
  nnet3::AmNnetSimple am_nnet;
  {
    bool binary;
    Input ki(params.nnet3_rxfilename, &binary);
    trans_model.Read(ki.Stream(), binary);
    am_nnet.Read(ki.Stream(), binary);
    SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
    SetDropoutTestMode(true, &(am_nnet.GetNnet()));
    nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
  }

  // this object contains precomputed stuff that is used by all decodable
  // objects.  It takes a pointer to am_nnet because if it has iVectors it has
  // to modify the nnet to accept iVectors at intervals.
  nnet3::DecodableNnetSimpleLoopedInfo decodable_info(params.decodable_opts,
      &am_nnet);

  fst::SymbolTable *word_syms = NULL;
  if (params.word_syms_rxfilename != "")
    if (!(word_syms = fst::SymbolTable::ReadText(params.word_syms_rxfilename)))
      KALDI_ERR << "Could not read symbol table from file "
        << params.word_syms_rxfilename;

  int32 num_done = 0, num_err = 0;
  double tot_like = 0.0;
  int64 num_frames = 0;

  SequentialTokenVectorReader spk2utt_reader(params.spk2utt_rspecifier);
  RandomAccessTableReader<WaveHolder> wav_reader(params.wav_rspecifier);


  OnlineTimingStats timing_stats;

  int my_idx=0;
  int last_idx=0;
  int next_idx=0;
  int num_processed=0;

  while(!spk2utt_reader.Done())
  {
    //grab next utterance

    if(params.replicate) {

      my_idx=next_idx;
      next_idx++;      //increment count by one

    } else {
      //grab unique index
      my_idx=params.utt_idx++;
    }

    //printf("THREAD: %d, IDX: %d\n", th_idx, my_idx);

    //advance reader until i am at the right utterance
    while(!spk2utt_reader.Done() && last_idx<my_idx) {
      //printf("THREAD: %d, skip\n", th_idx);
      spk2utt_reader.Next();
      last_idx++;
    }
    //no more utterances left so exit
    if(spk2utt_reader.Done()) {
      //printf("thread: %d, exit\n", th_idx);
      break;
    }
    last_idx=my_idx;
    num_processed++;

    std::string spk = spk2utt_reader.Key();
    //printf("THREAD: %d, utt: %s\n", th_idx, spk.c_str());
    const std::vector<std::string> &uttlist = spk2utt_reader.Value();
    OnlineIvectorExtractorAdaptationState adaptation_state(
        feature_info.ivector_extractor_info);
    for (size_t i = 0; i < uttlist.size(); i++) {
      std::string utt = uttlist[i];
      if (!wav_reader.HasKey(utt)) {
        KALDI_WARN << "Did not find audio for utterance " << utt;
        num_err++;
        continue;
      }
      const WaveData &wave_data = wav_reader.Value(utt);
      // get the data for channel zero (if the signal is not mono, we only
      // take the first channel).
      SubVector<BaseFloat> data(wave_data.Data(), 0);

      OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
      feature_pipeline.SetAdaptationState(adaptation_state);

      OnlineSilenceWeighting silence_weighting(
          trans_model,
          feature_info.silence_weighting_config,
          params.decodable_opts.frame_subsampling_factor);
      
      SingleUtteranceNnet3FasterDecoder decoder(params.decoder_opts, trans_model,
          decodable_info, *params.decode_fst, &feature_pipeline);


      OnlineTimer decoding_timer(utt);


      BaseFloat samp_freq = wave_data.SampFreq();
      int32 chunk_length;
      if (params.chunk_length_secs > 0) {
        chunk_length = int32(samp_freq * params.chunk_length_secs);
        if (chunk_length == 0) chunk_length = 1;
      } else {
        chunk_length = std::numeric_limits<int32>::max();
      }

      int32 samp_offset = 0;
      std::vector<std::pair<int32, BaseFloat> > delta_weights;
      
      while (samp_offset < data.Dim()) {
        int32 samp_remaining = data.Dim() - samp_offset;
        int32 num_samp = chunk_length < samp_remaining ? chunk_length
          : samp_remaining;

        SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
        feature_pipeline.AcceptWaveform(samp_freq, wave_part);

        samp_offset += num_samp;
        decoding_timer.WaitUntil(samp_offset / samp_freq);
        if (samp_offset == data.Dim()) {
          // no more input. flush out last frames
          feature_pipeline.InputFinished();
        }

        if (silence_weighting.Active() &&
            feature_pipeline.IvectorFeature() != NULL) {
          //silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
          silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
              &delta_weights);
          feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
        }
        decoder.AdvanceDecoding();
      }
      Lattice lat;
      if(num_processed>0) {
        decoder.GetBestPath(true, &lat);
      }

      decoding_timer.OutputStats(&timing_stats);

      if(num_processed>0) {
        CompactLattice clat;
        ConvertLattice(lat, &clat);
        //        bool end_of_utterance = true;
        //        decoder.GetLattice(end_of_utterance, &clat);

        params.lock.lock();
        GetDiagnosticsAndPrintOutput(utt, word_syms, clat,
            &num_frames, &tot_like);


        // In an application you might avoid updating the adaptation state if
        // you felt the utterance had low confidence.  See lat/confidence.h
        //feature_pipeline.GetAdaptationState(&adaptation_state);

        // we want to output the lattice with un-scaled acoustics.
        //BaseFloat inv_acoustic_scale =
        //  1.0 / decodable_opts.acoustic_scale;
        //ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);

        if(params.write_lattice)
          params.clat_writer->Write(utt, clat);
        //KALDI_LOG << "Decoded utterance " << utt;
        params.lock.unlock();

        num_done++;
      }
    } //end for
  } //end while
  if(num_processed > 0 ) {
    //timing_stats.Print(online);
    double total_time;
    timing_stats.GetStats(total_time, params.total_audio[th_idx]);

    KALDI_LOG << "Thread: " << th_idx << " Decoded " << num_done << " utterances, "
      << num_err << " with errors.";
    KALDI_LOG << "Thread: " << th_idx << " Overall likelihood per frame was " << (tot_like / num_frames)
      << " per frame over " << num_frames << " frames.";
  }
  delete word_syms; // will delete if non-NULL.
  auto finish = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> total_time = finish-start;
  params.total_time[th_idx]= total_time.count();
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    DecodeParams params;
    const char *usage =
      "Reads in wav file(s) and simulates online decoding with neural nets\n"
      "(nnet3 setup), with optional iVector-based speaker adaptation and\n"
      "optional endpointing.  Note: some configuration values and inputs are\n"
      "set via config files whose filenames are passed as options\n"
      "\n"
      "Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
      "<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier>\n"
      "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
      "you want to decode utterance by utterance.\n";

    std::string word_syms_rxfilename;

    BaseFloat chunk_length_secs = 0.18;
    bool do_endpointing = false;
    bool online = true;
    int num_threads= 8;
    bool write_lattice = true;
    bool replicate=false;

    ParseOptions po(usage);

    po.Register("write-lattice",&write_lattice, "Output lattice to a file.  Setting to false is useful when benchmarking.");
    po.Register("num-threads", &num_threads, "number of threads to decode in parallel.");
    po.Register("replicate",&replicate,"Replicate computation across all threads (for benchmarking)\n");

    po.Register("chunk-length", &chunk_length_secs,
        "Length of chunk size in seconds, that we process.  Set to <= 0 "
        "to use all input in one chunk.");
    po.Register("word-symbol-table", &word_syms_rxfilename,
        "Symbol table for words [for debug output]");
    po.Register("do-endpointing", &do_endpointing,
        "If true, apply endpoint detection");
    po.Register("online", &online,
        "You can set this to false to disable online iVector estimation "
        "and have all the data for each utterance used, even at "
        "utterance start.  This is useful where you just want the best "
        "results and don't care about online operation.  Setting this to "
        "false has the same effect as setting "
        "--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
        "in the file given to --ivector-extraction-config, and "
        "--chunk-length=-1.");
    po.Register("num-threads-startup", &g_num_threads,
        "Number of threads used when initializing iVector extractor.");


    OnlineNnet2FeaturePipelineConfig  feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    FasterDecoderOptions decoder_opts;
    OnlineEndpointConfig endpoint_opts;

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po,false);
    endpoint_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      return 1;
    }
     

    auto start = std::chrono::high_resolution_clock::now();

    std::string nnet3_rxfilename = po.GetArg(1),
      fst_rxfilename = po.GetArg(2),
      spk2utt_rspecifier = po.GetArg(3),
      wav_rspecifier = po.GetArg(4),
      clat_wspecifier = po.GetArg(5);
    CompactLatticeWriter clat_writer(clat_wspecifier);

    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

    if(!online) chunk_length_secs = -1.0;

    vector<std::thread> threads;

    params.total_time.resize(num_threads);
    params.total_audio.resize(num_threads);

    params.replicate=replicate;
    params.num_threads=num_threads;
    params.chunk_length_secs=chunk_length_secs;
    params.online=online;
    params.write_lattice=write_lattice;
    params.feature_opts=feature_opts;
    params.decodable_opts=decodable_opts;
    params.decoder_opts=decoder_opts;
    params.clat_writer=&clat_writer; 
    params.decode_fst=decode_fst;
    params.nnet3_rxfilename=nnet3_rxfilename;
    params.wav_rspecifier=wav_rspecifier;
    params.word_syms_rxfilename=word_syms_rxfilename;
    params.spk2utt_rspecifier=spk2utt_rspecifier;

    params.utt_idx=0;


    //launch decoding threads
    for (int i=0;i<num_threads;i++) 
      threads.push_back(std::thread(decode_function,std::ref(params),i));

    //wait for threads to complete
    for(int i=0;i<num_threads;i++)
      threads[i].join();

    clat_writer.Close();
    delete decode_fst;
    
    auto finish = std::chrono::high_resolution_clock::now();
    double total_audio=0;
    for(int i=0;i<num_threads;i++) {
        total_audio+=params.total_audio[i];
  KALDI_LOG << "Thread: " << i 
                  << " Total Time: " << params.total_time[i] 
                  << " Total Audio: " << params.total_audio[i]
                  << " RealTimeX: " << params.total_audio[i]/params.total_time[i] << std::endl;
    }

    
    std::chrono::duration<double> total_time = finish-start;

    KALDI_LOG << "Aggregate Total Time: " << total_time.count()
              << " Total Audio: " << total_audio 
              << " RealTimeX: " << total_audio/total_time.count() << std::endl;


    return 0;

    //return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()

