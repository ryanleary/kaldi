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
#include "online2/online-nnet3-cuda-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"
#include <nvToolsExt.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include <atomic>
#include <thread>
#include <chrono>

#define MAX_DEVS 16
using namespace kaldi;
/**************************************************
 * Things that are not currenlty working/designed or may need improvements
 *
 * Reentrant/Online decoding. Currently API is enter once per file
 * 
 * Data duplication.  WaveData and Lattice are duplicated and copied.  This keeps the API simple at the cost of memory and performance
 *  An alternative design could have the user provide a reference to both the WaveData and Lattice and we use that directly.  This avoids
 *  redundant memory and copies and also may be useful for online processing
 *
 * Ivector and nnet processing must be extracted from the Kaldi decoder.  Currently it all happens under AdvanceDecoding/AdvanceChunk.  Ideally
 * these steps would be exposed directly to this class so that we could batch them up and execute seperately.
 */


/*
 *  ThreadedBatchedCudaDecoder uses multiple levels of parallelism in order to decode quickly on CUDA GPUs.
 *  It's API is utterance centric using deferred execution.  That is a user submits work one utterance at a time
 *  and the class batches that work behind the sceen. Utterance are passed into the API with a unqiue key of type string.
 *  The user must ensure this name is unique.  APIs are provided to enque work, query the best path, and flush compute.
 *  Compute is flushed automatically when the desired batch size is reached or when a user manually calls Flush. Flush must
 *  be called manually prior to querying the best path.  After querying the best path a user should call CloseDecodeHandle 
 *  for that utterance.  At that point the utterance is no longer valid, can no longer be queried, and the utterance key can 
 *  be reused.
 *  
 *  Example Usage is as follows:
 *  ThreadedBatchedCudaDecoder decoder;
 *  decoder.Initalize(decode_fst, am_nnet_rx_file);
 *   
 *  //some loop
 *    std::string utt_key = ...
 *    decoder.OpenDecodeHandle(utt_key,wave_data);
 *
 *  decoder.Flush();
 *
 *  //some loop
 *    Lattice lat;
 *    std::string utt_key = ...
 *    decoder.GetBestPath(utt_key,&lat);
 *    decoder.CloseDecodeHandle(utt_key);
 *
 *  decoder.Finalize();
 */
class ThreadedBatchedCudaDecoder {
  public:

    ThreadedBatchedCudaDecoder() : maxBatchSize_(10), numThreads_(1), taskCount_(0) {};

    void Register(ParseOptions *po) {
      feature_opts_.Register(po);
      decodable_opts_.Register(po);
      decoder_opts_.Register(po);
      po->Register("max-batch-size",&maxBatchSize_, "The maximum batch size to be used by the decoder.");
      po->Register("num-threads",&numThreads_, "The number of workpool threads to use in the ThreadedBatchedCudaDecoder");
    };

    //allocates reusable objects that are common across all decodings
    void Initialize(const fst::Fst<fst::StdArc> &decode_fst, std::string nnet3_rxfilename) {
      KALDI_LOG << "ThreadedBatchedCudaDecoder Initialize with " << numThreads_ << " threads\n";
      //Storing nnet3_rxfile.  TODO should we take in the nnet instead of reading it?
      cuda_fst_.Initialize(decode_fst); 

      thread_contexts_.resize(numThreads_);

      thread_status_ = new std::atomic<ThreadStatus>[numThreads_];
      free_threads_ = new std::atomic<int>[numThreads_+1];

      tasks_.resize(numThreads_);

      //create per thread state
      for (int i=0;i<numThreads_;i++) {
        thread_status_[i]=IDLE;
        free_threads_[i]=i;
        tasks_[i].reserve(maxBatchSize_);
      }
      free_threads_front_=0;
      free_threads_back_=numThreads_;
      	
      //initialize threads and save their contexts so we can join them later
      for (int i=0;i<numThreads_;i++) {
        thread_contexts_[i]=std::thread(&ThreadedBatchedCudaDecoder::ExecuteWorker,this,i);
      }
     
      //read transition model and nnet
		  bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model_.Read(ki.Stream(), binary);
      am_nnet_.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet_.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet_.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet_.GetNnet()));

    }
    void Finalize() {

      //Tell threads to exit and join them
      for(int i=0;i<numThreads_;i++) {
        while(thread_status_[i]==PROCESSING);
        thread_status_[i]=EXIT;
        thread_contexts_[i].join();
      }

      cuda_fst_.Finalize();

      delete[] thread_status_;
      delete[] free_threads_;
    }

    //query a specific key to see if compute on it is complete
    bool isFinished(const std::string &key) {
      auto it=tasks_lookup_.find(key);
      KALDI_ASSERT(it!=tasks_lookup_.end());
      return it->second.finished;
    }

    //remove an audio file from the decoding and clean up resources
    void CloseDecodeHandle(const std::string &key) {
      auto it=tasks_lookup_.find(key);
      KALDI_ASSERT(it!=tasks_lookup_.end());

      TaskState &state = it->second;
      KALDI_ASSERT(state.finished==true);

      tasks_lookup_.erase(it);
    }


    //Adds a decoding task to the decoder
    void OpenDecodeHandle(const std::string &key, const WaveData &wave_data) {

      //Wait for a thread to be free
      while (free_threads_front_==free_threads_back_) {
        //std::this_thread::sleep_for(std::chrono::nanoseconds(50)); 
      }

      //grab a free thread
      int threadId;
      threadId = free_threads_[free_threads_front_];

      //Get the task list for this thread
      auto &tasks = tasks_[threadId];

      //Create a new task in lookup map
      //TODO check if it already exists and throw an ASSERT?
      TaskState* t=&tasks_lookup_[key];

      t->Init(wave_data); 

      //Insert new task into work queue
      tasks.push_back(t);

      taskCount_++;

      //If maxBatchSize is reached start work
      if (tasks.size()==maxBatchSize_) {
				StartThread(threadId);
      }
    }

    void Flush() {

      //If there are not outstanding tasks then no need to flush
      if (taskCount_==0) return;

      //wait for a free thread
      while (free_threads_front_==free_threads_back_) {
        //std::this_thread::sleep_for(std::chrono::nanoseconds(50)); 
      }

      //grab a free thread
      int threadId;
      threadId = free_threads_[free_threads_front_];

      StartThread(threadId);
    }

    void GetBestPath(const std::string &key, Lattice *lat) {
      auto it=tasks_lookup_.find(key);
      KALDI_ASSERT(it!=tasks_lookup_.end());

      TaskState *state = &it->second;

      //Note it is assumed that the user has called Flush().  If they have not then this could deadlock.

      while(state->finished==false);
      
      //Store off the lattice
      *lat=state->lat;
    }


  private:

    //A status for each thread.  
    //IDLE=thread is ready to be assigned work.  
    //PROCESSING=thread has been assigned work and is processing it. 
    //EXIT=thread as been asked to exit.
    enum ThreadStatus { IDLE, PROCESSING, EXIT };
    
    //State needed for each decode task.  
    //WaveData is duplicated,  a reference may be necessary for online decoding.  This would allow wave data to populate while running.  //TODO figure this out
    struct TaskState {
      WaveData wave_data;   //Wave data input
      Lattice lat;          //Lattice output
      std::atomic<bool> finished;

      TaskState() : finished(false) {};
      void Init(const WaveData &wave_data_in) { wave_data=wave_data_in; finished=false; };
    };

    void StartThread(int threadId) {
      //Reset task count
      taskCount_=0;
      //Remove thread from free list
      free_threads_front_=(free_threads_front_+1)%(numThreads_+1);
		  //Notifiy worker thread to start processing
      thread_status_[threadId]=PROCESSING;
    }

    void ExecuteWorker(int threadId) {

      //Reference to thread state used to control this threads execution
      std::atomic<ThreadStatus> &thread_state=thread_status_[threadId];

      CuDevice::Instantiate().SelectGpuId(0);
      CuDevice::Instantiate().AllowMultithreading();

      //reusable across decodes
      std::vector<OnlineNnet2FeaturePipelineInfo*> feature_infos(maxBatchSize_);
      std::vector<nnet3::AmNnetSimple> am_nnets(maxBatchSize_);  //cannot currently be shared due to writes from DecodableNnetSimpleLoopedInfo
      std::vector<nnet3::DecodableNnetSimpleLoopedInfo*> decodable_infos(maxBatchSize_);
      std::vector<CudaDecoder*> cuda_decoders(maxBatchSize_);

      //reallocated each decode
      std::vector<OnlineNnet2FeaturePipeline*> feature_pipelines(maxBatchSize_);
      std::vector<SingleUtteranceNnet3CudaDecoder*> decoders(maxBatchSize_);

      am_nnets.clear();

      //TODO can any of this be shared across multiple decodes?  
      for (int i=0;i<maxBatchSize_;i++) {
        feature_infos[i]=new OnlineNnet2FeaturePipelineInfo(feature_opts_);

        feature_infos[i]->ivector_extractor_info.use_most_recent_ivector = true;
        feature_infos[i]->ivector_extractor_info.greedy_ivector_extractor = true;
     
        //copy from original nnet.  For some reason we cannot use these concurrently.  It looks like the next constructor
        //reallocates memory inside of am_nnets which is problematic for concurrency.
        am_nnets.push_back(am_nnet_);
  
        decodable_infos[i]=new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts_,&am_nnets[i]);
        cuda_decoders[i]=new CudaDecoder(cuda_fst_,decoder_opts_);
      }

      //This threads task list
      std::vector<TaskState*> &tasks = tasks_[threadId];
      
      while (thread_state!=EXIT) {   //check if master as asked us to exit
        if (thread_state==PROCESSING) {  //check if master has told us to process
          int batchSize=tasks.size();
          
          nvtxRangePushA("Allocation");
          for (int i=0;i<batchSize;i++) {
            feature_pipelines[i] = new OnlineNnet2FeaturePipeline(*feature_infos[i]);
            decoders[i] = new SingleUtteranceNnet3CudaDecoder(trans_model_, *decodable_infos[i], *cuda_decoders[i], feature_pipelines[i]);
          }
          nvtxRangePop();

          //process waveform
          nvtxRangePushA("Process Waveform");
          for (int i=0;i<batchSize;i++) {
            TaskState &state = *tasks[i];

            BaseFloat samp_freq = state.wave_data.SampFreq();
            SubVector<BaseFloat> data(state.wave_data.Data(), 0); 

            feature_pipelines[i]->AcceptWaveform(samp_freq, data);
            feature_pipelines[i]->InputFinished();
          }
          nvtxRangePop();

          //feature extract
          //TODO pull out of AdvanceDecoding/AdvanceChunk

          //nnet3
          //TODO pull out of AdvanceDecoding/AdvanceChunk

          nvtxRangePushA("AdvanceDecoding");
          for (int i=0;i<batchSize;i++) {
            decoders[i]->AdvanceDecoding();
          }

          //decode not currently be shared due to writes from DecodableNnetSimpleLoopedInfo
          //ensure compute is compete before proceeeding
          cudaStreamSynchronize(cudaStreamPerThread);
          nvtxRangePop();

          nvtxRangePushA("GetBestPath");
          for (int i=0;i<batchSize;i++) {
            TaskState &state = *tasks[i];
            decoders[i]->GetBestPath(true,&state.lat);
            state.finished=true;
          } 
          nvtxRangePop();
          
          nvtxRangePushA("Deallocate");
          for (int i=0;i<batchSize;i++) {
            delete feature_pipelines[i];
            delete decoders[i];
          }
          nvtxRangePop();

          //We are now complete. Clean up data structures
          tasks.clear();
          thread_state=IDLE;
          
          //add myself to the free threads list
          free_threads_mutex_.lock();
          {
            free_threads_[free_threads_back_]=threadId;
            free_threads_back_=(free_threads_back_+1)%(numThreads_+1);
          }
          free_threads_mutex_.unlock();
        
        } //end if
        //else { sleep(1); }
      } //End while !EXIT loop
     
      //cleanup
      for (int i=0;i<maxBatchSize_;i++) {
        delete decodable_infos[i];
        delete cuda_decoders[i];
        delete feature_infos[i];
      }
    }
    
    CudaFst cuda_fst_;

    OnlineNnet2FeaturePipelineConfig  feature_opts_;           //constant readonly
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts_; //constant readonly
    CudaDecoderConfig decoder_opts_;                           //constant readonly
    TransitionModel trans_model_;
    nnet3::AmNnetSimple am_nnet_;

    int maxBatchSize_;  //Target CUDA batch size
    int numThreads_;    //The number of CPU threads
    int taskCount_;     //Current number of tasks that have been enqueued but not launched
    
    //Free threads is a circular queue.  The front points to the current free thread. 
    //The back points to the slot to write a new free thread to.  If font==back then there
    //are no free threads available.  mutex and atomic required here to get a thread safe
    //memory model without excessive locking.
    std::mutex free_threads_mutex_;
    std::atomic<int> *free_threads_;
    std::atomic<int> free_threads_back_;
    int free_threads_front_;

    std::map<std::string,TaskState> tasks_lookup_; //Contains a map of utterance to TaskState
    std::vector<std::vector<TaskState*> > tasks_;  //List of tasks assigned to each thread.  The list contains pointers to the TaskState.
    std::atomic<ThreadStatus>* thread_status_;      //A list of thread status
    std::vector<std::thread> thread_contexts_;     //A list of thread contexts
};



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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

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

    bool write_lattice = true;
    int num_todo = INT_MAX;
    ParseOptions po(usage);

    po.Register("write-lattice",&write_lattice, "Output lattice to a file.  Setting to false is useful when benchmarking.");
    po.Register("word-symbol-table", &word_syms_rxfilename, "Symbol table for words [for debug output]");
    po.Register("file-limit", &num_todo, "Limits the number of files that are processed by this driver.  After N files are processed the remaing files are ignored.  Useful for profiling.");
    //Multi-threaded CPU and batched GPU decoder
    ThreadedBatchedCudaDecoder CudaDecoder;
    OnlineEndpointConfig endpoint_opts;  //TODO is this even used?  Config seems to need it but we never seem to use it.

    CudaDecoder.Register(&po);
    endpoint_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      return 1;
    }

    CuDevice::Instantiate().SelectGpuId(0);
    CuDevice::Instantiate().AllowMultithreading();

    nvtxRangePush("Global Timer");
    auto start = std::chrono::high_resolution_clock::now();

    std::string nnet3_rxfilename = po.GetArg(1),
      fst_rxfilename = po.GetArg(2),
      spk2utt_rspecifier = po.GetArg(3),
      wav_rspecifier = po.GetArg(4),
      clat_wspecifier = po.GetArg(5);

    CompactLatticeWriter clat_writer(clat_wspecifier);
    
    fst::Fst<fst::StdArc> *decode_fst= fst::ReadFstKaldiGeneric(fst_rxfilename);

    CudaDecoder.Initialize(*decode_fst, nnet3_rxfilename);

    delete decode_fst;

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
          << word_syms_rxfilename;

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
    OnlineTimingStats timing_stats;

    std::vector<std::string> processed;

    double total_audio=0;
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      nvtxRangePushA("Speaker Iteration");
      std::string spk = spk2utt_reader.Key();
      printf("Speaker: %s\n",spk.c_str());

      const std::vector<std::string> &uttlist = spk2utt_reader.Value();

      for (size_t i = 0; i < uttlist.size(); i++) {
        nvtxRangePushA("Utterance Iteration");

        std::string utt = uttlist[i];
        printf("Utterance: %s\n", utt.c_str());
        if (!wav_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find audio for utterance " << utt;
          num_err++;
          continue;
        }

        const WaveData &wave_data = wav_reader.Value(utt);
        total_audio+=wave_data.Duration();

        CudaDecoder.OpenDecodeHandle(utt,wave_data);
        processed.push_back(utt);
        num_done++;
        nvtxRangePop();
        if (num_done>num_todo) break;
      } //end utterance loop
      nvtxRangePop();
      if (num_done>num_todo) break;
    } //end speaker loop

    CudaDecoder.Flush();

    nvtxRangePushA("Lattice Write");
    for (int i=0;i<processed.size();i++) {
      std::string &utt = processed[i];
      Lattice lat;
      CompactLattice clat;

      CudaDecoder.GetBestPath(utt,&lat);
      ConvertLattice(lat, &clat);

      GetDiagnosticsAndPrintOutput(utt, word_syms, clat, &num_frames, &tot_like);

      if (write_lattice) {
        clat_writer.Write(utt, clat);
      }

      CudaDecoder.CloseDecodeHandle(utt);
    } //end for
    nvtxRangePop();

    KALDI_LOG << "Decoded " << num_done << " utterances, "
      << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
      << " per frame over " << num_frames << " frames.";

    delete word_syms; // will delete if non-NULL.

    clat_writer.Close();

    CudaDecoder.Finalize();  
    cudaDeviceSynchronize();

    auto finish = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

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

