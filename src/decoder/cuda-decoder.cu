// decoder/simple-decoder.cc

// Copyright 2009-2011 Microsoft Corporation
//           2012-2013 Johns Hopkins University (author: Daniel Povey)

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

#include "decoder/cuda-decoder.h"
#include "fstext/remove-eps-local.h"
#include <algorithm>
#include <nvToolsExt.h>
#include <cuda_runtime_api.h>
#include <float.h>
#include <math.h>
#include <cooperative_groups.h>

#define MEMADVISE

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

#define DIV_ROUND_UP(a,b) ((a+b-1)/b)
namespace kaldi {

  template <typename T>
  DEVICE __forceinline__ void load16(T *a, const T *b) {
    const ulong2 *src = reinterpret_cast<const ulong2*>(b);
    ulong2 &dst = *reinterpret_cast<ulong2*>(a);
    asm("ld.global.v2.u64 {%0,%1}, [%2];" : "=l"(dst.x), "=l"(dst.y) : "l"(src));
  }
  
template <typename T>
  DEVICE __forceinline__ void store16(T *a, const T *b) {
    const ulong2 src = *reinterpret_cast<const ulong2*>(b);
    asm("st.global.v2.u64 [%0], {%1,%2};" :: "l"(a), "l"(src.x), "l"(src.y));
  }




// Assumptions: 1-d grid and blocks. No threads "early-exit" the grid.
// No stream priorities
DEVICE inline void __gpu_sync_fast(volatile int *fast_epoch)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        // gridDim.x-1 blocks are adding 1
        // and one block is adding 0x80000000 - (gridDim.x-1)
        // so the whole sum is 0x80000000
        int nb = 1;
        if (blockIdx.x == 0) {
            nb = 0x80000000 - (gridDim.x-1);
        }
 
        int old_epoch = *fast_epoch;
        __threadfence();
        atomicAdd((int*)fast_epoch, nb);
 
        // wait for the sign bit to commute   
        while (((*fast_epoch) ^ old_epoch) >= 0)
            ;
    }
    __syncthreads();
}

DEVICE __noinline__ void __grid_sync_nv_internal(int *barrier)
{
    __gpu_sync_fast((volatile int*)barrier);
}

  template<typename T> 
    inline DEVICE void swap(T &a, T &b) {
      T c = a;
      a = b;
      b = c;
    }

  /******************************************CudaVector Implementation*******************************/
  template<typename T>
    HOST DEVICE inline T& CudaVector<T>::operator[](uint32_t idx) { 
#ifdef __CUDA_ARCH__
      return mem_d[idx];
#else
      return mem_h[idx];
#endif
    }

  template<typename T>
    HOST DEVICE inline const T& CudaVector<T>::operator[](uint32_t idx) const { 
#ifdef __CUDA_ARCH__
      return mem_d[idx];
#else
      return mem_h[idx];
#endif
    } 

  template<typename T>
    inline void CudaVector<T>::allocate(uint32_t max_size) {
      this->max_size=max_size;

      cudaMallocHost(&count_h,sizeof(uint32_t));
      cudaMalloc(&count_d, sizeof(uint32_t));
      cudaMemset(count_d, 0,sizeof(uint32_t));
      *count_h=0;

      cudaMalloc(&mem_d,max_size*sizeof(T));
      cudaMallocHost(&mem_h,max_size*sizeof(T));
    }

  template<typename T>
    inline size_t CudaVector<T>::getCudaMallocBytes() {
      return sizeof(uint32_t)+max_size*sizeof(T);
    }

  template<typename T>
    inline void CudaVector<T>::free() { 
      cudaFree(mem_d); 
      cudaFreeHost(mem_h);
      cudaFreeHost(count_h);
    }


  template<typename T>
    inline void CudaVector<T>::copy_all_to_host(cudaStream_t stream) {
      cudaStreamSynchronize(stream);
      cudaMemcpy(count_h,count_d,sizeof(int32),cudaMemcpyDeviceToHost);
      cudaMemcpyAsync(mem_h,mem_d,*count_h*sizeof(T),cudaMemcpyDeviceToHost, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_all_to_device(cudaStream_t stream) {
      cudaStreamSynchronize(stream);
      cudaMemcpyAsync(count_d,count_h,sizeof(int32),cudaMemcpyHostToDevice);
      cudaMemcpyAsync(mem_d,mem_h,*count_h*sizeof(T),cudaMemcpyHostToDevice, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_size_to_host(cudaStream_t stream) {
      cudaMemcpyAsync(count_h,count_d,sizeof(int32),cudaMemcpyDeviceToHost, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_size_to_device(cudaStream_t stream) {
      cudaMemcpyAsync(count_d,count_h,sizeof(int32),cudaMemcpyHostToDevice, stream);
    }
  
template<typename T>
    inline void CudaVector<T>::copy_data_to_host(cudaStream_t stream) {
      cudaMemcpyAsync(mem_h,mem_d,*count_h*sizeof(T),cudaMemcpyDeviceToHost, stream);
    }

  template<typename T>
    inline void CudaVector<T>::copy_data_to_device(cudaStream_t stream) {
      cudaMemcpyAsync(mem_d,mem_h,*count_h*sizeof(T),cudaMemcpyHostToDevice, stream);
    }


  //Note:  This will cause page faults back and forth when we switch from host to device.
  template<typename T>
    HOST DEVICE inline uint32_t CudaVector<T>::size() const 
    {
#ifdef __CUDA_ARCH__
      return *count_d; 
#else
      return *count_h;
#endif
    }

  template<typename T> 
    HOST DEVICE inline void CudaVector<T>::push_back(const T &val) { 
#ifdef __CUDA_ARCH__
      //assert(*count_d<max_size);
      uint32_t idx = atomicAdd(count_d,1);
      mem_d[idx]=val; 
#else
      assert(*count_h<max_size);
      uint32_t idx = (*count_h)++;
      mem_h[idx]=val; 
#endif
    }
  template<typename T> 
    HOST DEVICE inline void CudaVector<T>::clear(cudaStream_t stream) { 
#ifdef __CUDA_ARCH__
      *count_d = 0;
#else
      *count_h = 0; 
      cudaMemsetAsync(count_d,0,sizeof(int32),stream); 
#endif
    }
  template<typename T> 
    inline bool CudaVector<T>::empty() const { return size()==0; }
  template<typename T> 
    inline void CudaVector<T>::swap(CudaVector<T> &v) {
      std::swap(mem_h,v.mem_h);
      std::swap(mem_d,v.mem_d);
      std::swap(count_h,v.count_h);
      std::swap(count_d,v.count_d);
      std::swap(max_size,v.max_size);
    }
  /**************************************End CudaVector Implementation**********************************/

  /***************************************CudaFst Implementation*****************************************/
  HOST DEVICE inline float CudaFst::Final(StateId state) const {
    #ifdef __CUDA_ARCH__
    return final_d[state];
    #else
    return final_h[state];
    #endif

  }
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
    cudaMalloc(&final_d,sizeof(float)*numStates);

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

    cudaMemcpyAsync(final_d,final_h,sizeof(float)*numStates,cudaMemcpyHostToDevice,cudaStreamPerThread);
    
    cudaMemcpyAsync(e_offsets_d,e_offsets_h,sizeof(unsigned int)*(numStates+1),cudaMemcpyHostToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(ne_offsets_d,ne_offsets_h,sizeof(unsigned int)*(numStates+1),cudaMemcpyHostToDevice, cudaStreamPerThread);


    //Allocate non-zero arrays
    cudaMallocHost(&arc_weights_h,arc_count*sizeof(BaseFloat));
    cudaMallocHost(&arc_nextstates_h,arc_count*sizeof(StateId));
    cudaMallocHost(&arc_ilabels_h,arc_count*sizeof(int32));
    cudaMallocHost(&arc_olabels_h,arc_count*sizeof(int32));

    cudaMalloc((void**)&arc_weights_d,arc_count*sizeof(BaseFloat));  bytes_cudaMalloc+=arc_count*sizeof(BaseFloat);
    cudaMalloc((void**)&arc_nextstates_d,arc_count*sizeof(StateId));  bytes_cudaMalloc+=arc_count*sizeof(StateId);
    cudaMalloc((void**)&arc_ilabels_d,arc_count*sizeof(int32));  bytes_cudaMalloc+=arc_count*sizeof(int32);
    
    //now populate arc data
    int e_idx=1;          //save room for dummy arc (so start at 1)
    int ne_idx=e_count+1; //starts where e_offsets ends

    //create dummy arc
    arc_weights_h[0]=StdWeight::One().Value();
    arc_nextstates_h[0]=fst.Start();
    arc_ilabels_h[0]=0;
    arc_olabels_h[0]=0;

    for(int i=0;i<numStates;i++) {
      //count emmiting and non_emitting arcs

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

    cudaMemcpyAsync(arc_weights_d,arc_weights_h,arc_count*sizeof(BaseFloat),cudaMemcpyHostToDevice,cudaStreamPerThread);
    cudaMemcpyAsync(arc_nextstates_d,arc_nextstates_h,arc_count*sizeof(StateId),cudaMemcpyHostToDevice,cudaStreamPerThread);
    cudaMemcpyAsync(arc_ilabels_d,arc_ilabels_h, arc_count*sizeof(int32),cudaMemcpyHostToDevice,cudaStreamPerThread);

    cudaStreamSynchronize(cudaStreamPerThread);
    nvtxRangePop();
  }

  void CudaFst::finalize() {
    nvtxRangePushA("CudaFst destructor");
    cudaFreeHost(final_h);
    cudaFree(final_d);
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

  DEVICE inline void allocateAllTokens_function(CudaDecoder::TokenLookupElem *current_tokens_lookup, int32 numStates,  CudaDecoder::TokenAllocator allocator) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<numStates; i+=blockDim.x*gridDim.x) {
      CudaDecoder::Token *token = allocator.getToken(i);
      token->cost_ = INFINITY;
      token->prev_ = NULL;
      CudaDecoder::TokenLookupElem elem;
      elem.token=token;
      elem.active=false;
      store16(&current_tokens_lookup[i], &elem);
    }
  }
  __global__ void allocateAllTokens(CudaDecoder::TokenLookupElem *current_tokens_lookup, int32 numStates,  CudaDecoder::TokenAllocator allocator, int *barrier) {
    allocateAllTokens_function(current_tokens_lookup,numStates,allocator);
     __grid_sync_nv_internal(barrier);
     if(blockIdx.x==0 && threadIdx.x==0) {
      allocator.advanceFront(numStates);
     }
  }

  DEVICE inline void allocateNewTokens_function(CudaDecoder::TokenLookupElem *current_tokens_lookup, CudaDecoder::TokenVector cur_toks, CudaDecoder::TokenAllocator allocator) {
    int32 size = cur_toks.size();
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=blockDim.x*gridDim.x) {
      CudaDecoder::Token *token = allocator.getToken(i);
      token->cost_ = INFINITY;
      token->prev_ = NULL;
      CudaDecoder::StateId state=cur_toks[i].state;
      CudaDecoder::TokenLookupElem elem;
      elem.token=token;
      elem.active=false;
      store16(&current_tokens_lookup[state], &elem);
    }
  }

  
  void CudaDecoder::TokenAllocator::prefetch_next_to_device(cudaStream_t stream) {
    prefetch_next_to_device(stream,prefetch_size);
  }

  void CudaDecoder::TokenAllocator::prefetch_next_to_device(cudaStream_t stream, int count) {
    int front = *front_h;
    //clamp to maximum size
    if(count>size-front)
      count = size-front;

    cudaMemPrefetchAsync(tokens_allocation+front,sizeof(Token)*count,device,stream);  
  }

  void CudaDecoder::TokenAllocator::prefetch_allocated_to_host(cudaStream_t stream) {
    cudaMemPrefetchAsync(tokens_allocation,sizeof(Token)* *front_h,cudaCpuDeviceId,stream);  
  }

  size_t CudaDecoder::TokenAllocator::getCudaMallocManagedBytes() {
    return bytes_cudaMallocManaged;
  }

  void CudaDecoder::TokenAllocator::reset() {
    *front_h=0;
    cudaMemset(front_d,0,sizeof(int));
  }

  void CudaDecoder::TokenAllocator::initialize(uint32_t size)  {
    cudaGetDevice(&device);
    prefetch_size=250000;

    this->size = size;

    //managed so getBestPath can easily access this data in the end
    cudaMallocManaged((void**)&tokens_allocation,sizeof(Token)*size);  
    bytes_cudaMallocManaged=sizeof(Token)*size;

    cudaMalloc((void**)&front_d,sizeof(uint32_t)); 
    cudaMallocHost((void**)&front_h,sizeof(uint32_t)); 

#ifdef MEMADVISE
    //If we do this we get faster perf as long as we don't over subscribe
    cudaMemAdvise(tokens_allocation,sizeof(Token)*size,cudaMemAdviseSetPreferredLocation,device);
    cudaMemPrefetchAsync(tokens_allocation,sizeof(Token)*size,device);  //force pages to allocate now
#endif

    reset();
  }

  void CudaDecoder::TokenAllocator::finalize() {
    cudaFree(tokens_allocation);
    cudaFree(front_d);
    cudaFreeHost(front_h);
  }

  DEVICE inline CudaDecoder::Token* CudaDecoder::TokenAllocator::getToken(uint32_t offset) {
    int idx = *front_d + offset;
    return &tokens_allocation[idx];
  }

  DEVICE inline void CudaDecoder::TokenAllocator::advanceFront(uint32_t num) {
    int front = *front_d + num;
    //assert(front<size);
    
    *front_d=front;
    *front_h=front;
  }


  CudaDecoder::CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config): fst_(fst), beam_(config.beam), bytes_cudaMalloc(0), bytes_cudaMallocManaged(0) {
    int device;
    cudaGetDevice(&device);


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,device);

    total_threads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount * config.gpu_fraction;

    allocator.initialize(config.max_tokens);

    bytes_cudaMallocManaged+=allocator.getCudaMallocManagedBytes();
    cur_toks_.allocate(config.max_tokens_per_frame);
    prev_toks_.allocate(config.max_tokens_per_frame);
    bytes_cudaMalloc+=cur_toks_.getCudaMallocBytes()+prev_toks_.getCudaMallocBytes();

    cudaEventCreateWithFlags(&event_pt,cudaEventDisableTiming);
    cudaEventCreateWithFlags(&event_pt_old,cudaEventDisableTiming);
    cudaEventCreateWithFlags(&event_ll,cudaEventDisableTiming);

    cudaStreamCreateWithFlags(&stream_comp, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_copy, cudaStreamNonBlocking);
    cudaStreamCreateWithPriority(&stream_ll, cudaStreamNonBlocking, -1);

    cudaMalloc(&pe_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&ne_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&fb_idx_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);
    cudaMalloc(&barrier_d, sizeof(int)); bytes_cudaMalloc+=sizeof(int);

    cudaMemset(pe_idx_d,0,sizeof(int));
    cudaMemset(ne_idx_d,0,sizeof(int));
    cudaMemset(fb_idx_d,0,sizeof(int));
    cudaMemset(barrier_d,0,sizeof(int));

    cudaMalloc(&cutoff_d, sizeof(CostType)); bytes_cudaMalloc+=sizeof(CostType);
    cudaMalloc(&modified_d, sizeof(int)*2); bytes_cudaMalloc+=sizeof(CostType)*2;

    cudaMalloc(&token_locks_d,sizeof(int)*fst_.numStates);  bytes_cudaMalloc+=sizeof(int)*fst_.numStates;
    cudaMemset((void*)token_locks_d,0,sizeof(int)*fst_.numStates);

    cudaMalloc((void**)&current_tokens_lookup_d,sizeof(TokenLookupElem)*fst_.numStates); bytes_cudaMalloc+=sizeof(TokenLookupElem)*fst_.numStates;

    cudaMallocHost(&loglikelihoods_h,sizeof(BaseFloat)*(fst_.max_ilabel+1));  
    cudaMallocHost(&loglikelihoods_old_h,sizeof(BaseFloat)*(fst_.max_ilabel+1));

    cudaMalloc((void**)&loglikelihoods_d,sizeof(BaseFloat)*(fst_.max_ilabel+1)); bytes_cudaMalloc+=sizeof(BaseFloat)*(fst_.max_ilabel+1);
    cudaMalloc((void**)&loglikelihoods_old_d,sizeof(BaseFloat)*(fst_.max_ilabel+1)); bytes_cudaMalloc+=sizeof(BaseFloat)*(fst_.max_ilabel+1);

    cudaStreamSynchronize(stream_comp);
    cudaStreamSynchronize(stream_copy);
    cudaStreamSynchronize(cudaStreamPerThread);

    //sgemm requires shared memory and we don't want cache config changing.  So set a device wide cache config.
    cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
  }

  CudaDecoder::~CudaDecoder() {

    cur_toks_.free();
    prev_toks_.free();
    allocator.finalize();

    cudaFreeHost(loglikelihoods_h);
    cudaFreeHost(loglikelihoods_old_h);
    cudaFree(loglikelihoods_d);
    cudaFree(loglikelihoods_old_d);
    cudaFree(current_tokens_lookup_d);

    cudaFree(pe_idx_d);
    cudaFree(ne_idx_d);
    cudaFree(fb_idx_d);
    cudaFree(barrier_d);

    cudaFree((void*)token_locks_d);
    cudaFree(cutoff_d);
    cudaFree(modified_d);

    cudaEventDestroy(event_pt);
    cudaEventDestroy(event_pt_old);
    cudaEventDestroy(event_ll);

    cudaStreamDestroy(stream_comp);
    cudaStreamDestroy(stream_copy);
    cudaStreamDestroy(stream_ll);

  }


  bool CudaDecoder::Decode(DecodableInterface *decodable) {
    nvtxRangePushA("CudaDecoder::Decode");

    InitDecoding();

    ComputeLogLikelihoods(decodable);

    while( !decodable->IsLastFrame(num_frames_decoded_ - 1)) {

#ifndef MEMADVISE
      //no need to prefetch if we have done a memadvise
      allocator.prefetch_next_to_device(cudaStreamPerThread);
#endif

      //TODO prefetch here

      cur_toks_.swap(prev_toks_);

      ProcessTokens();

      //computes log likelihoods for the next frame
      ComputeLogLikelihoods(decodable);
    }

    cur_toks_.copy_all_to_host(stream_comp);
    cudaStreamSynchronize(stream_comp);

    nvtxRangePop();

    return (!cur_toks_.empty());
  }

  __global__ void addOneToken(CudaDecoder::TokenLookupElem *current_tokens_lookup,  CudaDecoder::TokenVector cur_toks, CudaDecoder::Token tok, CudaDecoder::StateId state) {

    CudaDecoder::TokenLookupElem elem = current_tokens_lookup[state];
    *elem.token = tok;
    current_tokens_lookup[state].active = true;
    cur_toks.push_back(CudaDecoder::TokenState(elem.token,state));   //add token to current token list 
  }

  //putting this into a kernel to avoid extra latency of a memory copy
  __global__ void initializeCutoff(CudaDecoder::CostType *cutoff) {
    *cutoff = INFINITY;
  }

  void CudaDecoder::InitDecoding() {
    // clean up from last time:
    ClearToks(cur_toks_);
    ClearToks(prev_toks_);
    
    allocator.reset();
    int threads=64;
    int blocks=DIV_ROUND_UP(total_threads,threads);
    
    //start moving these / allocating them on the device
    allocator.prefetch_next_to_device(stream_comp, fst_.numStates+5000);

    allocateAllTokens<<<blocks,threads,0,stream_comp>>>(current_tokens_lookup_d, fst_.numStates, allocator, barrier_d);

    // initialize decoding:
    StateId start_state = fst_.Start();
    KALDI_ASSERT(start_state != fst::kNoStateId);

    cudaCheckError();
    Token tok(StdWeight::One().Value(), NULL, 0);
    //Token tok(StdWeight::One().Value(),0, NULL, 0);
    addOneToken<<<1,1,0,stream_comp>>>(current_tokens_lookup_d, cur_toks_, tok, start_state);
    cudaCheckError();

    initializeCutoff<<<1,1,0,stream_comp>>>(cutoff_d);

    num_frames_decoded_ = 0;
    ProcessNonemitting();

  }

  void CudaDecoder::AdvanceDecoding(DecodableInterface *decodable,
      int32 max_num_frames) {
    nvtxRangePushA("AdvanceDecoding");
    KALDI_ASSERT(num_frames_decoded_ >= 0 &&
        "You must call InitDecoding() before AdvanceDecoding()");
    int32 num_frames_ready = decodable->NumFramesReady();
    // num_frames_ready must be >= num_frames_decoded, or else
    // the number of frames ready must have decreased (which doesn't
    // make sense) or the decodable object changed between calls
    // (which isn't allowed).
    KALDI_ASSERT(num_frames_ready >= num_frames_decoded_);
    int32 target_frames_decoded = num_frames_ready;
    if (max_num_frames >= 0)
      target_frames_decoded = std::min(target_frames_decoded,
          num_frames_decoded_ + max_num_frames);

    ComputeLogLikelihoods(decodable);

    int threads=64;
    int blocks=DIV_ROUND_UP(total_threads,threads);
    
    while (num_frames_decoded_ < target_frames_decoded) {

#ifndef MEMADVISE
      //no need to prefetch if we have done a memadvise
      allocator.prefetch_next_to_device(cudaStreamPerThread);
#endif

      cur_toks_.swap(prev_toks_);
      
      ProcessTokens();
      
      //computes log likelihoods for the next frame
      ComputeLogLikelihoods(decodable);
      
    }   
    

    cur_toks_.copy_all_to_host(stream_comp);
    cudaStreamSynchronize(stream_comp);

    nvtxRangePop();
  }

  bool CudaDecoder::ReachedFinal() const {
    for (int i=0;i<cur_toks_.size();i++) {
      TokenState ts = cur_toks_[i];

      if (ts.token->cost_ != std::numeric_limits<BaseFloat>::infinity() &&
          fst_.Final(ts.state) != StdWeight::Zero())
        return true;
    }

    return false;
  }

  BaseFloat CudaDecoder::FinalRelativeCost() const {
    // as a special case, if there are no active tokens at all (e.g. some kind of
    // pruning failure), return infinity.
    CostType infinity = std::numeric_limits<CostType>::infinity();
    if (cur_toks_.empty())
      return infinity;
    CostType best_cost = infinity,
             best_cost_with_final = infinity;


    //for each active token
    //compute minimum cost
    for (int i=0;i<cur_toks_.size();i++) {
      TokenState ts = cur_toks_[i];
      StateId state = ts.state;
      CostType cost = ts.token->cost_;

      // Note: Plus is taking the minimum cost, since we're in the tropical
      // semiring.
      best_cost = std::min(best_cost, cost);
      best_cost_with_final = std::min(best_cost_with_final,
          cost +
          fst_.Final(state));
          //fst_.Final(state).Value());
    }

    BaseFloat extra_cost = best_cost_with_final - best_cost;
    if (extra_cost != extra_cost) { // NaN.  This shouldn't happen; it indicates some
      // kind of error, most likely.
      KALDI_WARN << "Found NaN (likely search failure in decoding)";
      return infinity;
    }
    // Note: extra_cost will be infinity if no states were final.
    return extra_cost;
  }

  // Outputs an FST corresponding to the single best path
  // through the lattice.
  bool CudaDecoder::GetBestPath(Lattice *fst_out, bool use_final_probs) const {
    nvtxRangePushA("GetBestPath");


    fst_out->DeleteStates();
    Token *best_tok = NULL;
    bool is_final = ReachedFinal();
    
    if (!is_final) {
      for(int i=0;i<cur_toks_.size();i++) {
        TokenState ts = cur_toks_[i];
        Token *tok = ts.token;
        if(best_tok==NULL || *best_tok < *tok) {
          best_tok = tok;
        }
      }
    } else {
      CostType infinity =std::numeric_limits<CostType>::infinity(),
               best_cost = infinity;
      for(int i=0;i<cur_toks_.size();i++) {
        TokenState ts = cur_toks_[i];
        Token  *tok = ts.token;
        StateId state = ts.state;
        CostType this_cost = tok->cost_ + fst_.Final(state);
        if (this_cost != infinity && this_cost < best_cost) {
          best_cost = this_cost;
          best_tok = tok;
        }
      }
    }

    if (best_tok == NULL) {
      nvtxRangePop();
      return false;  // No output.
    }

    int count=0;

    //for each token in reverse order
    //add arc to list
    std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.
    for (Token *tok = best_tok; tok != NULL; tok = tok->prev_) {
      count++;
      Token &t=*tok;

      uint32_t arc_idx=t.arc_index_;

      LatticeArc arc(fst_.arc_ilabels_h[arc_idx], fst_.arc_olabels_h[arc_idx], LatticeWeight(fst_.arc_weights_h[arc_idx], 0), fst_.arc_nextstates_h[arc_idx]);

      arcs_reverse.push_back(arc);
    }
    KALDI_ASSERT(arcs_reverse.back().nextstate == fst_.Start());
    arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

    //for each arc in reverse
    //generate new fst
    StateId cur_state = fst_out->AddState();
    fst_out->SetStart(cur_state);
    for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
      LatticeArc arc = arcs_reverse[i];
      arc.nextstate = fst_out->AddState();
      fst_out->AddArc(cur_state, arc);
      cur_state = arc.nextstate;
    }
    if (is_final && use_final_probs)
      fst_out->SetFinal(cur_state,
          LatticeWeight(fst_.Final(fst_.arc_nextstates_h[best_tok->arc_index_]),
            0.0));
    else
      fst_out->SetFinal(cur_state, LatticeWeight::One());
    fst::RemoveEpsLocal(fst_out);
    nvtxRangePop();
    return true;
  }

  inline DEVICE void atomicMin(double *address, double val) {
    unsigned long long *address_ull = (unsigned long long *)address;

    double minval = *address;

    while (val < minval) {  //if my value is less than minimum
      minval = val;         //update the minimum to my value locally
      val = __longlong_as_double(atomicExch(address_ull, __double_as_longlong(val))); //write minimum and read back value
    } //if the new value is < the minimum I wrote I need to try again.
  }
  inline DEVICE void atomicMin(float *address, float val) {
    unsigned int *address_ui = (unsigned int  *)address;

    float minval = *address;

    while (val < minval) {  //if my value is less than minimum
      minval = val;         //update the minimum to my value locally
      val = __uint_as_float(atomicExch(address_ui, __float_as_uint(val))); //write minimum and read back value
    } //if the new value is < the minimum I wrote I need to try again.
  }

  void CudaDecoder::ComputeLogLikelihoods(DecodableInterface *decodable) {
    nvtxRangePushA("ComputeLogLikelihoods");

    int32 frame = num_frames_decoded_;

    std::swap(loglikelihoods_h,loglikelihoods_old_h); //double buffering so we don't overwrite loglikelihoods_h before it is copied down
    std::swap(loglikelihoods_d,loglikelihoods_old_d); //double buffer

    //We really only need about 10% of these but finding out which 10% is more expensive then just computing all of them
    //Computing them inline in the next loop leads to lots of redundant computation
    decodable->ComputeLogLikelihoods(loglikelihoods_h,frame,fst_.max_ilabel+1);

    //copying in another stream to overlap transfer with compute
    cudaMemcpyAsync(loglikelihoods_d,loglikelihoods_h,sizeof(BaseFloat)*(fst_.max_ilabel+1),cudaMemcpyHostToDevice, stream_ll);

    cudaEventRecord(event_ll,stream_ll);  //mark log likelihoods are copied down to the device
    cudaStreamWaitEvent(stream_comp,event_ll,0); //ensure logliklihoods_d is updated before consuming

    nvtxRangePop();
  }

  //structs to hold kernel parameters.  Large numbers of parameters can slow down launch latency which matters when we are launching very short kernels
  struct processTokens_params {

    CudaDecoder::TokenVector prev_toks;
    CudaDecoder::TokenVector cur_toks;
    CudaDecoder::TokenAllocator allocator;
    CudaDecoder::CostType *cutoff;

    //never change
    const __restrict__ uint32_t *e_offsets;
    const __restrict__ uint32_t *ne_offsets;
    const __restrict__ int32 *arc_ilabels;
    const __restrict__ int32 *arc_olabels; 
    const __restrict__ BaseFloat *arc_weights;
    const __restrict__ CudaDecoder::StateId *arc_nextstates;
    const __restrict__ BaseFloat *loglikelihoods;
    CudaDecoder::TokenLookupElem *current_tokens_lookup;
    volatile int *token_locks;
    BaseFloat beam;
    volatile int *modified;
    int *pe_idx;
    int *ne_idx;
    int *fb_idx;
    int *barrier;

  };

  //blockDim.x threads per token
  template<int blockDimx, int blockDimy>
  inline DEVICE void findBestCutoff_function(processTokens_params params) {
    typedef CudaDecoder::TokenState TokenState;
    typedef CudaDecoder::Token Token; 
    typedef CudaDecoder::StateId StateId;
    typedef CudaDecoder::CostType CostType;

    int threadIdxy = threadIdx.x / blockDimx;

    auto group = cooperative_groups::tiled_partition<blockDimx>(cooperative_groups::this_thread_block());

    CudaDecoder::CostType local_cutoff = INFINITY;
    int32 size = params.prev_toks.size(); 

    //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
    while(true) { 
      int i;
      if(group.thread_rank()==0) { //thread 0 nominated to get new token
        i=atomicAdd(params.fb_idx,1);      //get token index
      }
      i=group.shfl(i,0);           //broadcast token index
      //i=__shfl_sync(0xffffffff,i,0);
      if(i>=size) break;  //Work complete
      
      TokenState ts = params.prev_toks[i];
      Token * tok = ts.token;
      StateId state = ts.state;

      uint32_t start=params.e_offsets[state], finish=params.e_offsets[state+1];
      
      int32 ilabel, ilabel_next;

      int j=start+group.thread_rank();

      if(j<finish) {
        ilabel_next = params.arc_ilabels[j];
      }
      int nextj;

      for(j;j<finish;j=nextj) {
        nextj = j+blockDimx;
        ilabel = ilabel_next;
        if(nextj<finish) {
          ilabel_next = params.arc_ilabels[nextj];
        }
        
        BaseFloat acoustic_cost = -params.loglikelihoods[ilabel]; //TODO can I prefetch this?
        CostType weight = params.arc_weights[j];
        
        CudaDecoder::CostType total_cost = tok->cost_ + weight + acoustic_cost + params.beam;

        if(total_cost<local_cutoff)
          local_cutoff = total_cost;
      }
    }

    //TODO reduce inside block first?
    if(local_cutoff!=INFINITY) {
      atomicMin(params.cutoff, local_cutoff);
    }
  }

  //blockDim.x threads per token
  template<int blockDimx, int blockDimy>
  inline DEVICE void processEmittingTokens_function(processTokens_params params) {
    typedef CudaDecoder::TokenState TokenState;
    typedef CudaDecoder::Token Token; 
    typedef CudaDecoder::StateId StateId;
    typedef CudaDecoder::CostType CostType;
    typedef CudaDecoder::TokenLookupElem TokenLookupElem; 
    int threadIdxy = threadIdx.x / blockDimx;
    
    auto group = cooperative_groups::tiled_partition<blockDimx>(cooperative_groups::this_thread_block());

    CostType cutoff=*params.cutoff;
    int32 size = params.prev_toks.size();
    //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
    while(true) {
      int i;
      if(group.thread_rank()==0) { //thread 0 nominated to get new token
        i=atomicAdd(params.pe_idx,1);      //get token index
      }
      i=group.shfl(i,0);           //broadcast token index
      //i=__shfl_sync(0xffffffff,i,0);
      if(i>=size) break;

      TokenState ts = params.prev_toks[i];
      Token * tok = ts.token;
      StateId state = ts.state;

      uint32_t start=params.e_offsets[state], finish=params.e_offsets[state+1];
      int32 ilabel, ilabel_next;  //prefetch ilabel since it leads to a dependent load

      int j=start+group.thread_rank();

      if(j<finish) {
        ilabel_next = params.arc_ilabels[j];
      }
      int nextj;

      for(j;j<finish;j=nextj) {
        nextj = j+blockDimx;

        ilabel = ilabel_next;

        if(nextj<finish) {
          ilabel_next = params.arc_ilabels[nextj];
        }
        BaseFloat acoustic_cost = -params.loglikelihoods[ilabel];  //TODO can I prefetch this?  
        BaseFloat weight = params.arc_weights[j];
        StateId nextstate = params.arc_nextstates[j];

        CostType total_cost = tok->cost_ + weight + acoustic_cost;

        if(total_cost<=cutoff) 
        {
          TokenLookupElem lookup_elem;
          load16(&lookup_elem, &params.current_tokens_lookup[nextstate]);
          
          Token *cur_tok = lookup_elem.token;  
          Token next_tok =  Token(acoustic_cost+weight, tok, j);

          //check if token is active or not.  Double check the lock.
          if(lookup_elem.active==0 && atomicCAS(&params.current_tokens_lookup[nextstate].active,0,1)==0) {        //grab sentinal to see who gets to add to cur_toks list
            params.cur_toks.push_back(TokenState(cur_tok,nextstate));                                             //add to cur_toks list
          }

          volatile Token* cur_tokv = reinterpret_cast<volatile Token*>(cur_tok);  //need volatile reads to ensure we don't get cached versions

          while(*cur_tokv < next_tok) {   //check if we need to update
            if(params.token_locks[nextstate]==0 && atomicExch((int*)&params.token_locks[nextstate],1)==0) {       //try and grab lock
              if(*cur_tokv < next_tok) {                                                                          //recheck if we are min
                
                if(sizeof(Token)==16)
                  store16(cur_tok,&next_tok);                                                                       //update token
                else
                  *cur_tok=next_tok;

                __threadfence();                                                                                  //ensure my write is visible to all threads   
              }
              
              atomicExch((int*)&params.token_locks[nextstate],0);                                                 //release lock
              break;                                                                                              //exit loop as our update is done
            }
            __threadfence();                                                                                      //ensure writes cur_tok and token_locks are visible
          } //end while
        } //end total_cost<=cutoff
      } //end arc loop
    } //end token loop
  }
  
    template<int blockDimx, int blockDimy>
  DEVICE __inline__ void processNonEmittingTokens_function(processTokens_params &params, CudaDecoder::CostType cutoff, uint32_t size,  volatile int *modified) {
    typedef CudaDecoder::TokenState TokenState;
    typedef CudaDecoder::Token Token; 
    typedef CudaDecoder::StateId StateId;
    typedef CudaDecoder::CostType CostType;
    typedef CudaDecoder::TokenLookupElem TokenLookupElem; 
    
    auto group = cooperative_groups::tiled_partition<blockDimx>(cooperative_groups::this_thread_block());

    int threadIdxy = threadIdx.x / blockDimx;

    //uses dynamically load balanced loop trips.  Tokens are assigned dynamically instead of statically
    while(true) {
      int i;
      if(group.thread_rank()==0) { //thread 0 nominated to get new token
        i=atomicAdd(params.ne_idx,1);      //get token index
      }
      i=group.shfl(i,0);           //broadcast token index
      //i=__shfl_sync(0xffffffff,i,0);
      if(i>=size) break;
      
      TokenState ts = params.cur_toks[i];
      Token * tok = ts.token;
      StateId state = ts.state;

      uint32_t start=params.ne_offsets[state], finish=params.ne_offsets[state+1];
      for(int j=start+group.thread_rank();j<finish;j+=blockDimx) {
        BaseFloat weight = params.arc_weights[j];
        StateId nextstate = params.arc_nextstates[j];

        Token next_tok = Token(weight, tok, j);

        if (next_tok.cost_ <= cutoff) {
          TokenLookupElem lookup_elem;
          load16(&lookup_elem,&params.current_tokens_lookup[nextstate]);
          Token *cur_tok = lookup_elem.token;
          
          //check if token is active or not.  If not then add it to the cur_toks list.  Double check the lock.
          if(lookup_elem.active==0 && atomicCAS(&params.current_tokens_lookup[nextstate].active,0,1)==0) {
            params.cur_toks.push_back(TokenState(cur_tok,nextstate));
          }

          volatile Token* cur_tokv = reinterpret_cast<volatile Token*>(cur_tok);  //need volatile reads to ensure we don't get cached versions

          while(*cur_tokv < next_tok) {   //check if we need to update
            if(params.token_locks[nextstate]==0 && atomicExch((int*)&params.token_locks[nextstate],1)==0) {  //try and grab locks
              if(*cur_tokv < next_tok) {                                                                     //recheck that we are minimum
                if(sizeof(Token)==16)
                  store16(cur_tok,&next_tok);                                                                       //update token
                else
                  *cur_tok=next_tok;

                __threadfence();                                                                             //ensure my write is visible to all threads
              }
              
              atomicExch((int*)&params.token_locks[nextstate],0);                                            //release lock
              
              (*modified) = true;                                                                            //mark as updated
              break;  //exit loop as our update is done
            }
            __threadfence(); //ensure writes to cur_tok and token_locks are visible
          } //end try update loop
        }
      }
    }
  }

  //Loop through all tokens repeatdly updating costs until nothing changes
  //__launch_bounds__(64,32)
  __global__ void processNonEmittingTokens_cg(processTokens_params params) {

    //auto grid = cooperative_groups::this_grid();
    //double buffer to reduce synchronization
    volatile int *modified0 = params.modified;    //modified flag for current iteration
    volatile int *modified1 = params.modified+1;  //modified flag for next/last iteration
    *modified1 = false;

    CudaDecoder::CostType cutoff=*params.cutoff;
    do {

      uint32_t size = params.cur_toks.size();

      *params.ne_idx=0;
      //grid.sync();  
      __grid_sync_nv_internal(params.barrier);

      //swap buffers
      swap(modified0,modified1);

      *modified1 = false;

      processNonEmittingTokens_function<32,2>(params,cutoff,size,modified0);

      //grid.sync();
      __grid_sync_nv_internal(params.barrier);

    } while ((*modified0)==true);
    
    //prepare for next iteration
    *params.cutoff = INFINITY;

    allocateNewTokens_function(params.current_tokens_lookup, params.cur_toks, params.allocator);
    __grid_sync_nv_internal(params.barrier);
    if(threadIdx.x==0 && blockIdx.x==0)
      params.allocator.advanceFront(params.cur_toks.size());
  }

  __launch_bounds__(64,32)
  __global__ void processTokens_cg(processTokens_params params) {
//    auto grid = cooperative_groups::this_grid();


    findBestCutoff_function<32,2>(params);
    //grid.sync();
    __grid_sync_nv_internal(params.barrier);
    
    
    volatile int *modified0 = params.modified;    //modified flag for current iteration
    volatile int *modified1 = params.modified+1;  //modified flag for next/last iteration
    *modified1 = false;
    CudaDecoder::CostType cutoff=*params.cutoff;

    processEmittingTokens_function<32,2>(params);
    //grid.sync();
    __grid_sync_nv_internal(params.barrier);  //ensure cur_toks size is final
    

    do {

      uint32_t size = params.cur_toks.size();

      *params.ne_idx=0;

      //grid.sync();  
      __grid_sync_nv_internal(params.barrier); //wait for everyone to read size and modified0

      //swap buffers
      swap(modified0,modified1); //double buffered to avoid extra sync when resetting modified to false

      *modified1 = false;

      processNonEmittingTokens_function<4,16>(params,cutoff,size,modified0);

      //grid.sync();
      __grid_sync_nv_internal(params.barrier);  //wait for everyone to finish process tokens and writes modified0

    } while ((*modified0)==true);


    allocateNewTokens_function(params.current_tokens_lookup, params.cur_toks, params.allocator);
  
    bool rank0 = blockIdx.x==0 && threadIdx.x==0;
    if(rank0) {
      //prepare for next iteration
      params.prev_toks.clear();
      *params.cutoff = INFINITY;
      *params.fb_idx=0;  
      *params.pe_idx=0;
    }
    
    __grid_sync_nv_internal(params.barrier);  //wait for allocation to finish
    
    if(rank0) {
      params.allocator.advanceFront(params.cur_toks.size());
    }

    
  }

  void CudaDecoder::ProcessNonemitting() {
    nvtxRangePushA("ProcessNonemitting");
    // Processes nonemitting arcs for one frame.  Propagates within
    // cur_toks_.

    dim3 threads(64,1);

    dim3 blocks(DIV_ROUND_UP(total_threads,(threads.x*threads.y)));

    processTokens_params params;

    params.modified=modified_d;
    params.cutoff=cutoff_d;
    params.cur_toks=cur_toks_;
    params.ne_offsets=fst_.ne_offsets_d;
    params.arc_weights=fst_.arc_weights_d;
    params.arc_nextstates=fst_.arc_nextstates_d;
    params.current_tokens_lookup=current_tokens_lookup_d;
    params.token_locks=token_locks_d;
    params.allocator=allocator;
    params.ne_idx=ne_idx_d;
    params.barrier=barrier_d;

#if 0
    void *args[] = { (void*) &params };

    cudaLaunchCooperativeKernel((void*)processNonEmittingTokens_cg, blocks, threads, args, 0, stream_comp);
#else
    processNonEmittingTokens_cg<<<blocks,threads,0,stream_comp>>>(params);
#endif

    cudaCheckError();
    nvtxRangePop();
  }

  void CudaDecoder::ProcessTokens() {
    nvtxRangePushA("ProcessTokens");

    processTokens_params params;
   
    dim3 threads(64,1);
    dim3 blocks(DIV_ROUND_UP(total_threads,(threads.x*threads.y)));


    params.prev_toks=prev_toks_;
    params.cur_toks=cur_toks_;
    params.allocator=allocator;
    params.e_offsets=fst_.e_offsets_d;
    params.ne_offsets=fst_.ne_offsets_d;
    params.arc_ilabels=fst_.arc_ilabels_d;
    params.arc_weights=fst_.arc_weights_d;
    params.arc_nextstates=fst_.arc_nextstates_d;
    params.cutoff=cutoff_d;
    params.loglikelihoods=loglikelihoods_d;
    params.current_tokens_lookup=current_tokens_lookup_d;
    params.token_locks=token_locks_d;
    params.modified=modified_d;
    params.beam=beam_;
    params.pe_idx=pe_idx_d;
    params.ne_idx=ne_idx_d;
    params.fb_idx=fb_idx_d;
    params.barrier=barrier_d;

    cudaStreamWaitEvent(stream_comp,event_ll,0); //make sure log likelihoods are on the device before starting these kernels

#if 0
    void *args[] = { (void*) &params };
    cudaLaunchCooperativeKernel((void*)processTokens_cg, blocks, threads, args, 0, stream_comp);
#else
    processTokens_cg<<<blocks,threads,0,stream_comp>>>(params);  //doesn't work
#endif
    cudaCheckError();
      
    cudaEventSynchronize(event_pt); //throttle
    cudaEventRecord(event_pt,stream_comp);

    num_frames_decoded_++;

    nvtxRangePop();
  }

  void CudaDecoder::ClearToks(TokenVector &toks) {
    //cannot acctually delete tokens as they may still be connected to active tokens
    toks.clear(stream_comp);
  }

} // end namespace kaldi.
