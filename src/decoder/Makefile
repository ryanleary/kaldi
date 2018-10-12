all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

TESTFILES =

OBJFILES = training-graph-compiler.o lattice-simple-decoder.o lattice-faster-decoder.o \
   lattice-faster-online-decoder.o simple-decoder.o faster-decoder.o \
   decoder-wrappers.o grammar-fst.o decodable-matrix.o

ifeq ($(CUDA), true)
    OBJFILES += cuda-decoder.o cuda-decoder-kernels.o cuda-decoder-utils.o
    LDFLAGS += $(CUDA_LDFLAGS)
    LDLIBS += $(CUDA_LDLIBS)
endif

ifeq ($(CUDA), true)
    OBJFILES += cuda-decoder.o cuda-decoder-kernels.o cuda-decoder-utils.o
    LDFLAGS += $(CUDA_LDFLAGS)
    LDLIBS += $(CUDA_LDLIBS)
endif

LIBNAME = kaldi-decoder

ADDLIBS = ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 

# Make sure we have CUDA_ARCH from kaldi.mk,
ifeq ($(CUDA), true)
  ifndef CUDA_ARCH
    $(error CUDA_ARCH is undefined, run 'src/configure')
  endif
endif

# Implicit rule for kernel compilation,
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../ -I$(OPENFSTINC)


# Make sure we have CUDA_ARCH from kaldi.mk,
ifeq ($(CUDA), true)
  ifndef CUDA_ARCH
    $(error CUDA_ARCH is undefined, run 'src/configure')
  endif
endif

# Implicit rule for kernel compilation,
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../ -I$(OPENFSTINC)


include ../makefiles/default_rules.mk
