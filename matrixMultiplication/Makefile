# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Location of the CUDA Toolkit binaries and libraries
NVCC            := $(shell which nvcc)
CUDA_BIN_PATH   := $(dir $(NVCC))
CUDA_PATH       ?= $(abspath $(CUDA_BIN_PATH)/..)
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

# Common binaries
GCC             ?= g++

# Extra user flags
EXTRA_NVCCFLAGS ?=

# CUDA code generation flags
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
# GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM30)

# OS-specific build flags
ifneq ($(DARWIN),)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      CCFLAGS   := -m32
  else
      CCFLAGS   := -m64
  endif
      LDFLAGS   += -L$(CUDA_LIB_PATH) -lcudart -Wl,-rpath,$(CUDA_LIB_PATH)
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
endif


# Common includes and paths for CUDA
INCLUDES := -I$(CUDA_INC_PATH)

BIN=$(basename $(wildcard *.c*))

all: $(BIN) .gitignore
run: $(BIN)
	./$^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<

%: %.o
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)

clean:
	rm -f $(BIN) *.o

.gitignore : $(wildcard *.c*)
	@for b in $(BIN); do echo $$b; done | sort >$@
