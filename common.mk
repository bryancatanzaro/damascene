################################################################################
#
# Common build script
#
################################################################################

.SUFFIXES : .cu .cu_dbg_o .c_dbg_o .cpp_dbg_o .cu_rel_o .c_rel_o .cpp_rel_o .cubin

CUDA_INSTALL_PATH ?= /usr/local/cuda

ifdef cuda-install
	CUDA_INSTALL_PATH := $(cuda-install)
endif



CUDA_SDK_PATH ?= /usr/local/cuda/samples

# Basic directory setup for SDK
# (override directories only if they are not already defined)
SRCDIR     ?= 
ROOTDIR    ?= ..
ROOTBINDIR ?= $(ROOTDIR)/bin
BINDIR     ?= $(ROOTBINDIR)/linux
ROOTOBJDIR ?= $(ROOTDIR)/obj
ROOTSODIR  ?= $(ROOTDIR)/lib
SODIR      ?= $(ROOTSODIR)/linux

LIBDIR     := $(CUDA_SDK_PATH)/common/lib
COMMONDIR  := $(CUDA_SDK_PATH)/common
ACMLDIR    ?= $(ROOTSODIR)/acml/ifort64

ifdef cuda-2.1
  CUDALIBSUFFIX := lib
  LIBDIR := $(CUDA_SDK_PATH)/lib
  COMMONDIR := $(CUDA_SDK_PATH)/common
else
  CUDALIBSUFFIX := lib64
endif

# Compilers
NVCC       := nvcc 
CXX        := g++
CC         := gcc
LINK       := g++

# Includes
INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/inc -I$(ROOTDIR)/include -I$(ACMLDIR)/include -I../include

# architecture flag for cubin build
CUBIN_ARCH_FLAG := -m64

# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
OPENGLLIB := -lGL -lGLU
ifeq ($(USEGLLIB),1)

	# detect if 32 bit or 64 bit system
	HP_64 =	$(shell uname -m | grep 64)

	ifeq "$(strip $(HP_64))" ""
		OPENGLLIB += -lGLEW
	else
		OPENGLLIB += -lGLEW_x86_64
	endif

	CUBIN_ARCH_FLAG := -m64
endif

# Libs
LIB       := -L$(CUDA_INSTALL_PATH)/$(CUDALIBSUFFIX) -L$(LIBDIR) -L$(COMMONDIR)/lib -L$(ACMLDIR)/lib -L$(ROOTSODIR) -lcuda -lcudart -lcublas -lblas -lacml -L../stencilMatrixMultiply/lib/linux/release ${OPENGLLIB} ${LIB}

# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

# Compiler-specific flags
NVCCFLAGS := -Xptxas -v -maxrregcount 32
CXXFLAGS  := $(CXXWARN_FLAGS)
CFLAGS    := $(CWARN_FLAGS)

# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX

# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g
	NVCCFLAGS   += -D_DEBUG
	BINSUBDIR   := debug
	LIBSUFFIX   := D
else 
	COMMONFLAGS += -O3 
	BINSUBDIR   := release
	LIBSUFFIX   :=
	NVCCFLAGS   += --compiler-options -fno-strict-aliasing
	CXXFLAGS    += -fno-strict-aliasing
	CFLAGS      += -fno-strict-aliasing
endif

ifeq ($(shared),1)
	NVCCFLAGS  += -Xcompiler -fPIC
	CFLAGS += -fPIC
	CXXFLAGS += -fPIC
	LINK += -shared
endif

# append optional arch/SM version flags (such as -arch sm_11)
SMVERSIONFLAGS ?= -arch sm_35
#SMVERSIONFLAGS ?= -arch sm_11
NVCCFLAGS += $(SMVERSIONFLAGS)

# architecture flag for cubin build
CUBIN_ARCH_FLAG := -m32

# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
OPENGLLIB := -lGL -lGLU
ifeq ($(USEGLLIB),1)

	# detect if 32 bit or 64 bit system
	HP_64 =	$(shell uname -m | grep 64)

	ifeq "$(strip $(HP_64))" ""
		OPENGLLIB += -lGLEW
	else
		OPENGLLIB += -lGLEW_x86_64
	endif

	CUBIN_ARCH_FLAG := -m64
endif

ifeq ($(USEPARAMGL),1)
	PARAMGLLIB := -lparamgl$(LIBSUFFIX)
endif

# Libs
LIB       := -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib -lcuda -lcudart ${OPENGLLIB} $(PARAMGLLIB) ${LIB}


# Lib/exe configuration
ifneq ($(STATIC_LIB),)
	TARGETDIR := $(LIBDIR)
	TARGET   := $(subst .a,$(LIBSUFFIX).a,$(LIBDIR)/$(STATIC_LIB))
	LINKLINE  = ar qv $(TARGET) $(OBJS) 
else
	#LIB += -lcutil$(LIBSUFFIX)
	# Device emulation configuration
	ifeq ($(emu), 1)
		NVCCFLAGS   += -deviceemu
		CUDACCFLAGS += 
		BINSUBDIR   := emu$(BINSUBDIR)
		# consistency, makes developing easier
		CXXFLAGS		+= -D__DEVICE_EMULATION__
		CFLAGS			+= -D__DEVICE_EMULATION__
	endif
	ifeq ($(shared),1)
		TARGETDIR := $(SODIR)/$(BINSUBDIR)
		TARGET := $(TARGETDIR)/lib$(EXECUTABLE).so
  else
		TARGETDIR := $(BINDIR)/$(BINSUBDIR)
		TARGET    := $(TARGETDIR)/$(EXECUTABLE)
	endif
	ifndef NOLINK
		LINKLINE  = $(LINK) -o $(TARGET) $(OBJS) $(LINKOBJS) $(LIB)
	else
		LINKLINE  =
	endif
endif

# check if verbose 
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif

################################################################################
# Check for input flags and set compiler flags appropriately
################################################################################
ifeq ($(fastmath), 1)
	NVCCFLAGS += -use_fast_math
endif

ifeq ($(keep), 1)
	NVCCFLAGS += -keep
	NVCC_KEEP_CLEAN := *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx
endif

ifdef maxregisters
	NVCCFLAGS += -maxrregcount $(maxregisters)
endif

###########################################
# Check for atomics support
############################################
ifeq ($(findstring sm_10, $(SMVERSIONFLAGS)),sm_10)
	NVCCFLAGS += -D __NO_ATOMIC 
endif
ifeq ($(findstring sm_11, $(SMVERSIONFLAGS)),sm_11)
	NVCCFLAGS += -D __NO_ATOMIC 
endif



# Add cudacc flags
NVCCFLAGS += $(CUDACCFLAGS)

# Add common flags
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)

ifeq ($(nvcc_warn_verbose),1)
	NVCCFLAGS += $(addprefix --compiler-options ,$(CXXWARN_FLAGS)) 
	NVCCFLAGS += --compiler-options -fno-strict-aliasing
endif

################################################################################
# Set up object files
################################################################################
OBJDIR := $(ROOTOBJDIR)/$(BINSUBDIR)

LINKOBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp_o,$(notdir $(LINKCCFILES)))
LINKOBJS +=  $(patsubst %.c,$(OBJDIR)/%.c_o,$(notdir $(LINKCFILES)))
LINKOBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cu_o,$(notdir $(LINKCUFILES)))

OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp_o,$(notdir $(CCFILES)))
OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c_o,$(notdir $(CFILES)))
OBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cu_o,$(notdir $(CUFILES)))

################################################################################
# Set up cubin files
################################################################################
CUBINDIR := $(SRCDIR)data
CUBINS +=  $(patsubst %.cu,$(CUBINDIR)/%.cubin,$(notdir $(CUBINFILES)))

################################################################################
# Rules
################################################################################
$(OBJDIR)/%.c_o : $(SRCDIR)%.c $(C_DEPS)
	$(VERBOSE)$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.cpp_o : $(SRCDIR)%.cpp $(C_DEPS)
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/%.cu_o : $(SRCDIR)%.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) -o $@ -c $< $(NVCCFLAGS)

$(CUBINDIR)/%.cubin : $(SRCDIR)%.cu cubindirectory
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) -o $@ -cubin $< $(NVCCFLAGS)

$(TARGET): makedirectories $(OBJS) $(CUBINS) Makefile
	$(VERBOSE)$(LINKLINE)

cubindirectory:
	@mkdir -p $(CUBINDIR)

makedirectories:
	@mkdir -p $(LIBDIR)
	@mkdir -p $(OBJDIR)
	@mkdir -p $(TARGETDIR)


tidy :
	@find | egrep "#" | xargs rm -f
	@find | egrep "\~" | xargs rm -f

clean : tidy
	$(VERBOSE)rm -f $(OBJS)
	$(VERBOSE)rm -f $(CUBINS)
	$(VERBOSE)rm -f $(TARGET)
	$(VERBOSE)rm -f $(NVCC_KEEP_CLEAN)

clobber : clean
	rm -rf $(ROOTOBJDIR)
