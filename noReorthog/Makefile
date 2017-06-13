################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= lanczos
# Cuda source files (compiled with cudacc)
CUFILES   := lanczosmain.cu lanczos.cu #spectralPb.cu
CU_DEPS		:= #spectralPb_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 
LINKCCFILES := Stencil.cpp
LINKCUFILES := stencilMVM.cu


################################################################################
# Rules and targets

SMVERSIONFLAGS := -arch sm_35
include ../common.mk
