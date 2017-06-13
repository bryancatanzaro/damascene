################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= sPb
# Cuda source files (compiled with cudacc)
CUFILES   := spbmain.cu spectralPb.cu
CU_DEPS		:= #spectralPb_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 
LINKCCFILES := filters.cpp 
LINKCUFILES := convert.cu


################################################################################
# Rules and targets

SMVERSIONFLAGS := -arch sm_35
include ../common.mk
