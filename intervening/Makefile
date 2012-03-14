################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= intervene
# Cuda source files (compiled with cudacc)
CUFILES   := intervening.cu interveningmain.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 
LINKCCFILES := Stencil.cpp 
LINKCUFILES := stencilMVM.cu #combine.cu


################################################################################
# Rules and targets

SMVERSIONFLAGS := -arch sm_10
include ../common.mk
