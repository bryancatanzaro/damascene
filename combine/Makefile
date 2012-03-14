################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= combine
# Cuda source files (compiled with cudacc)
CUFILES   := combine.cu combinemain.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 
LINKCCFILES := Stencil.cpp
LINKCUFILES := stencilMVM.cu 


################################################################################
# Rules and targets

SMVERSIONFLAGS := -arch sm_10
include ../common.mk
