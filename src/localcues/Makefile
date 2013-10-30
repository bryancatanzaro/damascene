################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= localcues
# Cuda source files (compiled with cudacc)
CUFILES   := gradient.cu parabola.cu localcuesmain.cu localcues.cu rotate.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= #localcues.c
LINKCCFILES := Stencil.cpp filters.cpp 
LINKCUFILES := stencilMVM.cu convert.cu texton.cu kmeans.cu combine.cu 

#CU_DEPS := spec.h

################################################################################
# Rules and targets

verbose := 1

SMVERSIONFLAGS := -arch sm_10
include ../common.mk
