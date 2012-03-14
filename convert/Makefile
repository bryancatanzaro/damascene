################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= convert
# Cuda source files (compiled with cudacc)
CUFILES   := convert.cu convertmain.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 
LINKCCFILES := Stencil.cpp
LINKCUFILES := stencilMVM.cu 


################################################################################
# Rules and targets

SMVERSIONFLAGS := -arch sm_10
include ../common.mk
