################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= stencilMVM
# Cuda source files (compiled with cudacc)
CUFILES   := stencilMVM.cu main.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= Stencil.cpp
LINKCCFILES := 
LINKCUFILES := 


################################################################################
# Rules and targets

SMVERSIONFLAGS := -arch sm_10
include ../common.mk
