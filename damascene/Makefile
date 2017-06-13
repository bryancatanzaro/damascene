################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= damascene
# CUDA source files (compiled with cudacc)
CUFILES		:= damascene.cu

LINKCUFILES := kmeans.cu texton.cu convert.cu stencilMVM.cu intervening.cu lanczos.cu localcues.cu gradient.cu rotate.cu parabola.cu spectralPb.cu globalPb.cu combine.cu nonmax.cu skeleton.cu


LINKCCFILES := Stencil.cpp filters.cpp


################################################################################
# Rules and targets

SMVERSIONFLAGS := -arch sm_35
include ../common.mk
