#include <cuda.h>
#include <cutil.h>
#include <stdio.h>

#define NUM_LUTS 8


void PostProcess(int width, int height, unsigned int matrixPitchInFloats,float* devGPb, float* devMPb, float* devGPb_thin);

void skeletonize(int width, int height, int matrixPitchInFloats, float* devGPb_thin);

void CPU_skeletonize(int width, int height, float* in_image, float* out_image, int* plut);
void NormalizeGpbAll(int p_nPixels, int p_nOrient, int matrixPitchInFloats, float* devGpball);

