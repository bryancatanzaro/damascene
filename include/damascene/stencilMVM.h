#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <getopt.h>
#include <vector>

#include <damascene/Stencil.h>
#include <damascene/util.h>
#define IMUL(a, b) __mul24(a, b)
#define XBLOCK 64
#define YBLOCK 1
#define UNROLL 3



void bindTexture(float* devVector, int nPixels);
void chooseLargestGPU(bool verbose);


__global__ void stencilSumRows(int width, int height, int nPixels, int nDimension, int nDimensionUnroll, float* devMatrix, int matrixPitchInFloats, float* devSum, float* devRSqrtSum);

__global__ void unGeneralizeMatrix(int width, int height, int nPixels, int nDimension, float* devMatrix,int matrixPitchInFloats,  float* devSum, float* devRSqrtSum);

__global__ void stencilMVM(int width, int height, int nPixels, int nDimension, int nDimUnroll, float* devMatrix, int matrixPitchInFloats, float* devResult);

__global__ void generalizeVectors(int nPixels, int nVectors, float* devVectors, int devVectorFloatPitch, float* devRSqrtSum);

__global__ void scaleEigByD(int width, int height, float* devRSqrtSum, float* p_dEigVectors, int p_nEigNum);

int findPitchInFloats(int width);

float* convertMatrix(Stencil* theStencil, dim3 gridDim, dim3 blockDim, int nDimension, float* devMatrix);

int findNDimUnroll(int nDimension);
