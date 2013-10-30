#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cutil.h>

#define TEXTON32 1
#define TEXTON64 2

#define XBLOCK 32

__constant__ float coefficients[12];
// tg constants are
// 1.3841
// 1.4152
// 1.3756
float hostCoefficients_64[] = {0.0123, 0.0110, 0.0117, 0.0169, 0.0176, 0.0198, 0.0138, 0.0138, 0.0145, 0.0104, 0.0105, 0.0115};
float hostCoefficients[] = {0.0123, 0.0110, 0.0117, 0.0169, 0.0176, 0.0198, 0.0138, 0.0138, 0.0145, 0.0144, 0.0149, 0.0158};

__constant__ float weights[12];
float hostweights_64[] = {0,   0,    0.0028,    0.0041,    0.0042,    0.0047,    0.0033,    0.0033,    0.0035,    0.0025,    0.0025,    0.0137,    0.0139};
float hostweights[] = {0,   0,    0.0028,    0.0041,    0.0042,    0.0047,    0.0033,    0.0033,    0.0035,    0.0035,    0.0035,    0.0188,    0.0139};


__global__ void combine_kernel(int nPixels, int cuePitchInFloats, float* devBg, float* devCga, float* devCgb, float* devTg, float* devMpb, float* devCombinedg) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int orientation = threadIdx.y;
  int orientedIndex = orientation * cuePitchInFloats + index;
  if (index < nPixels) {
    float accumulant = 0.0;
    float accumulant2=0.0;
    float* pointer = &devBg[orientedIndex];
    accumulant += *pointer * coefficients[0];
    accumulant2 += *pointer * weights[0];
    pointer += 8 * cuePitchInFloats;
    accumulant += *pointer * coefficients[1];
    accumulant2 += *pointer * weights[1];
    pointer += 8 * cuePitchInFloats;
    accumulant += *pointer * coefficients[2];
    accumulant2 += *pointer * weights[2];
    pointer = &devCga[orientedIndex];
    accumulant += *pointer * coefficients[3];
    accumulant2 += *pointer * weights[3];
    pointer += 8 * cuePitchInFloats;
    accumulant += *pointer * coefficients[4];
    accumulant2 += *pointer * weights[4];
    pointer += 8 * cuePitchInFloats;
    accumulant += *pointer * coefficients[5];
    accumulant2 += *pointer * weights[5];
    pointer = &devCgb[orientedIndex];
    accumulant += *pointer * coefficients[6];
    accumulant2 += *pointer * weights[6];
    pointer += 8 * cuePitchInFloats;
    accumulant += *pointer * coefficients[7];
    accumulant2 += *pointer * weights[7];
    pointer += 8 * cuePitchInFloats;
    accumulant += *pointer * coefficients[8];
    accumulant2 += *pointer * weights[8];
    pointer = &devTg[orientedIndex];
    accumulant += *pointer * coefficients[9];
    accumulant2 += *pointer * weights[9];
    pointer += 8 * cuePitchInFloats;
    accumulant += *pointer * coefficients[10];
    accumulant2 += *pointer * weights[10];
    pointer += 8 * cuePitchInFloats;
    accumulant += *pointer * coefficients[11];
    accumulant2 += *pointer * weights[11];
    devMpb[orientedIndex] = accumulant;
    devCombinedg[orientedIndex] = accumulant2;
  }
}

void combine(int width, int height, int cuePitchInFloats, float* devBg, float* devCga, float* devCgb, float* devTg, float** p_devMpb, float ** p_devCombinedg, int p_nTextonChoice) {
  int norients = 8;
  cudaMalloc((void**)p_devMpb, sizeof(float) * cuePitchInFloats * norients);
  cudaMalloc((void**)p_devCombinedg, sizeof(float) * cuePitchInFloats * norients);
  float* devMpb = *p_devMpb;
  float* devCombinedg = *p_devCombinedg;
  if (TEXTON32 == p_nTextonChoice)
  {
    cudaMemcpyToSymbol(coefficients, hostCoefficients, sizeof(float) * 12);
    cudaMemcpyToSymbol(weights, hostweights, sizeof(float) * 12);
  }
  else
  {
    cudaMemcpyToSymbol(coefficients, hostCoefficients_64, sizeof(float) * 12);
    cudaMemcpyToSymbol(weights, hostweights_64, sizeof(float) * 12);
  
  }
  int nPixels = width * height;
  dim3 gridDim = dim3((nPixels - 1)/XBLOCK + 1, 1, 1);
  dim3 blockDim = dim3(XBLOCK, 8, 1);
  combine_kernel<<<gridDim, blockDim>>>(nPixels, cuePitchInFloats, devBg, devCga, devCgb, devTg, devMpb, devCombinedg);
}
