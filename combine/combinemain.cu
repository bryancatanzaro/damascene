#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cutil.h>
#include "combine.h"
#include "stencilMVM.h"

void fillArray(float value, float* array, int sizeInFloats) {
  for(int i = 0; i < sizeInFloats; i++) {
    array[i] = value;
  }
}

void writeTextImage(const char* filename, uint width, uint height, float* image) {
  FILE* fp = fopen(filename, "w");
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      fprintf(fp, "%f ", image[row * width + col]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}
int main(int argc, char** argv) {
  chooseLargestGPU(true);
  int width = 2;
  int height = 2;
  int norients = 8;
  float* hostBg = (float*)malloc(sizeof(float) * width * height * norients * 3);
  fillArray(2.0, hostBg, width * height * norients * 3);
  float* hostCga = (float*)malloc(sizeof(float) * width * height * norients * 3);
  fillArray(3.0, hostCga, width * height * norients * 3);
  float* hostCgb = (float*)malloc(sizeof(float) * width * height * norients * 3);
  fillArray(4.0, hostCgb, width * height * norients * 3);
  float* hostTg = (float*)malloc(sizeof(float) * width * height * norients * 3);
  fillArray(5.0, hostTg, width * height * norients * 3);
  size_t cuePitch;
  float* devBg;
  float* devCga;
  float* devCgb;
  float* devTg;
  cudaMallocPitch((void**)&devBg, &cuePitch, sizeof(float) * width * height, norients * 3);
  cudaMallocPitch((void**)&devCga, &cuePitch, sizeof(float) * width * height, norients * 3);
  cudaMallocPitch((void**)&devCgb, &cuePitch, sizeof(float) * width * height, norients * 3);
  cudaMallocPitch((void**)&devTg, &cuePitch, sizeof(float) * width * height, norients * 3);
  
  cudaMemcpy2D(devBg, cuePitch, hostBg, sizeof(float) * width * height, sizeof(float) * width * height, norients * 3, cudaMemcpyHostToDevice);
  cudaMemcpy2D(devCga, cuePitch, hostCga, sizeof(float) * width * height, sizeof(float) * width * height, norients * 3, cudaMemcpyHostToDevice);
  cudaMemcpy2D(devCgb, cuePitch, hostCgb, sizeof(float) * width * height, sizeof(float) * width * height, norients * 3, cudaMemcpyHostToDevice);
  cudaMemcpy2D(devTg, cuePitch, hostTg, sizeof(float) * width * height, sizeof(float) * width * height, norients * 3, cudaMemcpyHostToDevice);
  float* devMpb;
  float* devCombinedGradient;
  int textonChoice = 1;
  combine(width, height, cuePitch/sizeof(float), devBg, devCga, devCgb, devTg, &devMpb, &devCombinedGradient, textonChoice);
  float* hostMpb = (float*)malloc(sizeof(float) * width * height * norients);
  cudaMemcpy2D(hostMpb, sizeof(float) * width * height, devMpb, cuePitch, sizeof(float) * width * height, norients, cudaMemcpyDeviceToHost);
  writeTextImage("mpb.txt", width, height, hostMpb);
  
}
