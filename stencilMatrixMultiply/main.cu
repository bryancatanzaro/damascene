#include <cuda.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <getopt.h>
#include <vector>

#include "Stencil.h"
#include "stencilMVM.h"


int main(int argc, char** argv) {
  
  
  chooseLargestGPU(false);
/*   int radius = 5; */
/*   int width = 321; */
/*   int height = 481; */
/*   char* filename = "polynesia.sma"; */
  
  int radius = 1;
  int width = 4;
  int height = 4;
  char* filename = "tiny.sma";
  int nPixels = width * height;
  int matrixPitchInFloats = findPitchInFloats(nPixels);

  Stencil myStencil(radius, width, height, matrixPitchInFloats);
  
  uint nDimension = myStencil.getStencilArea();
 
  dim3 blockDim(XBLOCK, YBLOCK);
  dim3 gridDim((nPixels - 1)/XBLOCK + 1, 1);
  

 
 
  
  printf("Reading matrix from file...\n");
  float* hostMatrix = myStencil.readStencilMatrix(filename);
  
  /* for(int i = 0; i < nDimension; i++) { */
/*     for(int j = 0; j < matrixPitchInFloats; j++) { */
/*       printf("%.4f ", hostMatrix[j + i * matrixPitchInFloats]); */
/*     } */
/*     printf("\n"); */
/*   } */
  //  for(int i = 0; i < nDimension; i++) {
  //printf
  
  printf("Copying matrix to GPU\n");
  float* devMatrix;
  size_t devMatrixPitch;
  cudaMallocPitch((void**)&devMatrix, &devMatrixPitch, nPixels * sizeof(float), nDimension);
  assert((devMatrixPitch/sizeof(float)) == matrixPitchInFloats);
 
	checkCudaErrors(cudaMemcpy(devMatrix, hostMatrix, matrixPitchInFloats * sizeof(float) * nDimension, cudaMemcpyHostToDevice));

  float* hostVector = (float*)malloc(nPixels * sizeof(float));
  for(int i = 0; i < nPixels; i++) {
    hostVector[i] = (float)(i+1)/(float)(nPixels+1);
    //hostVector[i] = i + 1;
  }
 
  float* devVector;
  checkCudaErrors(cudaMalloc((void**)&devVector, nPixels * sizeof(float)));

  bindTexture(devVector, nPixels);
  
  int nDimUnroll = findNDimUnroll(nDimension);
  convertMatrix(&myStencil, gridDim, blockDim, nDimension, devMatrix);
  
  
  
  float* hostResult = (float*)malloc(nPixels * sizeof(float));
  float* devResult;
  checkCudaErrors(cudaMalloc((void**)&devResult, width *sizeof(float) * height));

 
  int iterationMax = 1000;
  printf("Doing %i Matrix Vector Multiplies\n", iterationMax);
  cudaMemcpy(devVector, hostVector, nPixels * sizeof(float), cudaMemcpyHostToDevice);

  
  StopWatchInterface *iterationTimer=NULL;
  sdkCreateTimer(&iterationTimer);
  sdkStartTimer(&iterationTimer);
  int i;

  
  for(i = 0; i < iterationMax; i++) {
    stencilMVM<<<gridDim, blockDim>>>(width, height, nPixels, nDimension, nDimUnroll, devMatrix, matrixPitchInFloats, devResult);

  }
  cudaThreadSynchronize();
  sdkStopTimer(&iterationTimer);
  
  checkCudaErrors(cudaMemcpy(hostResult, devResult, width * sizeof(float) * height, cudaMemcpyDeviceToHost));

  float mulTime = sdkGetTimerValue(&iterationTimer);
  sdkDeleteTimer(&iterationTimer);
  mulTime /= (float)i;
  printf("MVM time: %f microseconds\n", mulTime * 1000);
  printf("Writing result to file...\n");
  FILE* fp;
  fp = fopen("result.txt", "w");
  
  for(int i = 0; i < nPixels; i++) {
    fprintf(fp, "%f\n", hostResult[i]);
  }
  fclose(fp);
/*   if (hostMatrixAlloced) { */
/*     free(hostMatrix); */
/*   } */
  return(0);

}


