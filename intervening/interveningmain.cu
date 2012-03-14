#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cutil.h>
#include <assert.h>
#include "intervening.h"
#include "stencilMVM.h"

float* loadArray(char* filename, int& width, int& height) {
  FILE* fp;
  fp = fopen(filename, "r");
  int dim;
  fread(&dim, sizeof(int), 1, fp);
  assert(dim == 2);
  fread(&width, sizeof(int), 1, fp);
  fread(&height, sizeof(int), 1, fp);
  float* buffer = (float*)malloc(sizeof(float) * width * height);
  int counter = 0;
  for(int col = 0; col < width; col++) {
    for(int row = 0; row < height; row++) {
      float element;
      fread(&element, sizeof(float), 1, fp);
      counter++;
      buffer[row * width + col] = element;
    }
  }
 /*  for(int row = 0; row < height; row++) { */
/*     for(int col = 0; col < width; col++) { */
/*       printf("%f ", buffer[row*width + col]); */
/*     } */
/*     printf("\n"); */
/*   } */
  return buffer;
}

int main(int argc, char** argv) {
  chooseLargestGPU(true);
  char* filename = "smallMpb.dat";
  //char* filename = "mPb.dat";
  //char* filename = "medium.dat";
  
  int radius = 5;
  int width;
  int height;
  float* hostMPb = loadArray(filename, width, height);
  assert(hostMPb != NULL);
  int nPixels = width * height;
  printf("width: %i, height: %i, nPixels: %i\n", width, height, nPixels);
  float* devMPb;
  cudaMalloc((void**)&devMPb, sizeof(float) * nPixels);
  cudaMemcpy(devMPb, hostMPb, sizeof(float) * nPixels, cudaMemcpyHostToDevice);

  int matrixPitchInFloats = findPitchInFloats(nPixels);
  int devMatrixPitch = matrixPitchInFloats * sizeof(float);
  Stencil theStencil(radius, width, height, matrixPitchInFloats);
  int nDimension = theStencil.getStencilArea();
  float* hostGoldenMatrix = theStencil.readStencilMatrix("small.sma");
  //float* hostGoldenMatrix = theStencil.readStencilMatrix("medium.sma");
  //float* hostGoldenMatrix = theStencil.readStencilMatrix("polynesia.sma");
  


  
/*   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); */
/*   cudaArray* mPbArray; */
/*   cudaMallocArray(&mPbArray, &channelDesc, width, height); */
/*   cudaBindTextureToArray(mPb, mPbArray); */
/*   cudaMemcpy2DToArray(mPbArray, 0, 0, hostMPb, sizeof(float) * width, sizeof(float) * width, height, cudaMemcpyHostToDevice); */
/*   int matrixPitchInFloats = findPitchInFloats(nPixels); */
  
/*   Stencil theStencil(radius, width, height, matrixPitchInFloats); */
  
  /* int nDimension = theStencil.getStencilArea(); */
/*   int diameter = theStencil.getDiameter(); */
/*   float* devMatrix; */
/*   size_t devMatrixPitch; */
/*   cudaMallocPitch((void**)&devMatrix, &devMatrixPitch, nPixels * sizeof(float), nDimension); */
/*   cudaMemset(devMatrix, 0, devMatrixPitch * nDimension); */
  float* hostMatrix = (float*)malloc(devMatrixPitch * nDimension);
  
  /* int hostDiagonalMap[CONSTSPACE]; */
/*   theStencil.copyDiagonalOffsets(hostDiagonalMap); */
/*   cudaMemcpyToSymbol(constDiagonals, hostDiagonalMap, sizeof(int) * diameter * diameter); */
 /*  printf("Diagonal Map\n"); */
/*   for(int row = 0; row < diameter; row++) { */
/*     for(int col = 0; col < diameter; col++) { */
/*       printf("%2i ", hostDiagonalMap[row * diameter + col]); */
/*     } */
/*     printf("\n"); */
/*   } */
                                           
 /*  dim3 gridDim = dim3((width - 1)/XBLOCK + 1, (height - 1)/YBLOCK + 1); */
/*   dim3 blockDim = dim3(XBLOCK, YBLOCK); */
  //  __global__ void findAffinities(int width, int height, int radius, int diameter, float rsigma, float* devMatrix, int matrixPitchInFloats) {

  /* float* devScratch; */
/*   cudaMalloc((void**)&devScratch, sizeof(float) * 16);  */
  
/*   findAffinities<<<gridDim, blockDim>>>(width, height, radius, diameter, 1.0f/0.1f, devMatrix, matrixPitchInFloats, devScratch); */
/*   float* hostScratch = (float*)malloc(sizeof(float) * 16); */
/*   cudaMemcpy(hostScratch, devScratch, sizeof(float) * 16, cudaMemcpyDeviceToHost); */
/* /\*   printf("Maxpbs: "); *\/ */
/* /\*   for(int i = 0; i < 5; i++) { *\/ */
/* /\*     printf("(%2.0f, %2.0f): %f ", hostScratch[3*i], hostScratch[3*i+1], hostScratch[3*i+2]); *\/ */
/* /\*   } *\/ */
/* /\*   printf("\n"); *\/ */
/*   printf("Entries: "); */
/*   for(int i = 0; i < 5; i++) { */
/*     printf("%f ", hostScratch[i]); */
/*   } */
/*   printf("\n"); */

  
  /* CUDA_SAFE_CALL(cudaMemcpy(hostMatrix, devMatrix, devMatrixPitch * nDimension, cudaMemcpyDeviceToHost)); */
  /* int row = 3; */
/*   int col = 6; */
/*   printf("row: %i, col %i\n", row, col); */
/*   for(int drow = -radius; drow <= radius; drow++) { */
/*     for(int dcol = -radius; dcol <= radius; dcol++) { */
/*       if (drow*drow + dcol*dcol <= radius * radius) { */
/*         int dimension = hostDiagonalMap[(drow + radius) * diameter + (dcol + radius)]; */
       
/*         printf("%1.7f ", hostMatrix[matrixPitchInFloats * dimension + row * width + col]); */
/*       } else { */
/*         printf("          "); */
/*       } */
/*     } */
/*     printf("\n"); */
          
/*   } */
/*   printf("\n");   */

/*   row = 4; */
/*   col = 4; */
/*   printf("row: %i, col %i\n", row, col); */
/*   for(int drow = -radius; drow <= radius; drow++) { */
/*     for(int dcol = -radius; dcol <= radius; dcol++) { */
/*       if (drow*drow + dcol*dcol <= radius * radius) { */
/*         int dimension = hostDiagonalMap[(drow + radius) * diameter + (dcol + radius)]; */
       
/*         printf("%1.7f ", hostMatrix[matrixPitchInFloats * dimension + row * width + col]); */
/*       } else { */
/*         printf("          "); */
/*       } */
/*     } */
/*     printf("\n"); */
          
/*   } */
/*   printf("\n");  */
  

  /* symmetrizeMatrix<<<gridDim, blockDim>>>(width, height, radius, diameter, nDimension, devMatrix, matrixPitchInFloats); */
  
/*   shapeMatrix<<<gridDim, blockDim>>>(width, height, radius, diameter, nDimension, devMatrix, matrixPitchInFloats); */
 /*  float* devScratch; */
/*   cudaMalloc((void**)&devScratch, 16 * sizeof(float)); */
/*   cudaMemset(devScratch, 0, 16 * sizeof(float)); */
  float* devMatrix;
  intervene(theStencil, devMPb, &devMatrix);


  /* float* hostScratch = (float*)malloc(sizeof(float) * 16);  */
/*   cudaMemcpy(hostScratch, devScratch, sizeof(float) * 16, cudaMemcpyDeviceToHost); */
/* /\*   printf("Maxpbs: "); *\/ */
/* /\*   for(int i = 0; i < 5; i++) { *\/ */
/* /\*     printf("(%2.0f, %2.0f): %f ", hostScratch[3*i], hostScratch[3*i+1], hostScratch[3*i+2]); *\/ */
/* /\*   } *\/ */
/* /\*   printf("\n"); *\/ */
/*   printf("Entries: "); */
/*   for(int i = 0; i < 5; i++) { */
/*     printf("%f ", hostScratch[i]); */
/*   } */
/*   printf("\n"); */
  
  CUDA_SAFE_CALL(cudaMemcpy(hostMatrix, devMatrix, devMatrixPitch * nDimension, cudaMemcpyDeviceToHost));
/*   row = 3; */
/*   col = 6; */
/*   printf("row: %i, col %i\n", row, col); */
/*   for(int drow = -radius; drow <= radius; drow++) { */
/*     for(int dcol = -radius; dcol <= radius; dcol++) { */
/*       if (drow*drow + dcol*dcol <= radius * radius) { */
/*         int dimension = hostDiagonalMap[(drow + radius) * diameter + (dcol + radius)]; */
       
/*         printf("%1.7f ", hostMatrix[matrixPitchInFloats * dimension + row * width + col]); */
/*       } else { */
/*         printf("          "); */
/*       } */
/*     } */
/*     printf("\n"); */
          
/*   } */
/*   printf("\n"); */
/*   for(int drow = -radius; drow <= radius; drow++) { */
/*     for(int dcol = -radius; dcol <= radius; dcol++) { */
/*       if (drow*drow + dcol*dcol <= radius * radius) { */
/*         int dimension = hostDiagonalMap[(drow + radius) * diameter + (dcol + radius)]; */
       
/*         printf("%1.7f ", hostGoldenMatrix[matrixPitchInFloats * dimension + row * width + col]); */
/*       } else { */
/*         printf("          "); */
/*       } */
/*     } */
/*     printf("\n"); */
          
/*   } */
  bool good = true;
  int errors = 0;
  int truths = 0;
  for(int diagonal = 0; diagonal < nDimension; diagonal++) {
    float* gpuPointer = &hostMatrix[matrixPitchInFloats * diagonal];
    float* cpuPointer = &hostGoldenMatrix[matrixPitchInFloats * diagonal];
    for(int row = 0; row < height; row++) {
      for(int col = 0; col < width; col++) {
        if (fabs(*gpuPointer - *cpuPointer) > 0.0001) {
          good = false;
          printf("%i diagonal, (%i, %i) entry: should be %f, is %f\n", diagonal, col, row, *cpuPointer, *gpuPointer);
          errors++;
        } else {
          truths++;
        }
        cpuPointer++;
        gpuPointer++;
      }
    }
  }
  printf("%s\n", good ? "MATRIX CHECKS OUT" : "oops");
  printf("%i correct entries, %i errors\n", truths, errors);

  /* for(int diagonal = 0; diagonal < nDimension; diagonal++) { */
/*     float* gpuPointer = &hostMatrix[matrixPitchInFloats * diagonal]; */
/*     for (int entry = 0; entry < nPixels; entry++) { */
/*       printf("%1.4f ", *gpuPointer); */
/*       gpuPointer++; */
/*     } */
/*     printf("\n"); */
/*   } */

/*   printf("\n"); */
/*   for(int diagonal = 0; diagonal < nDimension; diagonal++) { */
/*     float* cpuPointer = &hostGoldenMatrix[matrixPitchInFloats * diagonal]; */
/*     for (int entry = 0; entry < nPixels; entry++) { */
/*       printf("%1.4f ", *cpuPointer); */
/*       cpuPointer++; */
/*     } */
/*     printf("\n"); */
/*   } */
  
  /* for(int i = width; i < 2*width; i++) { */
/*     printf("%f ", hostGoldenMatrix[i]); */
/*   } */
}
