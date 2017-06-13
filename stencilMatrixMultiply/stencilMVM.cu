// vim: ts=4 syntax=cpp comments=

#include <helper_cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <getopt.h>
#include <vector>
#include <algorithm>

#include "Stencil.h"
#include "stencilMVM.h"

__constant__ int constOffsets[STENCILAREAMAX];

texture<float, 1> texVector;

void bindTexture(float* devVector, int nPixels) {
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  size_t offset = 0;
  cudaBindTexture(&offset, &texVector, devVector, &channelDesc, nPixels * sizeof(float));
}

 class Point { 
  public: 
  Point(int rowIn = 0, int colIn = 0, double valueIn = 0) { 
    row = rowIn; 
    col = colIn; 
    value = valueIn; 
  } 
  int row; 
  int col; 
  double value;

  }; 

bool lesscol(Point p1, Point p2)
{
    return (p1.col < p2.col);
}

int findNDimUnroll(int nDimension) {
  return (nDimension / UNROLL) * UNROLL;
}

__global__ void stencilSumRows(int width, int height, int nPixels, int nDimension, int nDimensionUnroll, float* devMatrix, int matrixPitchInFloats, float* devSum, float* devRSqrtSum) {
  int row = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  int col = row;
  float accumulant = 0;
  int matrixOffset = row;
  int currentDimension = 0;
 
  if (row < nPixels) {
    
    while (currentDimension < nDimensionUnroll) {
      col += constOffsets[currentDimension];
      if ((col >= 0) && (col < nPixels)) {
        float matrixEntry = devMatrix[matrixOffset];
        accumulant += matrixEntry;
      }
      currentDimension++;
      matrixOffset += matrixPitchInFloats;

      col += constOffsets[currentDimension];
      if ((col >= 0) && (col < nPixels)) {
        float matrixEntry = devMatrix[matrixOffset];
        accumulant += matrixEntry;
      }
      currentDimension++;
      matrixOffset += matrixPitchInFloats;

      col += constOffsets[currentDimension];
      if ((col >= 0) && (col < nPixels)) {
        float matrixEntry = devMatrix[matrixOffset];
        accumulant += matrixEntry;
      }
      currentDimension++;
      matrixOffset += matrixPitchInFloats;
      
    }

    while (currentDimension < nDimension) {
      col += constOffsets[currentDimension];
      
      if ((col >= 0) && (col < nPixels)) {
        float matrixEntry = devMatrix[matrixOffset];
        accumulant += matrixEntry;
      }
      currentDimension++;
      matrixOffset += matrixPitchInFloats;
    }
    
    devSum[row] = accumulant;
    devRSqrtSum[row] = rsqrt(accumulant);
  }
}

__global__ void unGeneralizeMatrix(int width, int height, int nPixels, int nDimension, float* devMatrix, int matrixPitchInFloats, float* devSum, float* devRSqrtSum) {
  int row = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  int col = row;
  int matrixOffset = row;
  int currentDimension = 0;
 
  if (row < nPixels) {
    float dii = devSum[row];
    float diim5 = devRSqrtSum[row];
    while (currentDimension < nDimension) {
      col += constOffsets[currentDimension];
      
      if ((col >= 0) && (col < nPixels)) {
        float matrixEntry = devMatrix[matrixOffset];
        if (col == row) { 
          //(Dii - Wii)/Dii
          devMatrix[matrixOffset] = (dii - matrixEntry)/dii;
        } else {
          float djjm5 = devRSqrtSum[col];
          //0-Wij/(sqrt(Dii) * sqrt(Djj))
          devMatrix[matrixOffset] = -matrixEntry * diim5 * djjm5;
        }
      }
      currentDimension++;
      matrixOffset += matrixPitchInFloats;
    }
  }
}

__global__ void stencilMVM(int width, int height, int nPixels, int nDimension, int nDimensionUnroll, float* devMatrix, int matrixPitchInFloats, float* devResult) {
  int row = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  int col = row;
  float accumulant = 0;
  int matrixOffset = row;
  int currentDimension = 0;
 
  if (row < nPixels) {
    
    while (currentDimension < nDimensionUnroll) {
      col += constOffsets[currentDimension];
      if ((col >= 0) && (col < nPixels)) {
        float matrixEntry = devMatrix[matrixOffset];
        float vectorEntry = tex1Dfetch(texVector, col);
        accumulant += matrixEntry * vectorEntry;
      }
      currentDimension++;
      matrixOffset += matrixPitchInFloats;

      col += constOffsets[currentDimension];
      if ((col >= 0) && (col < nPixels)) {
        float matrixEntry = devMatrix[matrixOffset];
        float vectorEntry = tex1Dfetch(texVector, col);
        accumulant += matrixEntry * vectorEntry;
      }
      currentDimension++;
      matrixOffset += matrixPitchInFloats;

      col += constOffsets[currentDimension];
      if ((col >= 0) && (col < nPixels)) {
        float matrixEntry = devMatrix[matrixOffset];
        float vectorEntry = tex1Dfetch(texVector, col);
        accumulant += matrixEntry * vectorEntry;
      }
      currentDimension++;
      matrixOffset += matrixPitchInFloats;
      
    }

    while (currentDimension < nDimension) {
      col += constOffsets[currentDimension];
      
      if ((col >= 0) && (col < nPixels)) {
        float matrixEntry = devMatrix[matrixOffset];
        float vectorEntry = tex1Dfetch(texVector, col);
        accumulant += matrixEntry * vectorEntry;
      }
      currentDimension++;
      matrixOffset += matrixPitchInFloats;
    }
    
    devResult[row] = accumulant;
  }
}

__global__ void scaleEigByD(int width, int height, float* devRSqrtSum, float* p_dEigVectors, int p_nEigNum)
{
	int y;
	int x;
	y = IMUL(blockIdx.y, YBLOCK) + threadIdx.y;
	x = IMUL(blockIdx.x, XBLOCK) + threadIdx.x;

	int nDimension = width * height;
	for (int i = 0; i < p_nEigNum; i++)
	{
		if ((x < width) && (y < height))
		{
			int index = IMUL(width, y) + x;
			p_dEigVectors[i*nDimension + index] *= devRSqrtSum[index];
		}
	}
}


__global__ void generalizeVectors(int nPixels, int nVectors, float* devVectors, int devVectorFloatPitch, float* devRSqrtSum) {
  int row = IMUL(blockIdx.x, XBLOCK) + threadIdx.x;
  if (row < nPixels) {
    float diim5 = devRSqrtSum[row];
    int matrixOffset = row;
    for (int i = 0; i < nVectors; i++) {
      float currentEntry = devVectors[matrixOffset];
      devVectors[matrixOffset] = currentEntry * diim5;
      matrixOffset += devVectorFloatPitch;
    }
  }
}


int findPitchInFloats(int width) {
  float* test;
  size_t pitch;
  checkCudaErrors(cudaMallocPitch((void**)&test, &pitch, width * sizeof(float), 1));
  cudaFree(test);
  return pitch/sizeof(float);
}

float* convertMatrix(Stencil* theStencil, dim3 gridDim, dim3 blockDim, int nDimension, float* devMatrix) {
  int width = theStencil->getWidth();
  int height = theStencil->getHeight();
  int nPixels = width * height;
  int matrixPitchInFloats = theStencil->getMatrixPitchInFloats();
  
  int hostConstOffsets[STENCILAREAMAX];

  theStencil->copyOffsets(hostConstOffsets);
  /* printf("Offsets:\n"); */
/*   for(int i = 0; i < theStencil->getStencilArea(); i++) { */
/*     printf("%i\n", hostConstOffsets[i]); */
/*   } */
  
/* /\*   printf("\nMatrixOffsets: "); *\/ */
/* /\*   for(int i = 0; i < theStencil->getStencilArea(); i++) { *\/ */
/* /\*     printf("%i ", hostConstMatrixOffsets[i]); *\/ */
/* /\*   } *\/ */
/* /\*   printf("\n"); *\/ */

  cudaMemcpyToSymbol(constOffsets, hostConstOffsets, sizeof(hostConstOffsets));

  float* devSum;
  checkCudaErrors(cudaMalloc((void**)&devSum, width * sizeof(float) * height));
  float* devRSqrtSum;
  checkCudaErrors(cudaMalloc((void**)&devRSqrtSum, width * sizeof(float) * height));

  
  int nDimUnroll = findNDimUnroll(nDimension);
  stencilSumRows<<<gridDim, blockDim>>>(width, height, nPixels, nDimension, nDimUnroll, devMatrix, matrixPitchInFloats, devSum, devRSqrtSum);

/*   /\* float* hostSum = (float*)malloc(width * height * sizeof(float)); *\/ */
/* /\*   memset(hostSum, 0, width * height * sizeof(float)); *\/ */

/* /\*   checkCudaErrors(cudaMemcpy(hostSum, devSum, width * sizeof(float) * height, cudaMemcpyDeviceToHost)); *\/ */
/* /\*   for(int row = 0; row < height; row++) { *\/ */
/* /\*     for (int col = 0; col < width; col++) { *\/ */
/* /\*       printf("%f ", hostSum[row * width + col]); *\/ */
/* /\*     } *\/ */
/* /\*     printf("\n"); *\/ */
/* /\*   } *\/ */
  

  
  unGeneralizeMatrix<<<gridDim, blockDim>>>(width, height, nPixels, nDimension, devMatrix, matrixPitchInFloats, devSum, devRSqrtSum);

/* insert stuff here  
  float* hostMatrix = new float[matrixPitchInFloats*nDimension];
  checkCudaErrors(cudaMemcpy(hostMatrix, devMatrix, matrixPitchInFloats * sizeof(float) * nDimension, cudaMemcpyDeviceToHost)); 
 
   printf("Writing matrix to file ...\n");
   FILE* fpw = fopen("ungeneralized.sma", "w"); 
   fwrite(&nPixels, sizeof(int), 1, fpw); 

   std::vector<Point> nonzeros;
   
//   for(int diag = 0; diag < nDimension; diag++) { 
//     int offset = hostConstOffsets[diag];
//    
//     for(int row = 0; row < height; row++) { 
//       for(int col = 0; col < width; col++) { 
//         int currentX = col + xOffset; 
//         int currentY = row + yOffset; 
//         if ((currentX >= 0) && (currentX < width) && 
//             (currentY >= 0) && (currentY < height)) { 
//           int currentRow = row * width + col; 
//           int currentCol = currentY * width + currentX; 
//           float entry = hostMatrix[diag * matrixPitchInFloats + currentY * widthPitchInFloats + currentX]; 
//           nonzeros.push_back(Point(currentRow, currentCol, entry)); 
//         } 
//       } 
//     }     
//   } 
   int eltstart=0;
   int eltsend=0;
   for(int row=0;row<width*height;row++)
   {
   int currentDimension = 0;
   int col=row;
   int matrixOffset = row;
    while (currentDimension < nDimension) {
      col += hostConstOffsets[currentDimension];
      
      if ((col >= 0) && (col < nPixels)) {
        float matrixEntry = hostMatrix[matrixOffset];
        //float vectorEntry = tex1Dfetch(texVector, col);
        //accumulant += matrixEntry * vectorEntry;
        nonzeros.push_back(Point(row,col,matrixEntry));
        eltsend++;
      }
      currentDimension++;
      matrixOffset += matrixPitchInFloats;
    }

    //sort(nonzeros.begin()+eltstart, nonzeros.begin()+eltsend+1, lesscol);
    eltstart = eltsend;
    if((row+1) % 200000 == 0)printf("Processed row : %d\n", row+1);
   }

   
  
   int nnz = nonzeros.size(); 
   printf("%i nonzeros found\n", nnz); 
   fwrite(&nnz, sizeof(int), 1, fpw); 
   int* n = new int[width * height]; 
   int* cols = new int[nnz]; 
   double* vals = new double[nnz]; 

   memset(n, 0, sizeof(int) * width * height); 
  
   int* colPtr = cols; 
   double* valPtr = vals; 
   int start = 0;
   int totalnnz = 0;
   for(int index = 0; index < width * height; index++) {
  
     int i;
     for(i = start; i < nnz && nonzeros[i].row == index; i++) { 
       Point currentPoint = nonzeros[i]; 
       if (index == currentPoint.row) { 
         //printf("Row: %i, Col: %i, Val: %.2f\n", index, currentPoint.col, currentPoint.value);  
         n[index]++; 
         *colPtr = currentPoint.col; 
         //if(i>0) if(*colPtr < nonzeros[i-1].col) printf("%d %d, %d %d\n", nonzeros[i].row,nonzeros[i].col, nonzeros[i-1].row, nonzeros[i-1].col); 
         //if(i>0) assert(*colPtr > nonzeros[i-1].col);

         *valPtr = currentPoint.value; 
         colPtr++; 
         valPtr++;
         totalnnz ++;
       } 
     }
    
     //if(i!=start) printf("row %d : elts = %d\n", index, i-start);
     start = i;
    if((index+1) % 200000 == 0)printf("Processed index : %d\n", index+1);
   }
   if(nnz != totalnnz) printf("init nnz = %d next nnz = %d\n", nnz, totalnnz);

   assert(totalnnz == nnz);


  
   int current = 0;
   for (int row = 0; row < nPixels; row++) { 
     int nz = n[row]; 
     fwrite(&nz, sizeof(int), 1, fpw); 
     fwrite(&vals[current], sizeof(double), nz, fpw); 
     fwrite(&cols[current], sizeof(int), nz, fpw); 
     current = current + nz; 
   } 
   fclose(fpw); 
   printf("%i nonzeros found\n", nnz); 
   delete [] hostMatrix;
   delete [] n;
   delete [] cols;
   delete [] vals;

insert ends */

  cudaFree(devSum);
  return devRSqrtSum;
}





/* class Point { */
/*  public: */
/*  Point(int rowIn = 0, int colIn = 0, double valueIn = 0) { */
/*    row = rowIn; */
/*    col = colIn; */
/*    value = valueIn; */
/*  } */
/*  int row; */
/*  int col; */
/*  double value; */
/* }; */


 /*  float* hostSum = (float*)malloc(width * height * sizeof(float)); */
/*   memset(hostSum, 0, width * height * sizeof(float)); */
  /* float* devSum; */
/*   size_t devSumPitch; */
/*   checkCudaErrors(cudaMallocPitch((void**)&devSum, &devSumPitch, width *sizeof(float), height)); */
/*   stencilSumRows<<<gridDim, blockDim>>>(width, height, nPixels, radius, nDimension, devMatrix, widthPitchInFloats, devSum); */

 /*  checkCudaErrors(cudaMemcpy2D(hostSum, width * sizeof(float), devSum, devSumPitch, width*sizeof(float), height, cudaMemcpyDeviceToHost)); */
/*   for(int row = 0; row < height; row++) { */
/*     for (int col = 0; col < width; col++) { */
/*       printf("%f ", hostSum[row * width + col]); */
/*     } */
/*     printf("\n"); */
/*   } */
  
  
//  unGeneralizeMatrix<<<gridDim, blockDim>>>(width, height, nPixels, radius, nDimension, devMatrix, widthPitchInFloats, devSum);

  /* checkCudaErrors(cudaMemcpy(hostMatrix, devMatrix, matrixPitchInFloats * sizeof(float) * nDimension, cudaMemcpyDeviceToHost)); */
  
/*   FILE* fpw = fopen("ungeneralized.sma", "w"); */
/*   fwrite(&nPixels, sizeof(int), 1, fpw); */

/*   std::vector<Point> nonzeros; */
/*   for(int diag = 0; diag < nDimension; diag++) { */
/*     int xOffset = hostConstXOffsets[diag]; */
/*     int yOffset = hostConstYOffsets[diag]; */
/*     for(int row = 0; row < height; row++) { */
/*       for(int col = 0; col < width; col++) { */
/*         int currentX = col + xOffset; */
/*         int currentY = row + yOffset; */
/*         if ((currentX >= 0) && (currentX < width) && */
/*             (currentY >= 0) && (currentY < height)) { */
/*           int currentRow = row * width + col; */
/*           int currentCol = currentY * width + currentX; */
/*           float entry = hostMatrix[diag * matrixPitchInFloats + currentY * widthPitchInFloats + currentX]; */
/*           nonzeros.push_back(Point(currentRow, currentCol, entry)); */
/*         } */
/*       } */
/*     }     */
/*   } */
  

  
  
/*   int nnz = nonzeros.size(); */
/*   fwrite(&nnz, sizeof(int), 1, fpw); */
/*   int* n = new int[width * height]; */
/*   int* cols = new int[nnz]; */
/*   double* vals = new double[nnz]; */

/*   memset(n, 0, sizeof(int) * width * height); */
  
/*   int* colPtr = cols; */
/*   double* valPtr = vals; */
/*   for(int index = 0; index < width * height; index++) { */
/*     for(int i = 0; i < nnz; i++) { */
/*       Point currentPoint = nonzeros[i]; */
/*       if (index == currentPoint.row) { */
/*         //printf("Row: %i, Col: %i, Val: %.2f\n", index, currentPoint.col, currentPoint.value);  */
/*         n[index]++; */
/*         *colPtr = currentPoint.col; */
/*         *valPtr = currentPoint.value; */
/*         colPtr++; */
/*         valPtr++; */
/*       } */
/*     } */
/*   } */

  
/*   int current = 0; */
/*   for (int row = 0; row < nPixels; row++) { */
/*     int nz = n[row]; */
/*     fwrite(&nz, sizeof(int), 1, fpw); */
/*     fwrite(&vals[current], sizeof(double), nz, fpw); */
/*     fwrite(&cols[current], sizeof(int), nz, fpw); */
/*     current = current + nz; */
/*   } */
/*   fclose(fpw); */
/*   printf("%i nonzeros found\n", nnz); */

/*   return 0; */





void chooseLargestGPU(bool verbose) {
  int cudaDeviceCount;
  cudaGetDeviceCount(&cudaDeviceCount);
  int cudaDevice = 0;
  int maxSps = 0;
  struct cudaDeviceProp dp;
  for (int i = 0; i < cudaDeviceCount; i++) {
    cudaGetDeviceProperties(&dp, i);
    if (dp.multiProcessorCount >= maxSps) {
      maxSps = dp.multiProcessorCount;
      cudaDevice = i;
    }
  }
  cudaGetDeviceProperties(&dp, cudaDevice);
  if (verbose) {
    printf("Using cuda device %i: %s\n", cudaDevice, dp.name);
  }
  cudaSetDevice(cudaDevice);
}





