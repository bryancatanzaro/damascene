// vim: ts=4 syntax=cpp comments=



#include <cuda.h>
#include <helper_image.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <getopt.h>
#include <time.h>
#include <math.h>
#include "cublas.h"
#include <acml.h>
#include <vector>
#include "stencilMVM.h"
#include "lanczos.h"
//#include "spectralPb.h"


int main(int argc, char** argv) {
  chooseLargestGPU(true);

  int width = 321;
  int height = 481;
  int radius = 5;
  char* filename = "polynesia.sma";
  int nMatrixDimension = width * height;

  int getNEigs = 25;
  
  

  dim3 blockDim(XBLOCK, 1);
  dim3 gridDim((width * height - 1)/XBLOCK + 1, 1);
  
  int matrixPitchInFloats = findPitchInFloats(nMatrixDimension);
  Stencil myStencil(radius, width, height, matrixPitchInFloats);

  float* devMatrix;

  printf("Reading matrix from file...\n");
  float* hostMatrix = myStencil.readStencilMatrix(filename);
  printf("Copying matrix to GPU\n");

  
  uint nDimension = myStencil.getStencilArea();
  
  cudaMalloc((void**)&devMatrix, nDimension * nMatrixDimension * sizeof(float));
 
	checkCudaErrors(cudaMemcpy(devMatrix, hostMatrix, nMatrixDimension * nDimension * sizeof(float), cudaMemcpyHostToDevice));
 
  struct timeval start;
  gettimeofday(&start, 0);
 
 
  float* eigenValues;
  float* devEigenVectors = 0;
  float fTolerance = 1e-3;
  generalizedEigensolve(myStencil, devMatrix, matrixPitchInFloats, getNEigs, &eigenValues, &devEigenVectors, fTolerance);

  float* eigenVectors = (float*)malloc(nMatrixDimension*sizeof(float)*getNEigs);
  checkCudaErrors(cudaMemcpy(eigenVectors, devEigenVectors, nMatrixDimension*getNEigs*sizeof(float), cudaMemcpyDeviceToHost));
  
/*   initEigs(getNEigs, nMatrixDimension, &eigenValues, &eigenVectors); */




/*   int nOrthoChoice = 1; */
/*   if (argc > 1) */
/*     nOrthoChoice = atoi(argv[1]); */
/*   lanczos(nMatrixDimension, gridDim, blockDim, &myStencil, devMatrix,   */
          
/*           getNEigs, eigenValues, eigenVectors, nOrthoChoice, devRSqrtSum); */
  struct timeval stop;
  gettimeofday(&stop, 0);
  float solveTime = (float)(stop.tv_sec - start.tv_sec)  + ((float)(stop.tv_usec - start.tv_usec))*1e-6f;
  
/*   NormalizeEigVecs(nMatrixDimension, eigenVectors, getNEigs); */
  printf("Solve time: %f seconds\n", solveTime);
  FILE* fp;
  fp = fopen("eigenVectors.txt", "w");
	//Print out the eigenvectors
  for (int j = 0; j < nMatrixDimension; j++) {
    for (int i = 0; i < getNEigs; i++) {
      fprintf(fp, "%f ", eigenVectors[i*nMatrixDimension+j]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  sdkSavePGM("eigvec1.pgm", eigenVectors+1*nMatrixDimension, width,height);
  
  fp = fopen("eigenValues.txt", "w");
	for (int i = 0; i < getNEigs; i++) {
		fprintf(fp, "%e\n", eigenValues[i]);
	}
	fclose(fp);

  //spectralPb(eigenValues, eigenVectors, width, height, getNEigs);
  //clearEigs(eigenValues, eigenVectors);
  cudaFree(devEigenVectors);
  
}


