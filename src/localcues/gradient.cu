#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cutil.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include "rotate.h"
#include "convert.h"
#include <cutil.h>
#include <assert.h>


#define uint unsigned int

#define IMUL(a, b) __mul24(a, b)

#define TEXTON32 1
#define TEXTON64 2

int* computeGoldenIntegrals(int width, int height, int nbins, int* inputImage) {
  int* integrals = (int*)malloc(sizeof(int)*width*height*nbins);
  memset(integrals, 0, sizeof(int) * width * height * nbins);
  for(int bin = 0; bin < nbins; bin++) {
    for(int row = 0; row < height; row++) {
      for(int col = 0; col < width; col++) {
        int integralValue = 0;
        if (row == 0) {
          if (col == 0) {
            integralValue = ((inputImage[0] == bin) ? 1 : 0);
          } else {
            integralValue = integrals[(col - 1) * nbins + bin] + ((inputImage[col] == bin) ? 1 : 0);
          }
        } else {
          if (col == 0) {
            integralValue = integrals[((row - 1) * width) * nbins + bin] + ((inputImage[row * width] == bin) ? 1 : 0);
          } else {
            integralValue = integrals[((row - 1) * width + col) * nbins + bin] + integrals[(row * width + col - 1)*nbins + bin] - integrals[((row - 1) * width + col - 1) * nbins + bin] + ((inputImage[row * width + col] == bin) ? 1 : 0);
          }
        }
        integrals[(row * width + col)*nbins + bin] = integralValue;
      }
    }
  }
  return integrals;
}

void checkIntegrals(int width, int height, int nbins, int* goldenIntegrals, int goldenIntegralPitch, int* suspectIntegrals, int suspectIntegralPitch) {
  bool error = false;
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      for(int bin = 0; bin < nbins; bin++) {
        if (goldenIntegrals[(row * width + col) * goldenIntegralPitch + bin] !=
            suspectIntegrals[(row * width + col) * suspectIntegralPitch + bin]) {
          printf("Error at: %d, %d, %d\n", row, col, bin);
          error = true;
        }
      }
    }
  }
  if (!error) {
    printf("Integrals check out!\n");
  }
}

__global__ void integrateBins(int width, int height, int nbins, int* devImage, int binPitch, int* devIntegrals) {
  __shared__ int pixels[16];
  const int blockX = blockDim.y * blockIdx.x;
  const int threadX = threadIdx.y;
  const int bin = threadIdx.x;
  const int x = blockX + threadX;
  if (x >= width) return;
  if (bin > nbins) return;
  int* imagePointer = devImage + x;
  int* outputPointer = devIntegrals + binPitch * x + bin;
  int accumulant = 0;
  for(int y = 0; y < height; y++) {
    if (bin == 0) {
      pixels[threadX] = *imagePointer;
    }
    __syncthreads();
    if (pixels[threadX] == bin) accumulant++;
    *outputPointer = accumulant;
    imagePointer += width;
    outputPointer += width * binPitch;
  }
}

__global__ void integrateBinsT(int width, int height, int nbins, int binPitch, int* devIntegrals) {
  const int blockY = blockDim.y * blockIdx.x;
  const int threadY = threadIdx.y;
  const int bin = threadIdx.x;
  const int y = blockY + threadY;
  if (y >= height) return;
  if (bin >= binPitch) return;
  int* imagePointer = devIntegrals + binPitch * y * width + bin;
  int accumulant = 0;
  for(int x = 0; x < width; x++) {
    accumulant += *imagePointer;
    *imagePointer = accumulant;
    imagePointer += binPitch;
  }
}

/**
 * For a given orientation, computes the integral images for each of the histogram bins
 */
void formIntegralImages(int width, int height, int nbins, int* devImage,
                        int binPitch, int* devIntegrals) {
  int pixelsPerCTA = 4;
  dim3 gridDim  = dim3((width - 1) / pixelsPerCTA + 1);
  dim3 blockDim = dim3(nbins, pixelsPerCTA);

  integrateBins<<<gridDim, blockDim>>>(width, height, nbins, devImage, binPitch, devIntegrals);

  gridDim = dim3((height - 1)/pixelsPerCTA + 1);
  integrateBinsT<<<gridDim, blockDim>>>(width, height, nbins, binPitch, devIntegrals);
}



float* getImage(uint width, uint height, float* devImage) {
  int imageSize = width * height * sizeof(float);
  float* result = (float*)malloc(imageSize);
  CUDA_SAFE_CALL(cudaMemcpy(result, devImage, imageSize, cudaMemcpyDeviceToHost));
  return result;
}

int* getImage(uint width, uint height, int* devImage) {
  int imageSize = width * height * sizeof(int);
  int* result = (int*)malloc(imageSize);
  CUDA_SAFE_CALL(cudaMemcpy(result, devImage, imageSize, cudaMemcpyDeviceToHost));
  return result;
}

int findPitchInInts(int width) {
  /* int* test; */
/*   size_t pitch; */
/*   cudaMallocPitch((void**)&test, &pitch, width * sizeof(int), 1); */
/*   cudaFree(test); */
/*   return pitch/sizeof(int); */
  return ((width - 1)/16 + 1) * 16;
}


int pixelPitch;
int binPitch;
int border;
int width;
int height;
int borderWidth;
int borderHeight;
int* devQuantized;
int* devMirrored;
float* devGradientA;
float* devGradientB;
int* devTurned;
int* devImageT;
int* devIntegralCol;
int* devIntegralColT;
int* devIntegralsT;
int* devIntegrals;
float* devGradients;
uint norients;
uint nscale;

int initializeGradients(uint widthIn, uint heightIn, uint borderIn, uint maxbins, uint norientsIn, uint nscaleIn, uint textonChoice) {
  width = widthIn;
  height = heightIn;
  border = borderIn;
  norients = norientsIn;
  nscale = nscaleIn;
  borderWidth = width + 2 * border;
  borderHeight = height + 2 * border;
  
  CUDA_SAFE_CALL(cudaMalloc((void**)&devGradients, sizeof(float) * norients * nscale * borderWidth * borderHeight));
  CUDA_SAFE_CALL(cudaMalloc((void**)&devQuantized, sizeof(int) * width * height));
  CUDA_SAFE_CALL(cudaMalloc((void**)&devMirrored, sizeof(int) * borderWidth * borderHeight));
  CUDA_SAFE_CALL(cudaMalloc((void**)&devGradientA, sizeof(float) * width * height));
  CUDA_SAFE_CALL(cudaMalloc((void**)&devGradientB, sizeof(float) * width * height));

  int maxWidth = borderWidth + borderHeight;
  int maxHeight = maxWidth;
  int maxBins = 32;
  if (textonChoice == TEXTON64)
      maxBins = 64;
  //pixelPitch = findPitchInInts(maxWidth * maxHeight);
  binPitch = findPitchInInts(maxBins);
  

  
  CUDA_SAFE_CALL(cudaMalloc((void**)&devTurned, sizeof(int) * maxWidth * maxHeight));
  CUDA_SAFE_CALL(cudaMalloc((void**)&devIntegrals, sizeof(int) * binPitch * maxWidth * maxHeight));
  return binPitch;
}


void finalizeGradients() {
  CUDA_SAFE_CALL(cudaFree(devIntegrals));
  CUDA_SAFE_CALL(cudaFree(devTurned));
  CUDA_SAFE_CALL(cudaFree(devGradientB));
  CUDA_SAFE_CALL(cudaFree(devGradientA));
  CUDA_SAFE_CALL(cudaFree(devMirrored));
  CUDA_SAFE_CALL(cudaFree(devQuantized));
  CUDA_SAFE_CALL(cudaFree(devGradients));
}



float* gradients(float* devImage, uint nbins, bool blur, float sigma, uint* radii, int textonChoice) {
 
  quantizeImage(width, height, nbins, devImage, devQuantized);
  mirrorImage(width, height, border, devQuantized, devMirrored);
    
  for(int orientation = 0; orientation < norients/2; orientation++) {
    float thetaPi = -float(orientation)/float(norients);
    int newWidth;
    int newHeight;
    rotateImage(borderWidth, borderHeight, devMirrored, thetaPi, newWidth, newHeight, devTurned);
    int* devTurnedImage = devTurned;
    if (orientation == 0) {
      devTurnedImage = devMirrored;
    }
    
    
    formIntegralImages(newWidth, newHeight, nbins, devTurnedImage, binPitch, devIntegrals);
    
    
    for (int scale = 0; scale < nscale; scale++) {

	if (TEXTON32 == textonChoice)
	{
	    dispatchGradient(false, width, height, border, nbins, thetaPi, newWidth, radii[scale], blur, (int)(sigma*(float)nbins), devIntegrals, binPitch, devGradientA, devGradientB);
	}
	else
	{
	    
	    dispatchGradient_64(width, height, border, nbins, thetaPi, newWidth, radii[scale], blur, (int)(sigma*(float)nbins), devIntegrals, binPitch, devGradientA, devGradientB);
	}
	mirrorImage(width, height, border, devGradientA, &devGradients[borderWidth * borderHeight * (scale * norients + orientation + norients / 2)]);
      mirrorImage(width, height, border, devGradientB, &devGradients[borderWidth * borderHeight * (scale * norients + orientation)]);
    } 
  }
  return devGradients;
}

float* gradients(int* devImage, uint nbins, bool blur, float sigma, uint* radii, int textonChoice) {
 
  mirrorImage(width, height, border, devImage, devMirrored);
    
  for(int orientation = 0; orientation < norients/2; orientation++) {
    float thetaPi = -float(orientation)/float(norients);
    int newWidth;
    int newHeight;
    rotateImage(borderWidth, borderHeight, devMirrored, thetaPi, newWidth, newHeight, devTurned);
    int* devTurnedImage = devTurned;
    if (orientation == 0) {
      devTurnedImage = devMirrored;
    }
    
    
    formIntegralImages(newWidth, newHeight, nbins, devTurnedImage, binPitch, devIntegrals);
    
    
    for (int scale = 0; scale < nscale; scale++) {

	if (TEXTON32 == textonChoice)
	{
	    dispatchGradient(true, width, height, border, nbins, thetaPi, newWidth, radii[scale], blur, (int)(sigma*(float)nbins), devIntegrals, binPitch, devGradientA, devGradientB);
	}
	else
	{
	    dispatchGradient_64(width, height, border, nbins, thetaPi, newWidth, radii[scale], blur, (int)(sigma*(float)nbins), devIntegrals, binPitch, devGradientA, devGradientB);
	
	}
	mirrorImage(width, height, border, devGradientA, &devGradients[borderWidth * borderHeight * (scale * norients + orientation + norients / 2)]);
      mirrorImage(width, height, border, devGradientB, &devGradients[borderWidth * borderHeight * (scale * norients + orientation)]);
    } 
  }
  return devGradients;
}
