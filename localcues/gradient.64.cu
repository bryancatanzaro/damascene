#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <helper_cuda.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include "rotate.h"
#include "convert.h"
#include <helper_cuda.h>
#include <assert.h>


#define uint unsigned int

#define IMUL(a, b) __mul24(a, b)


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


template <int compress>
__global__ void integrateImage(int width, int height, int bin, int* inputImage, int* integralImage) {
  __shared__ int data[512];
  __shared__ int temp[16];
  int x = threadIdx.x;
  int y = blockIdx.y;
  int currentX = threadIdx.x * compress;
  int index = y * width + currentX;
  int localData[compress];
  int accumulant = 0;
  bool dead = ((currentX >= width) || (y >= height));
  if (!dead) {
    for(int i = 0; i < compress; i++) {
      if (currentX < width) {
        int pixel = inputImage[index];
        localData[i] = pixel;
        if (pixel == bin) accumulant++;
        index++;
        currentX++;
      } else {
        localData[i] = -1;
      }
    }
  }
  data[threadIdx.x] = accumulant;
  __syncthreads();
  int warpId = threadIdx.x >> 5;
  int warpIdx = threadIdx.x & 0x1f;
  if (warpIdx >= 16)  data[x] = data[x - 16] + data[x];
  if (warpIdx >=  8)  data[x] = data[x -  8] + data[x];
  if (warpIdx >=  4)  data[x] = data[x -  4] + data[x];
  if (warpIdx >=  2)  data[x] = data[x -  2] + data[x];
  if (warpIdx >=  1)  data[x] = data[x -  1] + data[x];
  if (warpIdx == 31)   temp[warpId] = data[x];
  __syncthreads();
  if ((warpId == 0) && (warpIdx < 16)) {
    if (warpIdx >=  8)  temp[x] = temp[x -  8] + temp[x];
    if (warpIdx >=  4)  temp[x] = temp[x -  4] + temp[x];
    if (warpIdx >=  2)  temp[x] = temp[x -  2] + temp[x];
    if (warpIdx >=  1)  temp[x] = temp[x -  1] + temp[x];
  }
  __syncthreads();
  if (!dead) {
    accumulant = 0;
    if (warpId > 0) {
      accumulant = temp[warpId - 1];
    }
    if (warpIdx > 0) {
      accumulant += data[x - 1];
    }
    currentX = threadIdx.x * compress;
    index = y * width + currentX;
    for (int i = 0; i < compress; i++) {
      if (localData[i] == bin) accumulant++;
      if (currentX < width) {
        integralImage[index] = accumulant;
        index++;
        currentX++;
      }
    }
  }
}

//    integrateImageT<1><<<dim3(1, height), dim3(512, 1)>>>(width, height, devImage, binIndex, binPitch, devIntegrals);
template <int compress>
__global__ void integrateImageT(int width, int height, int* inputImage, int binIndex, int binPitch, int* devIntegrals) {
  __shared__ int data[512];
  __shared__ int temp[16];
  int x = threadIdx.x;
  int y = blockIdx.y;
  int currentX = threadIdx.x * compress;
  int index = y * width + currentX;
  int localData[compress];
  int accumulant = 0;
  bool dead = ((currentX >= width) || (y >= height));
  if (!dead) {
    for(int i = 0; i < compress; i++) {
      if (currentX < width) {
        int pixel = inputImage[index];
        localData[i] = pixel;
        accumulant += pixel;
        index++;
        currentX++;
      } else {
        localData[i] = 0;
      }
    }
  }
  data[threadIdx.x] = accumulant;
  __syncthreads();
  int warpId = threadIdx.x >> 5;
  int warpIdx = threadIdx.x & 0x1f;
  if (warpIdx >= 16)  data[x] = data[x - 16] + data[x];
  if (warpIdx >=  8)  data[x] = data[x -  8] + data[x];
  if (warpIdx >=  4)  data[x] = data[x -  4] + data[x];
  if (warpIdx >=  2)  data[x] = data[x -  2] + data[x];
  if (warpIdx >=  1)  data[x] = data[x -  1] + data[x];
  if (warpIdx == 31)   temp[warpId] = data[x];
  __syncthreads();
  if ((warpId == 0) && (warpIdx < 16)) {
    if (warpIdx >=  8)  temp[x] = temp[x -  8] + temp[x];
    if (warpIdx >=  4)  temp[x] = temp[x -  4] + temp[x];
    if (warpIdx >=  2)  temp[x] = temp[x -  2] + temp[x];
    if (warpIdx >=  1)  temp[x] = temp[x -  1] + temp[x];
  }
  __syncthreads();
  if (!dead) {
    accumulant = 0;
    if (warpId > 0) {
      accumulant = temp[warpId - 1];
    }
    if (warpIdx > 0) {
      accumulant += data[x - 1];
    }
    currentX = threadIdx.x * compress;
    index = binIndex + (y * width + currentX) * binPitch;
    for (int i = 0; i < compress; i++) {
      accumulant += localData[i];
      if (currentX < width) {
        devIntegrals[index] = accumulant;
        index += binPitch;
        currentX++;
      }
    }
  }
}

/* __global__ void integrateImage(int width, int height, int bin, int* inputImage, int* integralImage) { */
/*   __shared__ int data[512]; */
/*   __shared__ int temp[16]; */
/*   __shared__ bool moreWork; */
/*   int x = threadIdx.x; */
/*   int y = blockIdx.y; */
/*   int currentX = threadIdx.x; */
/*   int index = y * width + currentX; */
 
/*   if (x == 0) { */
/*     moreWork = true; */
/*     temp[15] = 0; */
/*   } */

/*   __syncthreads(); */
  
/*   while (moreWork) { */
/*     bool dead = ((currentX >= width) || (y >= height)); */
/*     int currentValue = 0; */
    
/*     if (!dead) { */
/*       if (currentX < width)  { */
/*         if (inputImage[index] == bin) { */
/*           currentValue = 1; */
/*         } */
/*       }  */
/*     } */
/*     if (x == 0) { */
/*       currentValue += temp[15]; */
/*     } */
/*     data[x] = currentValue; */
/*     __syncthreads(); */
/*     int warpId = threadIdx.x >> 5; */
/*     int warpIdx = threadIdx.x & 0x1f; */
/*     if (warpIdx >= 16)  data[x] = data[x - 16] + data[x]; */
/*     if (warpIdx >=  8)  data[x] = data[x -  8] + data[x]; */
/*     if (warpIdx >=  4)  data[x] = data[x -  4] + data[x]; */
/*     if (warpIdx >=  2)  data[x] = data[x -  2] + data[x]; */
/*     if (warpIdx >=  1)  data[x] = data[x -  1] + data[x]; */
/*     if (warpIdx == 31)   temp[warpId] = data[x]; */
/*     __syncthreads(); */
/*     if ((warpId == 0) && (warpIdx < 16)) { */
/*       if (warpIdx >=  8)  temp[x] = temp[x -  8] + temp[x]; */
/*       if (warpIdx >=  4)  temp[x] = temp[x -  4] + temp[x]; */
/*       if (warpIdx >=  2)  temp[x] = temp[x -  2] + temp[x]; */
/*       if (warpIdx >=  1)  temp[x] = temp[x -  1] + temp[x]; */
/*     } */
/*     __syncthreads(); */
  
/*     if (!dead) { */
/*       if (currentX < width) { */
/*         int accumulant = 0; */
/*         if (warpId > 0) { */
/*           accumulant = temp[warpId - 1]; */
/*         } */
/*         accumulant += data[x]; */
/*         integralImage[index] = accumulant; */
/*       } */
/*     } */
/*     index += 512; */
/*     currentX += 512; */
    
/*     if (x == 0) { */
/*       if (currentX >= width) { */
/*         moreWork = false; */
/*       } */
/*     } */
/*     __syncthreads(); */
/*   } */
/* } */

/* __global__ void integrateImage(int width, int height, int* inputImage, int* integralImage) { */
/*   __shared__ int data[512]; */
/*   __shared__ int temp[16]; */
/*   __shared__ bool moreWork; */
/*   int x = threadIdx.x; */
/*   int y = blockIdx.y; */
/*   int currentX = threadIdx.x; */
/*   int index = y * width + currentX; */
 
/*   if (x == 0) { */
/*     moreWork = true; */
/*     temp[15] = 0; */
/*   } */
/*   __syncthreads(); */
  
/*   while (moreWork) { */
/*     bool dead = ((currentX >= width) || (y >= height)); */
/*     int currentValue = 0; */
/*     if (!dead) { */
/*       if (currentX < width)  { */
/*         currentValue = inputImage[index]; */
/*       }  */
/*     } */
/*     if (x == 0) { */
/*       currentValue += temp[15]; */
/*     } */
/*     data[x] = currentValue; */
/*     __syncthreads(); */
/*     int warpId = threadIdx.x >> 5; */
/*     int warpIdx = threadIdx.x & 0x1f; */
/*     if (warpIdx >= 16)  data[x] = data[x - 16] + data[x]; */
/*     if (warpIdx >=  8)  data[x] = data[x -  8] + data[x]; */
/*     if (warpIdx >=  4)  data[x] = data[x -  4] + data[x]; */
/*     if (warpIdx >=  2)  data[x] = data[x -  2] + data[x]; */
/*     if (warpIdx >=  1)  data[x] = data[x -  1] + data[x]; */
/*     if (warpIdx == 31)   temp[warpId] = data[x]; */
/*     __syncthreads(); */
/*     if ((warpId == 0) && (warpIdx < 16)) { */
/*       if (warpIdx >=  8)  temp[x] = temp[x -  8] + temp[x]; */
/*       if (warpIdx >=  4)  temp[x] = temp[x -  4] + temp[x]; */
/*       if (warpIdx >=  2)  temp[x] = temp[x -  2] + temp[x]; */
/*       if (warpIdx >=  1)  temp[x] = temp[x -  1] + temp[x]; */
/*     } */
/*     __syncthreads(); */
  
/*     if (!dead) { */
/*       if (currentX < width) { */
/*         int accumulant = 0; */
/*         if (warpId > 0) { */
/*           accumulant = temp[warpId - 1]; */
/*         } */
/*         accumulant += data[x]; */
/*         integralImage[index] = accumulant; */
/*       } */
/*     } */
/*     index += 512; */
/*     currentX += 512; */
    
    
/*     if (x == 0) { */
/*       if (currentX >= width) { */
/*         moreWork = false; */
/*       } */
/*     } */
/*     __syncthreads(); */
/*   } */
/* } */





void dispatchIntegration(int width, int height, int bin, int* devImage, int* devIntegral) {
  //integrateImage<<<dim3(1, height), dim3(512, 1)>>>(width, height, bin, devImage, devIntegral);
  
  if (width <= 512) {
    integrateImage<1><<<dim3(1, height), dim3(512, 1)>>>(width, height, bin, devImage, devIntegral);
  } else if (width <= 1024) {
    integrateImage<2><<<dim3(1, height), dim3(512, 1)>>>(width, height, bin, devImage, devIntegral);
  } else if (width <= 1536) {
    integrateImage<3><<<dim3(1, height), dim3(512, 1)>>>(width, height, bin, devImage, devIntegral);
  } else if (width <= 2048) {
    integrateImage<4><<<dim3(1, height), dim3(512, 1)>>>(width, height, bin, devImage, devIntegral);
  } else if (width <= 2560) {
    integrateImage<5><<<dim3(1, height), dim3(512, 1)>>>(width, height, bin, devImage, devIntegral);
  } else if (width <= 3072) {
    integrateImage<6><<<dim3(1, height), dim3(512, 1)>>>(width, height, bin, devImage, devIntegral);
  } else if (width <= 3584) {
    integrateImage<7><<<dim3(1, height), dim3(512, 1)>>>(width, height, bin, devImage, devIntegral);
  } else if (width <= 4096) {
    integrateImage<8><<<dim3(1, height), dim3(512, 1)>>>(width, height, bin, devImage, devIntegral);
  } else {
    printf("Can't handle images with dimensions > 4096\n");
    exit(1);
  }
}

void dispatchIntegrationT(int width, int height, int* devImage, int binIndex, int binPitch, int* devIntegrals) {
  //integrateImage<<<dim3(1, height), dim3(512, 1)>>>(width, height, devImage, devIntegral);
  if (width <= 512) {
    integrateImageT<1><<<dim3(1, height), dim3(512, 1)>>>(width, height, devImage, binIndex, binPitch, devIntegrals);
  } else if (width <= 1024) {
    integrateImageT<2><<<dim3(1, height), dim3(512, 1)>>>(width, height, devImage, binIndex, binPitch, devIntegrals);
  } else if (width <= 1536) {
    integrateImageT<3><<<dim3(1, height), dim3(512, 1)>>>(width, height, devImage, binIndex, binPitch, devIntegrals);
  } else if (width <= 2048) {
    integrateImageT<4><<<dim3(1, height), dim3(512, 1)>>>(width, height, devImage, binIndex, binPitch, devIntegrals);
  } else if (width <= 2560) {
    integrateImageT<5><<<dim3(1, height), dim3(512, 1)>>>(width, height, devImage, binIndex, binPitch, devIntegrals);
  } else if (width <= 3072) {
    integrateImageT<6><<<dim3(1, height), dim3(512, 1)>>>(width, height, devImage, binIndex, binPitch, devIntegrals);
  } else if (width <= 3584) {
    integrateImageT<7><<<dim3(1, height), dim3(512, 1)>>>(width, height, devImage, binIndex, binPitch, devIntegrals);
  } else if (width <= 4096) {
    integrateImageT<8><<<dim3(1, height), dim3(512, 1)>>>(width, height, devImage, binIndex, binPitch, devIntegrals);
  } else {
    printf("Can't handle images with dimensions > 4096\n");
  }
}



/**
 * For a given orientation, computes the integral images for each of the histogram bins
 */
void formIntegralImages(int width, int height, int nbins, int* devImage,
                        int* devImageT, int* devIntegralCol, int* devIntegralColT,
                        int binPitch, int* devIntegrals) {
  //uint integralTimer;
  //cutCreateTimer(&integralTimer);
  //cutStartTimer(integralTimer);
  //int pixelPitch = findPitchInInts(width * height);
  //int* devIntegralCol;
  //checkCudaErrors(cudaMalloc((void**)&devIntegralCol, sizeof(int) * width * height));
  //int* devIntegralsT;
  //checkCudaErrors(cudaMalloc((void**)&devIntegralsT, sizeof(int) * pixelPitch * nbins));
 
  //int* hostIntegralCol = (int*)malloc(sizeof(int) * width * height);
  //int* devImageT;
  //cudaMalloc((void**)&devImageT, sizeof(int) * width * height);
  dim3 gridDim = dim3((width - 1)/16 + 1, (height - 1)/16 + 1);
  dim3 gridDimT = dim3((height - 1)/16 + 1, (width - 1)/16 + 1);
  dim3 blockDim = dim3(16, 16);
  transposeImage<<<gridDim, blockDim>>>(width, height, devImage, devImageT);

  //int* devIntegralColT;
  //cudaMalloc((void**)&devIntegralColT, sizeof(int) * width * height);
  //int bin = 0;
  //cudaThreadSynchronize();
  //cutStopTimer(integralTimer);
  //printf("Integral preamble: %f ms\n", cutGetTimerValue(integralTimer));
  //cutStartTimer(integralTimer);
  for(int bin = 0; bin < nbins; bin++) {
    dispatchIntegration(height, width, bin, devImageT, devIntegralCol);
   /*  int* hostIntegralCol = (int*)malloc(sizeof(int) * width * height); */
/*     cudaMemcpy(hostIntegralCol, devIntegralCol, sizeof(int) * width * height, cudaMemcpyDeviceToHost); */
/*     writeFile("integralCol.pb", height, width, hostIntegralCol); */
    CUT_CHECK_ERROR("After column integration");
    transposeImage<<<gridDimT, blockDim>>>(height, width, devIntegralCol, devIntegralColT);
   /*  int* hostIntegralColT = (int*)malloc(sizeof(int) * width * height); */
/*     cudaMemcpy(hostIntegralColT, devIntegralColT, sizeof(int) * width * height, cudaMemcpyDeviceToHost); */
/*     writeFile("integralColT.pb", width, height, hostIntegralColT); */

    
    dispatchIntegrationT(width, height, devIntegralColT, bin, binPitch, devIntegrals);
   /*  int* hostIntegral = (int*)malloc(sizeof(int) * width * height); */
/*     cudaMemcpy(hostIntegral, devIntegralsT, sizeof(int) * width * height, cudaMemcpyDeviceToHost); */
/*     writeFile("integral.pb", width, height, hostIntegral); */
    CUT_CHECK_ERROR("After row integration");
  }
  //cudaThreadSynchronize();
  //cutStopTimer(integralTimer);
  //printf("Integral Body: %f ms\n", cutGetTimerValue(integralTimer));
  
  //cutStartTimer(integralTimer);
  //cudaFree(devIntegralColT);
  //cudaFree(devImageT);
  //cudaFree(devIntegralCol);
  //integralPitch = findPitchInInts(nbins);
  CUT_CHECK_ERROR("After finding integral pitch");
  //cudaMalloc((void**)p_devIntegrals, sizeof(int) * integralPitch * width * height);
  
  
  // int nPixels = width * height;
  //gridDim = dim3((nPixels - 1)/16 + 1, (nbins - 1)/16 + 1);
  //blockDim = dim3(16, 16);
  //transposeImage<<<gridDim, blockDim>>>(width * height, nbins, devIntegralsT, pixelPitch, devIntegrals, binPitch);
  //CUT_CHECK_ERROR("After transpose");
  //cudaFree(devIntegralsT);
  //cudaThreadSynchronize();
  //cutStopTimer(integralTimer);
  //printf("Integral postscript: %f ms\n\n", cutGetTimerValue(integralTimer));
}



float* getImage(uint width, uint height, float* devImage) {
  int imageSize = width * height * sizeof(float);
  float* result = (float*)malloc(imageSize);
  checkCudaErrors(cudaMemcpy(result, devImage, imageSize, cudaMemcpyDeviceToHost));
  return result;
}

int* getImage(uint width, uint height, int* devImage) {
  int imageSize = width * height * sizeof(int);
  int* result = (int*)malloc(imageSize);
  checkCudaErrors(cudaMemcpy(result, devImage, imageSize, cudaMemcpyDeviceToHost));
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

int initializeGradients(uint widthIn, uint heightIn, uint borderIn, uint maxbins, uint norientsIn, uint nscaleIn) {
  width = widthIn;
  height = heightIn;
  border = borderIn;
  norients = norientsIn;
  nscale = nscaleIn;
  borderWidth = width + 2 * border;
  borderHeight = height + 2 * border;
  
  checkCudaErrors(cudaMalloc((void**)&devGradients, sizeof(float) * norients * nscale * borderWidth * borderHeight));
  checkCudaErrors(cudaMalloc((void**)&devQuantized, sizeof(int) * width * height));
  checkCudaErrors(cudaMalloc((void**)&devMirrored, sizeof(int) * borderWidth * borderHeight));
  checkCudaErrors(cudaMalloc((void**)&devGradientA, sizeof(float) * width * height));
  checkCudaErrors(cudaMalloc((void**)&devGradientB, sizeof(float) * width * height));

  int maxWidth = borderWidth + borderHeight;
  int maxHeight = maxWidth;
  int maxBins = 64;
  //pixelPitch = findPitchInInts(maxWidth * maxHeight);
  binPitch = findPitchInInts(maxBins);
  

  
  checkCudaErrors(cudaMalloc((void**)&devTurned, sizeof(int) * maxWidth * maxHeight));
  checkCudaErrors(cudaMalloc((void**)&devImageT, sizeof(int) * maxWidth * maxHeight));
  checkCudaErrors(cudaMalloc((void**)&devIntegralCol, sizeof(int) * maxWidth * maxHeight));
  checkCudaErrors(cudaMalloc((void**)&devIntegralColT, sizeof(int) * maxWidth * maxHeight));
  //checkCudaErrors(cudaMalloc((void**)&devIntegralsT, sizeof(int) * pixelPitch * maxBins));
  checkCudaErrors(cudaMalloc((void**)&devIntegrals, sizeof(int) * binPitch * maxWidth * maxHeight));
  return binPitch;
}


void finalizeGradients() {
  checkCudaErrors(cudaFree(devIntegrals));
 /*  checkCudaErrors(cudaFree(devIntegralsT)); */
  checkCudaErrors(cudaFree(devIntegralColT));
  checkCudaErrors(cudaFree(devIntegralCol));
  checkCudaErrors(cudaFree(devImageT));
  checkCudaErrors(cudaFree(devTurned));
  checkCudaErrors(cudaFree(devGradientB));
  checkCudaErrors(cudaFree(devGradientA));
  checkCudaErrors(cudaFree(devMirrored));
  checkCudaErrors(cudaFree(devQuantized));
  checkCudaErrors(cudaFree(devGradients));
}



float* gradients(float* devImage, uint nbins, bool blur, float sigma, uint* radii) {
 
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
    
    
    formIntegralImages(newWidth, newHeight, nbins, devTurnedImage,
                       devImageT, devIntegralCol, devIntegralColT,
                       binPitch, devIntegrals);
    
    
    for (int scale = 0; scale < nscale; scale++) {
      dispatchGradient(width, height, border, nbins, thetaPi, newWidth, radii[scale], blur, (int)(sigma*(float)nbins), devIntegrals, binPitch, devGradientA, devGradientB);
      mirrorImage(width, height, border, devGradientA, &devGradients[borderWidth * borderHeight * (scale * norients + orientation + norients / 2)]);
      mirrorImage(width, height, border, devGradientB, &devGradients[borderWidth * borderHeight * (scale * norients + orientation)]);
    } 
  }
  return devGradients;
}

float* gradients(int* devImage, uint nbins, bool blur, float sigma, uint* radii) {
 
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
    
    
    formIntegralImages(newWidth, newHeight, nbins, devTurnedImage,
                       devImageT, devIntegralCol, devIntegralColT,
                       binPitch, devIntegrals);
    
    
    for (int scale = 0; scale < nscale; scale++) {
      dispatchGradient(width, height, border, nbins, thetaPi, newWidth, radii[scale], blur, (int)(sigma*(float)nbins), devIntegrals, binPitch, devGradientA, devGradientB);
      mirrorImage(width, height, border, devGradientA, &devGradients[borderWidth * borderHeight * (scale * norients + orientation + norients / 2)]);
      mirrorImage(width, height, border, devGradientB, &devGradients[borderWidth * borderHeight * (scale * norients + orientation)]);
    } 
  }
  return devGradients;
}
