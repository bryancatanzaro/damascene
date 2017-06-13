// vim: ts=4 syntax=cpp comments=

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_image.h>
#include "Stencil.h"

#define XBLOCK 16
#define YBLOCK 16

__global__ void rgbUtoGreyF_kernel(int width, int height, unsigned int* rgbU, float* grey) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if ((x < width) && (y < height)) {
    int index = y * width + x;
    unsigned int rgb = rgbU[index];
    float r = (float)(rgb & 0xff)/255.0;
    float g = (float)((rgb & 0xff00) >> 8)/255.0;
    float b = (float)((rgb & 0xff0000) >> 16)/255.0;
    grey[index] =
      (0.29894 * r)
      + (0.58704 * g)
      + (0.11402 * b);
  }
}

__global__ void rgbUtoLab3F_kernel(int width, int height, float gamma, unsigned int* rgbU, float* devL, float* devA, float* devB) {
  int x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int y0 = blockDim.y * blockIdx.y + threadIdx.y;
  if ((x0 < width) && (y0 < height)) {
    int index = y0 * width + x0;
    unsigned int rgb = rgbU[index];
    float r = (float)(rgb & 0xff)/255.0;
    float g = (float)((rgb & 0xff00) >> 8)/255.0;
    float b = (float)((rgb & 0xff0000) >> 16)/255.0;
    r = powf(r, gamma);
    g = powf(g, gamma);
    b = powf(b, gamma);
    float x = (0.412453 * r) +  (0.357580 * g) + (0.180423 * b);
    float y = (0.212671 * r) +  (0.715160 * g) + (0.072169 * b);
    float z = (0.019334 * r) +  (0.119193 * g) + (0.950227 * b);
    /*D65 white point reference */
    const float x_ref = 0.950456;
    const float y_ref = 1.000000;
    const float z_ref = 1.088754;
    /* threshold value  */
    const float threshold = 0.008856;
    x = x / x_ref;
    y = y / y_ref;
    z = z / z_ref;
    
    float fx =
      (x > threshold) ? powf(x,(1.0/3.0)) : (7.787*x + (16.0/116.0));
    float fy =
      (y > threshold) ? powf(y,(1.0/3.0)) : (7.787*y + (16.0/116.0));
    float fz =
      (z > threshold) ? powf(z,(1.0/3.0)) : (7.787*z + (16.0/116.0));
    /* compute Lab color value */
    devL[index] =
         (y > threshold) ? (116*powf(y,(1.0/3.0)) - 16) : (903.3*y);
    devA[index] = 500.0f * (fx - fy);
    devB[index] = 200.0f * (fy - fz);
  }
}

__global__ void normalizeLab_kernel(uint width, uint height, float* devL, float* devA, float* devB) {
  int x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int y0 = blockDim.y * blockIdx.y + threadIdx.y;
  if ((x0 < width) && (y0 < height)) {
    int index = y0 * width + x0;
    const float ab_min = -73;
    const float ab_max = 95;
    const float ab_range = ab_max - ab_min;
    /* normalize Lab image */
    float l_val = devL[index] / 100.0f;
    float a_val = (devA[index] - ab_min) / ab_range;
    float b_val = (devB[index] - ab_min) / ab_range;
    if (l_val < 0) { l_val = 0; } else if (l_val > 1) { l_val = 1; }
    if (a_val < 0) { a_val = 0; } else if (a_val > 1) { a_val = 1; }
    if (b_val < 0) { b_val = 0; } else if (b_val > 1) { b_val = 1; }
    devL[index] = l_val;
    devA[index] = a_val;
    devB[index] = b_val;
  }
}


void loadPPM_rgbU(char* filename, uint* p_width, uint* p_height, uint** p_devRgbU) {
  unsigned int* data;
  sdkLoadPPM4ub(filename, (unsigned char**)&data, p_width, p_height);
  //cutLoadPPMub(filename, (unsigned char**)&data, p_width, p_height);
  uint imageSize = sizeof(uint) * (*p_width) * (*p_height);
  cudaMalloc((void**)p_devRgbU, imageSize);
  cudaMemcpy(*p_devRgbU, data, imageSize, cudaMemcpyHostToDevice);
}

void rgbUtoLab3F(uint width, uint height, float gamma, uint* devRgbU, float** p_devL, float** p_devA, float** p_devB) {
  dim3 gridDim = dim3((width - 1)/XBLOCK + 1, (height - 1)/YBLOCK + 1);
  dim3 blockDim = dim3(XBLOCK, YBLOCK);
  uint imageSize = sizeof(float) * width * height;
  cudaMalloc((void**)p_devL, imageSize);
  cudaMalloc((void**)p_devA, imageSize);
  cudaMalloc((void**)p_devB, imageSize);
  rgbUtoLab3F_kernel<<<gridDim, blockDim>>>(width, height, gamma, devRgbU, *p_devL, *p_devA, *p_devB);
}

void rgbUtoGreyF(uint width, uint height, uint* devRgbU, float** p_devGrey) {
  dim3 gridDim = dim3((width - 1)/XBLOCK + 1, (height - 1)/YBLOCK + 1);
  dim3 blockDim = dim3(XBLOCK, YBLOCK);
  uint imageSize = sizeof(float) * width * height;
  checkCudaErrors(cudaMalloc((void**)p_devGrey, imageSize));
  rgbUtoGreyF_kernel<<<gridDim, blockDim>>>(width, height, devRgbU, *p_devGrey);
}

void normalizeLab(uint width, uint height, float* devL, float* devA, float* devB) {
   dim3 gridDim = dim3((width - 1)/XBLOCK + 1, (height - 1)/YBLOCK + 1);
   dim3 blockDim = dim3(XBLOCK, YBLOCK);
   normalizeLab_kernel<<<gridDim, blockDim>>>(width, height, devL, devA, devB);
}


__global__ void mirrorImage_kernel(uint width, uint height, uint border, uint borderWidth, uint borderHeight, float* devInput, float* devOutput) {
  int x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int y0 = blockDim.y * blockIdx.y + threadIdx.y;
  if ((x0 < borderWidth) && (y0 < borderHeight)) {
    int x1 = 0;
    int y1 = 0;
    if (x0 < border) {
      x1 = border - x0 - 1;
    } else if (x0 < border + width) {
      x1 = x0 - border;
    } else {
      x1 = border + 2 * width - x0 - 1;
    }
    if (y0 < border) {
      y1 = border - y0 - 1;
    } else if (y0 < border + height) {
      y1 = y0 - border;
    } else {
      y1 = border + 2 * height - y0 - 1;
    }
    devOutput[y0 * borderWidth + x0] = devInput[y1 * width + x1];
  }
}

void mirrorImage(uint width, uint height, uint border, float* devInput, float** p_devOutput) {
  int borderWidth = width + 2 * border;
  int borderHeight = height + 2 * border;
  cudaMalloc((void**)p_devOutput, sizeof(float) * borderWidth * borderHeight);
  dim3 gridDim = dim3((borderWidth - 1)/XBLOCK + 1, (borderHeight - 1)/YBLOCK + 1);
  dim3 blockDim = dim3(XBLOCK, YBLOCK);
  mirrorImage_kernel<<<gridDim, blockDim>>>(width, height, border, borderWidth, borderHeight, devInput, *p_devOutput);
}

void mirrorImage(uint width, uint height, uint border, float* devInput, float* devOutput) {
  int borderWidth = width + 2 * border;
  int borderHeight = height + 2 * border;
  dim3 gridDim = dim3((borderWidth - 1)/XBLOCK + 1, (borderHeight - 1)/YBLOCK + 1);
  dim3 blockDim = dim3(XBLOCK, YBLOCK);
  mirrorImage_kernel<<<gridDim, blockDim>>>(width, height, border, borderWidth, borderHeight, devInput, devOutput);
}

__global__ void mirrorImage_kernel(uint width, uint height, uint border, uint borderWidth, uint borderHeight, int* devInput, int* devOutput) {
  int x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int y0 = blockDim.y * blockIdx.y + threadIdx.y;
  if ((x0 < borderWidth) && (y0 < borderHeight)) {
    int x1 = 0;
    int y1 = 0;
    if (x0 < border) {
      x1 = border - x0 - 1;
    } else if (x0 < border + width) {
      x1 = x0 - border;
    } else {
      x1 = border + 2 * width - x0 - 1;
    }
    if (y0 < border) {
      y1 = border - y0 - 1;
    } else if (y0 < border + height) {
      y1 = y0 - border;
    } else {
      y1 = border + 2 * height - y0 - 1;
    }
    devOutput[y0 * borderWidth + x0] = devInput[y1 * width + x1];
  }
}

void mirrorImage(uint width, uint height, uint border, int* devInput, int** p_devOutput) {
  int borderWidth = width + 2 * border;
  int borderHeight = height + 2 * border;
  cudaMalloc((void**)p_devOutput, sizeof(int) * borderWidth * borderHeight);
  dim3 gridDim = dim3((borderWidth - 1)/XBLOCK + 1, (borderHeight - 1)/YBLOCK + 1);
  dim3 blockDim = dim3(XBLOCK, YBLOCK);
  mirrorImage_kernel<<<gridDim, blockDim>>>(width, height, border, borderWidth, borderHeight, devInput, *p_devOutput);
}

void mirrorImage(uint width, uint height, uint border, int* devInput, int* devOutput) {
  int borderWidth = width + 2 * border;
  int borderHeight = height + 2 * border;
  dim3 gridDim = dim3((borderWidth - 1)/XBLOCK + 1, (borderHeight - 1)/YBLOCK + 1);
  dim3 blockDim = dim3(XBLOCK, YBLOCK);
  mirrorImage_kernel<<<gridDim, blockDim>>>(width, height, border, borderWidth, borderHeight, devInput, devOutput);
}


__global__ void unMirrorImage_kernel(uint width, uint height, uint border, uint borderWidth, uint borderHeight, float* devInput, float* devOutput) {
  int x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int y0 = blockDim.y * blockIdx.y + threadIdx.y;
  if ((x0 < borderWidth) && (y0 < borderHeight)) {
    int x1 = x0 + border;
    int y1 = y0 + border;
    devOutput[y0 * borderWidth + x0] = devInput[y1 * width + x1];
  }
}

void unMirrorImage(uint width, uint height, uint border, float* devInput, float** p_devOutput) {
  int borderWidth = width - 2 * border;
  int borderHeight = height - 2 * border;
  cudaMalloc((void**)p_devOutput, sizeof(float) * borderWidth * borderHeight);
  dim3 gridDim = dim3((borderWidth - 1)/XBLOCK + 1, (borderHeight - 1)/YBLOCK + 1);
  dim3 blockDim = dim3(XBLOCK, YBLOCK);
  unMirrorImage_kernel<<<gridDim, blockDim>>>(width, height, border, borderWidth, borderHeight, devInput, *p_devOutput);
}


void unMirrorImage(uint width, uint height, uint border, float* devInput, float* p_devOutput) {
  int borderWidth = width - 2 * border;
  int borderHeight = height - 2 * border;
  dim3 gridDim = dim3((borderWidth - 1)/XBLOCK + 1, (borderHeight - 1)/YBLOCK + 1);
  dim3 blockDim = dim3(XBLOCK, YBLOCK);
  unMirrorImage_kernel<<<gridDim, blockDim>>>(width, height, border, borderWidth, borderHeight, devInput, p_devOutput);
}


__global__ void quantizeImage_kernel(uint width, uint height, uint nbins, float* devInput, int* devOutput) {
  int x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int y0 = blockDim.y * blockIdx.y + threadIdx.y;
  if ((x0 < width) && (y0 < height)) {
    int index = y0 * width + x0;
    float input = devInput[index];
    int output = (int)floorf(input * (float)nbins);
    if (output == nbins) {
      output = nbins - 1;
    }
    devOutput[index] = output;
  }
}


void quantizeImage(uint width, uint height, uint nbins, float* devInput, int** p_devOutput) {
  cudaMalloc((void**)p_devOutput, sizeof(int) * width * height);
  dim3 gridDim = dim3((width - 1)/XBLOCK + 1, (height - 1)/YBLOCK + 1);
  dim3 blockDim = dim3(XBLOCK, YBLOCK);
  quantizeImage_kernel<<<gridDim, blockDim>>>(width, height, nbins, devInput, *p_devOutput);
}

void quantizeImage(uint width, uint height, uint nbins, float* devInput, int* devOutput) {
  dim3 gridDim = dim3((width - 1)/XBLOCK + 1, (height - 1)/YBLOCK + 1);
  dim3 blockDim = dim3(XBLOCK, YBLOCK);
  quantizeImage_kernel<<<gridDim, blockDim>>>(width, height, nbins, devInput, devOutput);
}
