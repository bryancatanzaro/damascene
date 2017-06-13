#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>

#define IMUL(a, b)  __mul24(a, b)
#define M_PIl           3.1415926535897932384626433832795029L  /* pi */
#define M_PI_2l         1.5707963267948966192313216916397514L  /* pi/2 */
#define M_PI_4l         0.7853981633974483096156608458198757L  /* pi/4 */
#define MAXLENGTH 4000
#define MAXKERNEL 64
__constant__ int baseLineX[MAXLENGTH];
__constant__ int baseLineY[MAXLENGTH];
__constant__ int guideLineX[MAXLENGTH];
__constant__ int guideLineY[MAXLENGTH];

__constant__ float blurKernel[MAXKERNEL];
__constant__ int rectangleOffsets[8];

int hostBaseLineX[MAXLENGTH];
int hostBaseLineY[MAXLENGTH];
int hostGuideLineX[MAXLENGTH];
int hostGuideLineY[MAXLENGTH];


#define uchar unsigned char

void precompute_gaussian(float sigma, int& kernel_radius, int& kernel_length, float** p_kernel)
{
  kernel_radius = (int)ceil(sigma*3);
	kernel_length = 2*kernel_radius + 1;

	*p_kernel = (float*)malloc(kernel_length*sizeof(float));

  float* kernel = *p_kernel;
  
  int i;
  float sigma2_inv = float(1)/sigma/sigma;
  float neg_two_sigma2_inv = float(-0.5)*sigma2_inv;
  float sum = 0;

	for (i=0; i<kernel_length; i++)
    {
      kernel[i] = exp(float(i-kernel_radius)*float(i-kernel_radius)*neg_two_sigma2_inv);
      sum += kernel[i];
    }

	for (i=0; i<kernel_length; i++)
    {
      kernel[i] /= sum;
    }

#if 0
  printf("computed kernel_length: %d\n", kernel_length);
  for (i=0; i<kernel_length; i++)
    {
      printf("%d: %9.6f\n", i, kernel[i]);
    }
  printf("\n");
#endif
}


void bresenhamLine(int x0, int y0, int x1, int y1, int& length, int* x, int* y) {
  int dx = x1 - x0;
  int dy = y1 - y0;
 
  int adx = abs(dx);
  int ady = abs(dy);

  length = (adx > ady) ? adx : ady;
  length++;
  
  // figure out what octant we're in for the bresenham algorithm;
  // octant i covers pi/4 * [i,i+1)
  int octant = -1;
  if (dx > 0 && dy >= 0) {           // quadrant 0
    octant = (adx > ady) ? 0 : 1;
  } else if (dx <= 0 && dy > 0) {    // quadrant 1
    octant = (adx < ady) ? 2 : 3;
  } else if (dy <= 0 && dx < 0) {    // quadrant 2
    octant = (adx > ady) ? 4 : 5;
  } else if (dx >= 0 && dy < 0) {    // quadrant 3
    octant = (adx < ady) ? 6 : 7;
  }

  // t is our bresenham counter
  int t = 0;
  switch (octant)
    {
    case 0: t = -adx; break;
    case 1: t = -ady; break;
    case 2: t = -ady; break;
    case 3: t = -adx; break;
    case 4: t = -adx; break;
    case 5: t = -ady; break;
    case 6: t = -ady; break;
    case 7: t = -adx; break;
      
    }

  int xi = x0;
  int yi = y0;
  int index = 1;
  x[0] = xi;
  y[0] = yi;
  while (xi != x1 || yi != y1)
    {
      // step one pixel on the bresenham line
      switch (octant)
        {
        case 0:
          xi++; t += (ady << 1);
          if (t > 0) { yi++; t -= (adx << 1); }
          break;
        case 1:
          yi++; t += (adx << 1);
          if (t > 0) { xi++; t -= (ady << 1); }
          break;
        case 2:
          yi++; t += (adx << 1);
          if (t > 0) { xi--; t -= (ady << 1); }
          break;
        case 3:
          xi--; t += (ady << 1);
          if (t > 0) { yi++; t -= (adx << 1); }
          break;
        case 4:
          xi--; t += (ady << 1);
          if (t > 0) { yi--; t -= (adx << 1); }
          break;
        case 5:
          yi--; t += (adx << 1);
          if (t > 0) { xi--; t -= (ady << 1); }
          break;
        case 6:
          yi--; t += (adx << 1);
          if (t > 0) { xi++; t -= (ady << 1); }
          break;
        case 7:
          xi++; t += (ady << 1);
          if (t > 0) { yi--; t -= (adx << 1); }
          break;
        }

      x[index] = xi;
      y[index] = yi;
      index++;
    } 
}

__global__ void turnImageP(int width, int height, int* inputImage, int* outputImage) {
  __shared__ int inputData[17*16];
  int x = blockDim.x * blockIdx.x + threadIdx.x - (gridDim.x*blockDim.x - width);
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int index = y * width + x;
  if ((x >= 0) && (y < height)) {
    inputData[IMUL(threadIdx.y, 17) + threadIdx.x] = inputImage[index];
  } else {
    inputData[IMUL(threadIdx.y, 17) + threadIdx.x] = 0;
  }
  int outputX = blockDim.y * blockIdx.y + threadIdx.x;
  int outputY = blockDim.x * (gridDim.x - blockIdx.x - 1) + threadIdx.y;
  index = outputY * height + outputX;
  __syncthreads();

  if ((outputX < height) && (outputY < width)) {
    outputImage[index] = inputData[IMUL(threadIdx.x, 17) + blockDim.x - threadIdx.y - 1];
  }  
}

__global__ void turnImageN(int width, int height, int* inputImage, int* outputImage) {
  __shared__ int inputData[17*16];
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y - (gridDim.y*blockDim.y - height);
  int index = y * width + x;
  if ((x < width) && (y >= 0)) {
    inputData[IMUL(threadIdx.y, 17) + threadIdx.x] = inputImage[index];
  } else {
    inputData[IMUL(threadIdx.y, 17) + threadIdx.x] = 0;
  }
  int outputX = blockDim.y * (gridDim.y - blockIdx.y - 1) + threadIdx.x;
  int outputY = blockDim.x * blockIdx.x + threadIdx.y;
  index = outputY * height + outputX;
  __syncthreads();

  if ((outputX < height) && (outputY < width)) {
    outputImage[index] = inputData[IMUL(blockDim.y - threadIdx.x - 1, 17) + threadIdx.y];
  }  
}

__global__ void transposeImage(int width, int height, int* inputImage, int* outputImage) {
  __shared__ int inputData[17*16];
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int index = y * width + x;
  if ((x < width) && (y < height)) {
    inputData[IMUL(threadIdx.y, 17) + threadIdx.x] = inputImage[index];
  } else {
    inputData[IMUL(threadIdx.y, 17) + threadIdx.x] = 0;
  }
  int outputX = blockDim.y * blockIdx.y + threadIdx.x;
  int outputY = blockDim.x * blockIdx.x + threadIdx.y;
  index = outputY * height + outputX;
  __syncthreads();
 
 
  if ((outputX < height) && (outputY < width)) {
    outputImage[index] = inputData[IMUL(threadIdx.x, 17) + threadIdx.y];
  }
  
}

__global__ void transposeImage(int width, int height, int* inputImage, int inputImagePitchInInts, int* outputImage, int outputImagePitchInInts) {
  __shared__ int inputData[17*16];
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int index = y * inputImagePitchInInts + x;
  if ((x < width) && (y < height)) {
    inputData[IMUL(threadIdx.y, 17) + threadIdx.x] = inputImage[index];
  } else {
    inputData[IMUL(threadIdx.y, 17) + threadIdx.x] = 0;
  }
  int outputX = blockDim.y * blockIdx.y + threadIdx.x;
  int outputY = blockDim.x * blockIdx.x + threadIdx.y;
  index = outputY * outputImagePitchInInts + outputX;
  __syncthreads();
 
 
  if ((outputX < height) && (outputY < width)) {
    outputImage[index] = inputData[IMUL(threadIdx.x, 17) + threadIdx.y];
  }
  
}


__global__ void fillImage(int width, int height, int value, int* devOutput) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int index = y * width + x;
  if ((y < height) && (x < width)) {
    devOutput[index] = value;
  }
}

__global__ void bresenhamRotate(int width, int height, int* devImage, int outputWidth, int* devOutput) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int index = y * width + x;
  if ((x < width) && (y < height)) {
    int value = devImage[index];
    int newX = guideLineX[y] + baseLineX[x];
    int newY = guideLineY[y] + baseLineY[x];
    int outputIndex = newY * outputWidth + newX;
    devOutput[outputIndex] = value;
    
  }
}


#define UNROLL 4
template<int nthreads, int nbins, bool blur, bool sense>
  __global__ void computeGradient(int width, int height, int nPixels, int border, int rotatedWidth, float aNorm, float bNorm, int kernelRadius, int kernelLength, int* devIntegrals, int integralImagePitch, float* devGradientA) {
  __shared__ float aHistogram[nthreads*UNROLL];
  __shared__ float bHistogram[nthreads*UNROLL];
  __shared__ float temp[nthreads*UNROLL];
 
  
  int x = blockIdx.x;
  int internalX = x + border;
  int bin = threadIdx.y * nthreads + threadIdx.x + kernelRadius;
  for(int y = threadIdx.y; y < height; y += UNROLL) {
    int internalY = y + border;
    int rotatedX = guideLineX[internalY] + baseLineX[internalX];
    int rotatedY = guideLineY[internalY] + baseLineY[internalX];
    int pixelIndex = rotatedY * rotatedWidth + rotatedX;

    
    if (threadIdx.x < nbins) {
      if (sense == false) {
        int pixelIndexA = (rectangleOffsets[0] + pixelIndex) * integralImagePitch + threadIdx.x;
        int pixelA = devIntegrals[pixelIndexA];
        
        int pixelIndexC = (rectangleOffsets[2] + pixelIndex) * integralImagePitch + threadIdx.x;
        int pixelC = devIntegrals[pixelIndexC];
        
        int pixelIndexF = (rectangleOffsets[5] + pixelIndex) * integralImagePitch + threadIdx.x;
        int pixelF = devIntegrals[pixelIndexF];
        
        int pixelIndexH = (rectangleOffsets[7] + pixelIndex) * integralImagePitch + threadIdx.x;
        int pixelH = devIntegrals[pixelIndexH];
        
        aHistogram[bin] = pixelA - pixelC;
        bHistogram[bin] = pixelH - pixelF;
        
        int pixelIndexD = (rectangleOffsets[3] + pixelIndex) * integralImagePitch + threadIdx.x;
        int pixelD = devIntegrals[pixelIndexD];
        
        int pixelIndexE = (rectangleOffsets[4] + pixelIndex) * integralImagePitch + threadIdx.x;
        int pixelE = devIntegrals[pixelIndexE];
        
        aHistogram[bin] += pixelE - pixelD;
        //aHistogram[bin] *= aNorm;
        
        bHistogram[bin] += pixelD - pixelE;
        //bHistogram[bin] *= bNorm;
      } else {
        int pixelIndexA = (rectangleOffsets[0] + pixelIndex) * integralImagePitch + threadIdx.x;
        int pixelA = devIntegrals[pixelIndexA];
        
        int pixelIndexC = (rectangleOffsets[2] + pixelIndex) * integralImagePitch + threadIdx.x;
        int pixelC = devIntegrals[pixelIndexC];
        
        int pixelIndexF = (rectangleOffsets[5] + pixelIndex) * integralImagePitch + threadIdx.x;
        int pixelF = devIntegrals[pixelIndexF];
        
        int pixelIndexH = (rectangleOffsets[7] + pixelIndex) * integralImagePitch + threadIdx.x;
        int pixelH = devIntegrals[pixelIndexH];

        aHistogram[bin] = pixelA - pixelF;
        bHistogram[bin] = pixelH - pixelC;
                
        int pixelIndexB = (rectangleOffsets[1] + pixelIndex) * integralImagePitch + threadIdx.x;
        int pixelB = devIntegrals[pixelIndexB];
        
        int pixelIndexG = (rectangleOffsets[6] + pixelIndex) * integralImagePitch + threadIdx.x;
        int pixelG = devIntegrals[pixelIndexG];

        aHistogram[bin] += pixelG - pixelB;
        //aHistogram[bin] *= aNorm;
        
        bHistogram[bin] += pixelB - pixelG;
        //bHistogram[bin] *= bNorm;

      }
    } 
    int offset = IMUL(threadIdx.y, nthreads);
    if (threadIdx.x < kernelRadius) {
      int zeroBinIndex = threadIdx.x + offset;
      aHistogram[zeroBinIndex] = 0;
      bHistogram[zeroBinIndex] = 0;
    }
    if (threadIdx.x < nthreads - nbins - kernelRadius) {
      int zeroBinIndex = threadIdx.x + kernelRadius + nbins + offset;
      aHistogram[zeroBinIndex] = 0;
      bHistogram[zeroBinIndex] = 0;
    }
    
    __syncthreads();
    
    if (blur) {
      float convA = 0;
      float convB = 0;
      if (threadIdx.x < nbins) {

        int nb = offset + threadIdx.x;
        for (int nk=0; nk < kernelLength;nk++) {
          float coefficient = blurKernel[nk];
          convA += aHistogram[nb] * coefficient;
          convB += bHistogram[nb] * coefficient;
          nb++;
        }
 
      
        __syncthreads();
        aHistogram[bin] = convA;
        bHistogram[bin] = convB;
      }

    }
    __syncthreads();
    temp[bin] = aHistogram[bin];
    if (threadIdx.x < nthreads - blockDim.x) {
      temp[bin + blockDim.x] = 0;
    }
    __syncthreads();
    int warpIdx = threadIdx.x & 0x1f;
    if (warpIdx < 16) temp[bin] = temp[bin] + temp[bin + 16];
    if (warpIdx <  8) temp[bin] = temp[bin] + temp[bin +  8];
    if (warpIdx <  4) temp[bin] = temp[bin] + temp[bin +  4];
    if (warpIdx <  2) temp[bin] = temp[bin] + temp[bin +  2];
    if (warpIdx <  1) temp[bin] = temp[bin] + temp[bin +  1];
    __syncthreads();
    if (nbins > 32) {
      if (threadIdx.x == 0) {
        temp[bin] = temp[bin] + temp[bin + 32];
      }
    }
    if (threadIdx.x == 0) {
      if (temp[bin] != 0) {
        temp[bin] = 8.0/temp[bin];
      }
    }
    __syncthreads();
    aHistogram[bin] = aHistogram[bin] * temp[kernelRadius + nthreads * threadIdx.y];
    __syncthreads();
    temp[bin] = bHistogram[bin];
    if (threadIdx.x < nthreads - blockDim.x) {
      temp[bin + blockDim.x] = 0;
    }
    __syncthreads();
    warpIdx = threadIdx.x & 0x1f;
    if (warpIdx < 16) temp[bin] = temp[bin] + temp[bin + 16];
    if (warpIdx <  8) temp[bin] = temp[bin] + temp[bin +  8];
    if (warpIdx <  4) temp[bin] = temp[bin] + temp[bin +  4];
    if (warpIdx <  2) temp[bin] = temp[bin] + temp[bin +  2];
    if (warpIdx <  1) temp[bin] = temp[bin] + temp[bin +  1];
    __syncthreads();
    if (nbins > 32) {
      if (threadIdx.x == 0) {
        temp[bin] = temp[bin] + temp[bin + 32];
      }
    }
    if (threadIdx.x == 0) {
      if (temp[bin] != 0) {
        temp[bin] = 8.0/temp[bin];
      }
    }
    __syncthreads();
    bHistogram[bin] = bHistogram[bin] * temp[kernelRadius + nthreads * threadIdx.y];
    __syncthreads();    
    
/*     if (threadIdx.x == 0) { */
/*       float asum = 0; */
/*       float bsum = 0; */
/*       const int begin = kernelRadius + threadIdx.y * nthreads; */
/*       const int end = begin + nbins; */
/*       for(int i = begin; i < end; i++) { */
/*         asum += aHistogram[i]; */
/*         bsum += bHistogram[i]; */
/*       } */
/*       if (asum > 0) { */
/*         asum = 8.0/asum; */
/*       } */
/*       if (bsum > 0) { */
/*         bsum = 8.0/bsum; */
/*       } */
/*       aNorms[threadIdx.y] = asum; */
/*       bNorms[threadIdx.y] = bsum; */
/*     } */
/*     __syncthreads(); */
/*     aHistogram[bin] *= aNorms[threadIdx.y]; */
/*     bHistogram[bin] *= bNorms[threadIdx.y]; */

    float sum = aHistogram[bin] + bHistogram[bin];
    float chiVal = 0.0f;
    if (sum != 0) {
      float diff = aHistogram[bin] - bHistogram[bin];
      chiVal = diff * diff / sum;
    }
    __syncthreads();
    float* chi = bHistogram;
    chi[bin] = chiVal;
    __syncthreads();
    
    warpIdx = threadIdx.x & 0x1f;
    if (warpIdx < 16) chi[bin] = chi[bin] + chi[bin + 16];
    if (warpIdx <  8) chi[bin] = chi[bin] + chi[bin +  8];
    if (warpIdx <  4) chi[bin] = chi[bin] + chi[bin +  4];
    if (warpIdx <  2) chi[bin] = chi[bin] + chi[bin +  2];
    if (warpIdx <  1) chi[bin] = chi[bin] + chi[bin +  1];
    __syncthreads();
    if (nbins > 32) {
      if (threadIdx.x == 0) {
        chi[bin] = chi[bin] + chi[bin + 32];
      }
    }
    if (threadIdx.x == 0) {
      int idx = x + y * width;
      float output = 0.5 * chi[bin];
      devGradientA[idx] = output;
    }
  }
}

/**
 * For a given orientation, dispatches the gradient computation
 */
void dispatchGradient(int width, int height, int border, int nbins, float thetaPi, int rotatedWidth, int radius, bool blur, float sigma, int* devIntegrals, int integralImagePitchInInts, float* devGradientA, float* devGradientB) {

  float theta = thetaPi * M_PIl;
 
  

  
  float effectiveRadius = (-4.0+4.0*float(radius)*sqrtf(M_PIl))/8.0;
  
  int x2 = (int)round(-effectiveRadius*sin(theta)-effectiveRadius*cos(theta));
  int y2 = (int)round(-effectiveRadius*sin(theta)+effectiveRadius*cos(theta));
  int x3 = (int)round(effectiveRadius*sin(theta)+effectiveRadius*cos(theta));
  int y3 = (int)round(effectiveRadius*sin(theta)-effectiveRadius*cos(theta));
  //printf("Radius: %d, effectiveRadius: %f\n", radius, effectiveRadius);
  //printf("(x2, y2): (%d, %d); (x3, y3): (%d, %d)\n", x2, y2, x3, y3);

  int xc = width/2;
  int yc = height/2;
  
  int x2c = x2 + xc;
  int y2c = y2 + yc;
  int x3c = x3 + xc;
  int y3c = y3 + yc;
  
  int x2rc = hostGuideLineX[y2c] + hostBaseLineX[x2c];
  int y2rc = hostGuideLineY[y2c] + hostBaseLineY[x2c];
  int x3rc = hostGuideLineX[y3c] + hostBaseLineX[x3c];
  int y3rc = hostGuideLineY[y3c] + hostBaseLineY[x3c];

  int xcr = hostGuideLineX[yc] + hostBaseLineX[xc];
  int ycr = hostGuideLineY[yc] + hostBaseLineY[xc];

  int x2r = x2rc - xcr;
  int y2r = y2rc - ycr;
  int x3r = x3rc - xcr;
  int y3r = y3rc - ycr;
  
  
  //printf("(x2r, y2r) : (%d, %d); (x3r, y3r): (%d, %d)\n", x2r, y2r, x3r, y3r);

  int xDiameter = x3r - x2r + 1;
  int yDiameter = y2r - y3r + 1;
  int xRadius = xDiameter / 2;
  int yRadius = yDiameter / 2;

  
  int x[8];
  int y[8];
  int topWidth = xDiameter;
  int topHeight;
  int bottomWidth = xDiameter;
  int bottomHeight;
  int leftWidth;
  int leftHeight = yDiameter;
  int rightWidth;
  int rightHeight = yDiameter;
  
  if (xDiameter & 0x1) {
    x[0] = -xRadius - 1;
    x[1] = 0;
    x[2] = xRadius;
    x[3] = -xRadius - 1;
    x[4] = xRadius;
    x[5] = -xRadius - 1;
    x[6] = 0;
    x[7] = xRadius;
    leftWidth = xRadius + 1;
    rightWidth = xRadius;
  } else {
    x[0] = -xRadius;
    x[1] = 0;
    x[2] = xRadius;
    x[3] = -xRadius;
    x[4] = xRadius;
    x[5] = -xRadius;
    x[6] = 0;
    x[7] = xRadius;
    leftWidth = xRadius;
    rightWidth = xRadius;
  }

  if (yDiameter & 0x1) {
    y[0] = -yRadius - 1;
    y[1] = -yRadius - 1;
    y[2] = -yRadius - 1;
    y[3] = 0;
    y[4] = 0;
    y[5] = yRadius;
    y[6] = yRadius;
    y[7] = yRadius;
    topHeight = yRadius + 1;
    bottomHeight = yRadius;
  } else {
    y[0] = -yRadius;
    y[1] = -yRadius;
    y[2] = -yRadius;
    y[3] = 0;
    y[4] = 0;
    y[5] = yRadius;
    y[6] = yRadius;
    y[7] = yRadius;
    topHeight = yRadius;
    bottomHeight = yRadius;
  }

  float cosTheta2 = cos(theta) * cos(theta);
  float topArea = floor(float(topWidth) * float(topHeight) * cosTheta2);
  float bottomArea = floor(float(bottomWidth) * float(bottomHeight) * cosTheta2);
  float leftArea = floor(float(leftWidth) * float(leftHeight) * cosTheta2);
  float rightArea = floor(float(rightWidth) * float(rightHeight) * cosTheta2);
/*   for(int i = 0; i < 8; i++) { */
/*      printf("i = %d: (%d, %d)\n", i, x[i], y[i]); */
/*   } */
  //printf("topArea: %3.0f, bottomArea: %3.0f, leftArea: %3.0f, rightArea: %3.0f\n", topArea, bottomArea, leftArea, rightArea);
/*   printf("topWidth: %d, topHeight: %d\n", topWidth, topHeight); */
/*   printf("bottomWidth: %d, bottomHeight: %d\n", bottomWidth, bottomHeight); */
/*   printf("leftWidth: %d, leftHeight: %d\n", leftWidth, leftHeight); */
/*   printf("rightWidth: %d, rightHeight: %d\n", rightWidth, rightHeight); */
  float topNorm = 1/topArea;
  float bottomNorm = 1/bottomArea;
  float rightNorm = 1/rightArea;
  float leftNorm = 1/leftArea;
  
  
  
  dim3 gridDim = dim3(width, 1);
  dim3 blockDim = dim3(nbins, UNROLL);
  size_t sharedMemoryPerThread = 3 * UNROLL * sizeof(float);
 
  int hostRectangleOffsets[8];
  for(int i = 0; i < 8; i++) {
    hostRectangleOffsets[i] = y[i] * rotatedWidth + x[i];
  }

  cudaMemcpyToSymbol(rectangleOffsets, hostRectangleOffsets, sizeof(int) * 8);


  
  if (nbins == 25) {
    blockDim = dim3(32, UNROLL);
    const int nThreads = 48;
    int kernelRadius;
    int kernelLength;
    //float sigma = 0.10 * float(nbins);
    float* hostKernel;
    precompute_gaussian(sigma, kernelRadius, kernelLength, &hostKernel);


   
    
    //printf("Kernel length: %d, radius = %d\n", kernelLength, kernelRadius);
    cudaMemcpyToSymbol(blurKernel, hostKernel, sizeof(float) * kernelLength);

    computeGradient<nThreads, 25, true, false><<<gridDim, blockDim, nThreads * sharedMemoryPerThread>>>(width, height, width * height, border, rotatedWidth, topNorm, bottomNorm, kernelRadius, kernelLength, devIntegrals, integralImagePitchInInts, devGradientA);
    computeGradient<nThreads, 25, true, true><<<gridDim, blockDim, nThreads * sharedMemoryPerThread>>>(width, height, width * height, border, rotatedWidth, leftNorm, rightNorm, kernelRadius, kernelLength, devIntegrals, integralImagePitchInInts, devGradientB);
  } else if (nbins == 64) {
    const int nThreads = 64;
    computeGradient<nThreads, 64, false, false><<<gridDim, blockDim, nThreads * sharedMemoryPerThread>>>(width, height, width * height, border, rotatedWidth, topNorm, bottomNorm, 0, 0, devIntegrals, integralImagePitchInInts, devGradientA);
    computeGradient<nThreads, 64, false, true><<<gridDim, blockDim, nThreads * sharedMemoryPerThread>>>(width, height, width * height, border, rotatedWidth, leftNorm, rightNorm, 0, 0, devIntegrals, integralImagePitchInInts, devGradientB);
  }
}


void rotateImage(int width, int height, int* devImage, float thetaPi, int& newWidth, int& newHeight, int* devOutput) {
    
  
  assert ((thetaPi <= 0) && (thetaPi > -.5));
  
  if (thetaPi == 0) {
    
    for(int i = 0; i < width; i++) {
      hostBaseLineX[i] = i;
      hostBaseLineY[i] = 0;
    }
    for(int i = 0; i < height; i++) {
      hostGuideLineX[i] = 0;
      hostGuideLineY[i] = i;
    }
    newWidth = width;
    newHeight = height;
  } else if ((thetaPi < 0) && (thetaPi >= -.25)) {
    float theta = thetaPi * M_PIl;
    float tanTheta = tanf(theta);
    newWidth = ceilf(width - (height - 1) * tanTheta);
    newHeight = ceilf(height - (width - 1) * tanTheta);
    int length;
    int yDescent = -round(((float)(width - 1)) * tanTheta);
    int xTravel = -round(((float)(height - 1)) * tanTheta);
    bresenhamLine(0, 0, width-1, yDescent, length, hostBaseLineX, hostBaseLineY);
    assert(length == width);
    bresenhamLine(xTravel, 0, 0, height-1, length, hostGuideLineX, hostGuideLineY);
    assert(length == height);
  } else if ((thetaPi < -.25) && (thetaPi > -.5)) {
    float theta = (thetaPi + 0.25) * M_PIl;
    float tanTheta = tanf(theta);
    newWidth = ceilf(height - (width - 1) * tanTheta);
    newHeight = ceilf(width - (height - 1) * tanTheta);
    int length;
    int xTravel = -round(((float)(width - 1)) * tanTheta);
    int yTravel = -round(((float)(height - 1)) * tanTheta);
    /* printf("(%d, %d) -> (%d, %d): ", 0, 0, xTravel, width - 1); */
    bresenhamLine(0, 0, xTravel, width - 1, length, hostBaseLineX, hostBaseLineY);
    /*     printf("%d\n", length); */
    assert(length == width);
    /* printf("(%d, %d) -> (%d, %d): ", height - 1, 0, 0, yTravel); */
    bresenhamLine(height - 1, 0, 0, yTravel, length, hostGuideLineX, hostGuideLineY);
    /*     printf("%d\n", length); */
    assert(length == height);
  }
  cudaMemcpyToSymbol(baseLineX, hostBaseLineX, width * sizeof(int));
  cudaMemcpyToSymbol(baseLineY, hostBaseLineY, width * sizeof(int));
  cudaMemcpyToSymbol(guideLineX, hostGuideLineX, height * sizeof(int));
  cudaMemcpyToSymbol(guideLineY, hostGuideLineY, height * sizeof(int));
  /*  for(int i = 0; i < width; i++) { */
  /*     printf("(%d, %d)\n", hostBaseLineX[i], hostBaseLineY[i]); */
  /*   } */
  /*   printf("\n"); */
  /*   for(int i = 0; i < height; i++) { */
  /*     printf("(%d, %d)\n", hostGuideLineX[i], hostGuideLineY[i]); */
  /*   } */
  
  dim3 gridDim = dim3((newWidth-1)/16+1, (newHeight-1)/16+1);
  fillImage<<<gridDim, dim3(16, 16)>>>(newWidth, newHeight, -1, devOutput);

  
  gridDim = dim3((width-1)/16 + 1, (height-1)/16 + 1);
  
  bresenhamRotate<<<gridDim, dim3(16,16)>>>(width, height, devImage, newWidth, devOutput);
}

