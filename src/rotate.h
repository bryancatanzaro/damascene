#ifndef ROTATE
#define ROTATE

__global__ void turnImageP(int width, int height, int* inputImage, int* outputImage);

__global__ void turnImageN(int width, int height, int* inputImage, int* outputImage);


__global__ void transposeImage(int width, int height, int* inputImage, int inputImagePitchInInts, int* outputImage, int outputImagePitchInInts);

__global__ void transposeImage(int width, int height, int* inputImage, int* outputImage);

__global__ void fillImage(int width, int height, int value, int* devOutput);

__global__ void bresenhamRotate(int width, int height, int* devImage, int outputWidth, int* devOutput);

void rotateImage(int width, int height, int* devImage, float thetaPi, int& newWidth, int& newHeight, int* devOutput);

void dispatchGradient(bool tg, int width, int height, int border, int nbins, float theta, int rotatedWidth, int radius, bool blur, float sigma, int* integralImages, int integralImagePitch, float* devGradientA, float* devGradientB);

void dispatchGradient_64(int width, int height, int border, int nbins, float theta, int rotatedWidth, int radius, bool blur, float sigma, int* integralImages, int integralImagePitch, float* devGradientA, float* devGradientB);

#endif

