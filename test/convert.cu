#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "convert.h"
#include "stencilMVM.h"
#include <cutil.h>

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

void writeTextImage(const char* filename, uint width, uint height, int* image) {
  FILE* fp = fopen(filename, "w");
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      fprintf(fp, "%d ", image[row * width + col]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

int main(int argc, char** argv) {
  cuInit(0);
  chooseLargestGPU(true);

  char* filename = "colors.ppm";
  unsigned int width;
  unsigned int height;
  unsigned int* devRgbU;
  loadPPM_rgbU(filename, &width, &height, &devRgbU);
  printf("Width: %i, height: %i\n", width, height, devRgbU);
  float* devL;
  float* devA;
  float* devB;
  float* devGrey;
  rgbUtoLab3F(width, height, 2.5, devRgbU, &devL, &devA, &devB);
  rgbUtoGreyF(width, height, devRgbU, &devGrey);
  normalizeLab(width, height, devL, devA, devB);
  float* L = getImage(width, height, devL);
  float* a = getImage(width, height, devA);
  float* b = getImage(width, height, devB);
  float* grey = getImage(width, height, devGrey);
  writeTextImage("L.txt", width, height, L);
  writeTextImage("a.txt", width, height, a);
  writeTextImage("b.txt", width, height, b);
  writeTextImage("grey.txt", width, height, grey);
  int border = 2;
  float* devBorderedGrey;
  mirrorImage(width, height, border, devGrey, &devBorderedGrey);
  int borderWidth = 2 * border + width;
  int borderHeight = 2 * border + height;
  float* borderedGrey = getImage(borderWidth, borderHeight, devBorderedGrey);
  writeTextImage("borderedGrey.txt", borderWidth, borderHeight, borderedGrey);
  cutSavePGMf("borderedGrey.pgm", borderedGrey, borderWidth, borderHeight);
  cutSavePGMf("grey.pgm", grey, width, height);
  int* devQuantizedBorderedGrey;
  quantizeImage(borderWidth, borderHeight, 25, devBorderedGrey, &devQuantizedBorderedGrey);
  int* quantizedBorderedGrey = getImage(borderWidth, borderHeight, devQuantizedBorderedGrey);
  writeTextImage("quantizedBorderedGrey.txt", borderWidth, borderHeight, quantizedBorderedGrey);
}
