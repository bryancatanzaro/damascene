#include <cuda.h>
#include <helper_cuda.h>
#include <helper_image.h>
#include <helper_timer.h>
#include <stdio.h>
#include <cublas.h>
#include "texton.h"

                                                               

                              

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


int main(int argc, char** argv) {
  chooseLargestGPU(true);
  printf("Loading image...");
  char* filename = "polynesia.pgm";
  //char* filename = "tiny.pgm";
  float* hostImage = 0;
  unsigned int width;
  unsigned int height;
  sdkLoadPGM(filename, &hostImage, &width, &height);
  int nPixels = width * height;
  printf("width = %i, height = %i\n", width, height);
  float* devImage;
  cudaMalloc((void**)&devImage, sizeof(float) * nPixels);
  cudaMemcpy(devImage, hostImage, sizeof(float) * nPixels, cudaMemcpyHostToDevice);
  int* devClusters;
  StopWatchInterface *textonTimer=NULL;
  sdkCreateTimer(&textonTimer);
  sdkStartTimer(&textonTimer);

  findTextons(width, height, devImage, &devClusters, 1);
  
/*   printf("Setting up\n"); */
/*   dim3 gridDim = dim3((width - 1)/XBLOCK + 1, (height - 1)/YBLOCK + 1); */
/*   dim3 blockDim = dim3(XBLOCK, YBLOCK); */
/* /\*   printf("gridDim: %i, %i; blockDim: %i, %i\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y); *\/ */
/*   int filterCount = 34; */
/*   int clusterCount = 64; */
/*   int nPixels = width * height; */
/*   float* devResponses; */
/*   cudaMalloc((void**)&devResponses, sizeof(float)*nPixels*filterCount); */
/*   cudaMemcpyToSymbol(radii, hRadii, sizeof(hRadii)); */
/*   cudaMemcpyToSymbol(coefficients, hCoefficients, sizeof(hCoefficients)); */
/*   cudaArray* imageArray; */
/*   cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>(); */
/*   cudaMallocArray(&imageArray, &floatTex, width, height); */
/*   cudaMemcpyToArray(imageArray, 0, 0, hostImage, nPixels * sizeof(float), cudaMemcpyHostToDevice); */
/*   cudaBindTextureToArray(image, imageArray); */
/*   printf("Convolving\n"); */
/*   convolve<<<gridDim, blockDim>>>(filterCount, nPixels, width, height, devResponses); */
 
  //float* responses = (float*)malloc(sizeof(float)*nPixels*filterCount);
  //cudaMemcpy(responses, devResponses, sizeof(float)*nPixels*filterCount, cudaMemcpyDeviceToHost);
 /*  printf("Writing filter responses...\n"); */
/*   float min = 1000000; */
/*   float max = -1000000; */
/*   for(int i = 0; i < nPixels * filterCount; i++) { */
/*     if (responses[i] < min) { */
/*       min = responses[i]; */
/*     } */
/*     if (responses[i] > max) { */
/*       max = responses[i]; */
/*     } */
/*   } */
/*   //printf("Min: %f, Max: %f\n", min, max); */
/*   for (int i = 0; i < nPixels * filterCount; i++) { */
/*     responses[i] = (responses[i] - min)/(max - min); */
/*   } */
/*   char* outputFilename = (char*)malloc(sizeof(char) * 80); */
/*   strcpy(outputFilename, "polynesia00.pgm"); */
/*   for(int i = 0; i < filterCount; i++) { */
/*     sprintf(&outputFilename[9], "%02u", i); */
/*     outputFilename[11] = '.'; */
/*     //printf("%s\n", outputFilename); */
/*     cutSavePGMf(outputFilename, &responses[i * nPixels], width, height); */
/*   } */
  
  //int* devClusters;
  //kmeans(nPixels, width, height, clusterCount, filterCount, devResponses, &devClusters);
/*   cudaMalloc((void**)&clusters, sizeof(int)*nPixels); */
  
/*   //__global__ void assignInitialClusters(int width, int height, int nPixels, int clusterCount, int* cluster, int filterCount, float* responses, int* intResponses) { */

/*   int* intResponses; */
/*   cudaMalloc((void**)&intResponses, sizeof(int) * nPixels * filterCount); */
/*   assignInitialClusters<<<gridDim, blockDim>>>(width, height, nPixels, clusterCount, clusters, filterCount, devResponses, intResponses); */

  int* hostClusters = (int*)malloc(sizeof(int)*nPixels);
/*   cudaMemcpy(hostClusters, clusters, sizeof(int) * nPixels, cudaMemcpyDeviceToHost); */
  unsigned char* hostClustersUb = (unsigned char*)malloc(sizeof(unsigned char) * nPixels);
/*   for(int i = 0; i < nPixels; i++) { */
/*     hostClustersUb[i] = (unsigned char)hostClusters[i] * 4; */
/*   } */
/*   cutSavePGMub("clusters.pgm", hostClustersUb, width, height); */
  
/*   dim3 linearGrid = dim3((width * height - 1)/512 + 1); */
/*   dim3 linearBlock = dim3(512); */

/*   dim3 clusterGrid = dim3((filterCount - 1)/XBLOCK + 1, (clusterCount - 1)/YBLOCK + 1); */
/*   dim3 clusterBlock = dim3(XBLOCK, YBLOCK); */
  
/*   int* centroidMass; */
/*   cudaMalloc((void**)&centroidMass, sizeof(int) * filterCount * clusterCount); */
/*   unsigned int* centroidCount; */
/*   cudaMalloc((void**)&centroidCount, sizeof(unsigned int) * clusterCount); */
/*   float* centroids; */
/*   cudaMalloc((void**)&centroids, sizeof(float) * filterCount * clusterCount); */
/*   int* changes; */
/*   cudaMalloc((void**)&changes, sizeof(int)); */
/*   int i; */

/*   float* pointsDots; */
/*   cudaMalloc((void**)&pointsDots, sizeof(int) * nPixels); */
/*   float* centroidsDots; */
/*   cudaMalloc((void**)&centroidsDots, sizeof(int) * clusterCount); */
/*   makeSelfDots<<<linearGrid, linearBlock>>>(devResponses, nPixels, pointsDots, nPixels, filterCount); */

/*   float* devDist; */
/*   size_t devDistPitch; */
/*   cudaMallocPitch((void**)&devDist, &devDistPitch, sizeof(float) * nPixels, clusterCount); */
/*   int devDistPitchInFloats = devDistPitch/sizeof(float); */

  
/*   for(i = 0; i < 10; i++) { */
/*     cudaMemset(centroidMass, 0, sizeof(int) * filterCount * clusterCount); */
/*     cudaMemset(centroidCount, 0, sizeof(int) * clusterCount); */
/*     cudaMemset(changes, 0, sizeof(int)); */
/*     findCentroids<<<linearGrid, linearBlock>>>(intResponses, nPixels, clusters, centroidMass, centroidCount); */
/*                /\*  int* hostMass = (int*)malloc(sizeof(int) * filterCount * clusterCount); *\/ */
/* /\*     int* hostCount = (int*)malloc(sizeof(int) * clusterCount); *\/ */
/* /\*     cudaMemcpy(hostMass, centroidMass, sizeof(int) * filterCount * clusterCount, cudaMemcpyDeviceToHost); *\/ */
/* /\*     cudaMemcpy(hostCount, centroidCount, sizeof(int) * clusterCount, cudaMemcpyDeviceToHost); *\/ */
    
/*     finishCentroids<<<clusterGrid, clusterBlock>>>(centroidMass, centroidCount, centroids); */
/*     findSgemmLabels(devResponses, nPixels, nPixels, centroids, clusterCount, clusterCount, filterCount, pointsDots, centroidsDots, devDist, devDistPitchInFloats, clusters, changes); */
/*     //findLabels<<<linearGrid, linearBlock>>>(nPixels, filterCount, clusterCount, devResponses, centroids, clusters, changes); */
/*     int hostChanges = 0; */
/*     cudaMemcpy(&hostChanges, changes, sizeof(int), cudaMemcpyDeviceToHost); */
/*     printf("Changes: %d\n", hostChanges); */
/*     if (hostChanges == 0) { */
/*       break; */
/*     } */
/*   } */
/*   printf("%i iterations until convergence\n", i); */
  sdkStopTimer(&textonTimer);
  printf("Texton time: %f\n", sdkGetTimerValue(&textonTimer));
  sdkDeleteTimer(&textonTimer);
  cudaMemcpy(hostClusters, devClusters, sizeof(int) * nPixels, cudaMemcpyDeviceToHost);
  for(int i = 0; i < nPixels; i++) {
    hostClustersUb[i] = (unsigned char)hostClusters[i] * 4;
  }
  sdkSavePGM("newClusters.pgm", hostClustersUb, width, height);
  /* float* sgemmResult; */
/*   cudaMalloc((void**)&sgemmResult, sizeof(float) * nPixels * clusterCount); */
/*   cublasSgemm('n', 't', nPixels, filterCount, clusterCount, 1.0f, devResponses, nPixels, centroids, clusterCount, 0.0f, sgemmResult, nPixels); */
 
/*   FILE* fp; */
/*   fp = fopen("iterationTimes.txt", "w"); */
/*   for (int j = 0; j < i; j++) { */
/*     fprintf(fp, "%i ", j); */
/*     floatVector* currentIteration = times[j]; */
/*     for(std::vector<float>::iterator it = currentIteration->begin(); it != currentIteration->end(); it++) { */
/*       fprintf(fp, "%e ", *it); */
/*     } */
/*     fprintf(fp, "\n"); */
/*   } */
/*   fclose(fp); */
  
  FILE* fp;
  fp = fopen("newClusters.txt", "w");
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      fprintf(fp, "%i ", hostClusters[col + row * width]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  //testSgemm();
}
