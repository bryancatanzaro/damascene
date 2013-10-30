#include <cuda.h>
#include "../include/kmeans.h"
#include <cublas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void assignInitialClusters(int width, int height, int nPixels, int clusterCount, int* cluster, int filterCount, float* responses, int* intResponses) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int pixel = y * width + x;
  if ((x < width) && (y < height)) {
    int xBlock = x / ((width - 1) / 8 + 1);
    int yBlock = y / ((height - 1) / 8 + 1);
    int assignedCluster = yBlock * 8 + xBlock;

    cluster[y * width + x] = assignedCluster;
    for(int i = 0; i < filterCount; i++) {
      int index = pixel + i * nPixels;
      int response = (int)(INTCONFACTOR * responses[index]);
      intResponses[index] = response;
    }
  }
}

// yunsup fixed
__global__ void findCentroidsAtomicFreeLocal(int afLocal, int* responses, int nPixels, int* cluster, int* centroidMass, unsigned int* centroidCount)
{
    int const af_id = blockIdx.x;
    int const cluster_id = blockIdx.y;
    int const filter_id = threadIdx.x;
    int* filter_responses = &responses[filter_id*nPixels];

    int local_responses = 0;
    int local_count = 0;

    int pixel_start = af_id*afLocal;
    int pixel_end = (af_id+1)*afLocal;

    pixel_end = pixel_end>nPixels?nPixels:pixel_end;

    for (int i=pixel_start; i<pixel_end; i++)
    {
        if (cluster[i] == cluster_id)
        {
            local_responses += filter_responses[i];
            local_count++;
        }
    }

    int idx = af_id * gridDim.y*blockDim.x + filter_id*64 + cluster_id;
    centroidMass[idx] = local_responses;
    centroidCount[idx] = local_count;
}

__global__ void findCentroidsAtomicFreeReduce(int afLocal, int* responses, int nPixels, int* cluster, int* centroidMass, unsigned int* centroidCount)
{
    int const af_id = blockIdx.x;
    int const cluster_id = blockIdx.y;
    int const filter_id = threadIdx.x;

    int local_mass = 0;
    int local_count = 0;

    if (af_id == 0)
    {
        int idx0 = filter_id*64 + cluster_id;

        for (int i=0; i<gridDim.x; i++)
        {
            int idxother = i * gridDim.y*blockDim.x + idx0;

            local_mass += centroidMass[idxother];
            local_count += centroidCount[idxother];
        }

        centroidMass[idx0] = local_mass;
        centroidCount[idx0] = local_count;
    }
}


__global__ void finishCentroids(int* centroidMass, unsigned int* centroidCount, float* centroids) {
  int centroidNumber = blockIdx.y * blockDim.y + threadIdx.y;
  int dimensionNumber = blockIdx.x * blockDim.x + threadIdx.x;
  if ((centroidNumber < 64) && (dimensionNumber < 34)) {
    float totalCount = (float)centroidCount[centroidNumber];
    float mass = (float)centroidMass[dimensionNumber * 64 + centroidNumber];
    centroids[dimensionNumber * 64 + centroidNumber] = mass / ((float)INTCONFACTOR * totalCount);
  }
}


/**
 * This function computes self dot products (Euclidean norm squared) for every vector in an array
 * @param devSource the vectors, in column major format
 * @param devSourcePitchInFloats the pitch of each row of the vectors (this is guaranteed to be >= sourceCount.  It might be greater due to padding, to keep each row of the source vectors aligned.
 * @param devDest a vector which will receive the self dot product
 * @param sourceCount the number of vectors
 * @param sourceLength the dimensionality of each vector
 */
__global__ void makeSelfDots(float* devSource, int devSourcePitchInFloats, float* devDest, int sourceCount, int sourceLength) {
	float dot = 0;
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < sourceCount) {
		for (int i = 0; i < sourceLength; i++) {
			float currentElement = *(devSource + IMUL(devSourcePitchInFloats, i) + index); 
			dot = dot + currentElement * currentElement;
		}
		devDest[index] = dot;
	}
}

/**
 * This function constructs a matrix devDots, where devDots_(i,j) = ||B_i||^2 + ||A_j||^2
 * @param devDots the output array
 * @param devDotsPitchInFloats the pitch of each row of devDots.  Guaranteed to be >= nA
 * @param devADots a vector containing ||A_j||^2 for all j in [0, nA - 1]
 * @param devBDots a vector containing ||B_i||^2 for all i in [0, nB - 1]
 * @param nA the number of points in A
 * @param nB the number of points in B
 */
__global__ void makeDots(float* devDots, int devDotsPitchInFloats, float* devADots, float* devBDots, int nA, int nB) {
	__shared__ float localADots[XBLOCK];
	__shared__ float localBDots[YBLOCK];
	int aIndex = IMUL(XBLOCK, blockIdx.x) + threadIdx.x;

	if ((aIndex < nA) && (threadIdx.x < XBLOCK) && (threadIdx.y == 0)) {
		localADots[threadIdx.x] = devADots[aIndex];
	}
	
	int bIndex = IMUL(YBLOCK, blockIdx.y) + threadIdx.x;
	if ((bIndex < nB) && (threadIdx.x < YBLOCK) && (threadIdx.y == 1)) {
		localBDots[threadIdx.x] = devBDots[bIndex];
	}
	
	__syncthreads();

	bIndex = IMUL(YBLOCK, blockIdx.y) + threadIdx.y;
  if ((aIndex < nA) && (bIndex < nB)) {
    devDots[IMUL(devDotsPitchInFloats, bIndex) + aIndex] = localADots[threadIdx.x] + localBDots[threadIdx.y];
  }
}


void findDiffNorm(float* devA, int nA, int aPitchInFloats, float* devB, int nB, int bPitchInFloats, int nDimension, float* devADots, float* devBDots, float* devDiff, int diffPitchInFloats) {
  dim3 linearGrid = dim3((nB - 1)/LINBLOCK + 1, 1);
  dim3 linearBlock = dim3(LINBLOCK, 1);
  makeSelfDots<<<linearGrid, linearBlock>>>(devB, bPitchInFloats, devBDots, nB, nDimension);
  /* float* hostClusterDots = (float*)malloc(sizeof(float)*nB); */
/*   cudaMemcpy(hostClusterDots, devBDots, sizeof(float)*nB, cudaMemcpyDeviceToHost); */
/*   printf("Printing Cluster dots: "); */
/*   for(int i = 0; i < nB; i++) { */
/*     printf("%f ", hostClusterDots[i]); */
/*   } */
/*   printf("\n"); */
  dim3 squareGrid = dim3((nA - 1)/XBLOCK + 1, (nB - 1)/YBLOCK + 1);
  dim3 squareBlock = dim3(XBLOCK, YBLOCK);
  /* printf("Dots grid: %i, %i; Dots block: %i, %i\n", squareGrid.x, squareGrid.y, squareBlock.x, squareBlock.y); */

  makeDots<<<squareGrid, squareBlock>>>(devDiff, diffPitchInFloats, devADots, devBDots, nA, nB);
/*   float* hostDots = (float*)malloc(sizeof(float) * nA * nB); */
/*   cudaMemcpy2D(hostDots, sizeof(float)*nA, devDiff, diffPitchInFloats*sizeof(float), sizeof(float)*nA, nB, cudaMemcpyDeviceToHost); */
/*   printf("Printing Dots:\n"); */
/*   for(int row = 0; row < nB; row++) { */
/*     for(int col = 0; col < nA; col++) { */
/*       printf("%f ", hostDots[col + row * nA]); */
/*     } */
/*     printf("\n"); */
/*   } */
  
  cublasSgemm('n', 't', nA, nB, nDimension, -2.0f, devA, aPitchInFloats, devB, bPitchInFloats, 1.0f, devDiff, diffPitchInFloats);
/*   printf("Printing Distances:\n"); */
/*   float* hostDiff = (float*)malloc(sizeof(float) * nA * nB); */
/*   cudaMemcpy2D(hostDiff, sizeof(float)*nA, devDiff, diffPitchInFloats*sizeof(float), sizeof(float)*nA, nB, cudaMemcpyDeviceToHost); */
/*   for(int row = 0; row < nB; row++) { */
/*     for(int col = 0; col < nA; col++) { */
/*       printf("%f ", hostDiff[col + row * nA]); */
/*     } */
/*     printf("\n"); */
/*   } */
}


__global__ void findDiffLabelsAtomicFree(float* devDiff, int diffPitchInFloats, int nPoints, int nClusters, int* devClusters, int* devChanges) {


  int x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x < nPoints) {
    int index = x;
    float minDistance = 10000000;
    int minCluster = -1;
    for(int cluster = 0; cluster < nClusters; cluster++) {
      float clusterDistance = devDiff[index];
      if (clusterDistance < minDistance) {
        minDistance = clusterDistance;
        minCluster = cluster;
      }
      index += diffPitchInFloats;
    }
    int previousCluster = devClusters[x];
    devClusters[x] = minCluster;
    if (minCluster != previousCluster) {
        //int change=*devChanges;
        //change++;
        //*devChanges = change;
    }
  }
}

void findSgemmLabels(float* devA, int nA, int aPitchInFloats, float* devB, int nB, int bPitchInFloats, int nDimension, float* devADots, float* devBDots, float* devDiff, int diffPitchInFloats, int* devClusters, int* devChanges) {
  findDiffNorm(devA, nA, aPitchInFloats, devB, nB, bPitchInFloats, nDimension, devADots, devBDots, devDiff, diffPitchInFloats);
  dim3 linearPointsGrid = dim3((nA - 1)/LINBLOCK + 1, 1);
  dim3 linearPointsBlock = dim3(LINBLOCK, 1);
  findDiffLabelsAtomicFree<<<linearPointsGrid, linearPointsBlock>>>(devDiff, diffPitchInFloats, nA, nB, devClusters, devChanges);
}

//float inc = 3.2f;
void fill(float* dest, int height, int width) {
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      dest[col + row * width] = (float)rand()/(float)RAND_MAX;
      //inc -= 0.1f;
    }
  }
}


/*
void testSgemm() {
  int nDimension = 4;
  int nPoints = 5;
  int nClusters = 3;
  srand(time(0));
  float* hostPoints = (float*)malloc(sizeof(float) * nPoints * nDimension);
  fill(hostPoints, nDimension, nPoints);
  float* devPoints;
  size_t devPointsPitch;
  cudaMallocPitch((void**)&devPoints, &devPointsPitch, nPoints * sizeof(float), nDimension);
  cudaMemcpy2D(devPoints, devPointsPitch, hostPoints, sizeof(float) * nPoints, sizeof(float) * nPoints, nDimension, cudaMemcpyHostToDevice);

  printf("Printing Points:\n");
  for(int dim = 0; dim < nDimension; dim++) {
    for(int point = 0; point < nPoints; point++) {
      printf("%f ", hostPoints[dim * nPoints + point]);
    }
    printf("\n");
  }
  
  float* hostCentroids = (float*)malloc(sizeof(float) * nClusters * nDimension);
  fill(hostCentroids, nDimension, nClusters);
  float* devCentroids;
  size_t devCentroidsPitch;
  cudaMallocPitch((void**)&devCentroids, &devCentroidsPitch, nClusters * sizeof(float), nDimension);
  cudaMemcpy2D(devCentroids, devCentroidsPitch, hostCentroids, sizeof(float) * nClusters, sizeof(float) * nClusters, nDimension, cudaMemcpyHostToDevice);

  printf("Printing Centroids:\n");
  for(int dim = 0; dim < nDimension; dim++) {
    for(int cluster = 0; cluster < nClusters; cluster++) {
      printf("%f ", hostCentroids[dim * nClusters + cluster]);
    }
    printf("\n");
  }
  
  float* devPointsDots;
  cudaMalloc((void**)&devPointsDots, sizeof(float) * nPoints);
  float* devCentroidsDots;
  cudaMalloc((void**)&devCentroidsDots, sizeof(float) * nClusters);

  float* devDiff;
  size_t devDiffPitch;
  cudaMallocPitch((void**)&devDiff, &devDiffPitch, sizeof(float) * nPoints, nClusters);
  
  dim3 linearPointsGrid = dim3((nPoints - 1)/LINBLOCK + 1, 1);
  dim3 linearPointsBlock = dim3(LINBLOCK, 1);
  makeSelfDots<<<linearPointsGrid, linearPointsBlock>>>(devPoints, devPointsPitch/sizeof(float), devPointsDots, nPoints, nDimension);

  float* hostPointsDots = (float*)malloc(sizeof(float) * nPoints);
  cudaMemcpy(hostPointsDots, devPointsDots, sizeof(float) * nPoints, cudaMemcpyDeviceToHost);
  printf("Printing Points dots: ");
  for(int i = 0; i < nPoints; i++) {
    printf("%f ", hostPointsDots[i]);
  }
  printf("\n");
  
  findDiffNorm(devPoints, nPoints, devPointsPitch/sizeof(float), devCentroids, nClusters, devCentroidsPitch/sizeof(float), nDimension, devPointsDots, devCentroidsDots, devDiff, devDiffPitch/sizeof(float));
  int* devClusters;
  cudaMalloc((void**)&devClusters, nPoints * sizeof(int));
  cudaMemset(devClusters, 0, sizeof(int) * nPoints);
  int* devChanges;
  cudaMalloc((void**)&devChanges, sizeof(int));
  cudaMemset(devChanges, 0, sizeof(int));
  findDiffLabels<<<linearPointsGrid, linearPointsBlock>>>(devDiff, devDiffPitch/sizeof(float), nPoints, nClusters, devClusters, devChanges);
  int* hostClusters = (int*)malloc(sizeof(int) * nPoints);
  int* hostChanges = (int*)malloc(sizeof(int));
  cudaMemcpy(hostClusters, devClusters, sizeof(int) * nPoints, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostChanges, devChanges, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Printing new clusters: ");
  for(int i = 0; i < nPoints; i++) {
    printf("%d ", hostClusters[i]);
  }
  printf("\nChanges: %d\n", hostChanges[0]);
}
*/


int kmeans(int nPixels, int width, int height, int clusterCount, int filterCount, float* devResponses, int** p_devClusters, int maxIter, int convThresh) {
  printf("Beginning kmeans %d\n", maxIter);
  dim3 gridDim = dim3((width - 1)/XBLOCK + 1, (height - 1)/YBLOCK + 1);
  dim3 blockDim = dim3(XBLOCK, YBLOCK);

  dim3 linearGrid = dim3((width * height - 1)/512 + 1);
  dim3 linearBlock = dim3(512);

  dim3 clusterGrid = dim3((filterCount - 1)/XBLOCK + 1, (clusterCount - 1)/YBLOCK + 1);
  dim3 clusterBlock = dim3(XBLOCK, YBLOCK);

  cudaMalloc((void**)p_devClusters, sizeof(int)*nPixels);
  int* devClusters = *p_devClusters;

  int* devIntResponses;
  cudaMalloc((void**)&devIntResponses, sizeof(int) * nPixels * filterCount);
  assignInitialClusters<<<gridDim, blockDim>>>(width, height, nPixels, clusterCount, devClusters, filterCount, devResponses, devIntResponses);

  // yunsup fixed
  int afLocal = 4096;
  int afCopies = (width*height-1)/afLocal + 1;
  printf("afLocal=%d, afCopies=%d\n", afLocal, afCopies);
  int* devCentroidMass;
  cudaMalloc((void**)&devCentroidMass, sizeof(int) * filterCount * clusterCount * afCopies);
  unsigned int* devCentroidCount;
  cudaMalloc((void**)&devCentroidCount, sizeof(unsigned int) * filterCount * clusterCount * afCopies);

  float* devCentroids;
  cudaMalloc((void**)&devCentroids, sizeof(float) * filterCount * clusterCount);
  // yunsup fixed
  int* devChanges;
  cudaMalloc((void**)&devChanges, sizeof(int));

  float* devPointsDots;
  cudaMalloc((void**)&devPointsDots, sizeof(int) * nPixels);
  float* devCentroidsDots;
  cudaMalloc((void**)&devCentroidsDots, sizeof(int) * clusterCount);
  makeSelfDots<<<linearGrid, linearBlock>>>(devResponses, nPixels, devPointsDots, nPixels, filterCount);

  float* devDist;
  size_t devDistPitch;
  cudaMallocPitch((void**)&devDist, &devDistPitch, sizeof(float) * nPixels, clusterCount);
  int devDistPitchInFloats = devDistPitch/sizeof(float);

    int* hostClusters = new int[10];
    cudaMemcpy(hostClusters, devCentroidMass, 10*sizeof(int), cudaMemcpyDeviceToHost);
    for(int k=0;k<10;k++)
    {
        printf("%d: %d\n", k, hostClusters[k]);
    }
    delete [] hostClusters;

  int i;
  int hostChanges;
  for(i = 0; i < maxIter; i++) {
    // yunsup fixed
    cudaMemset(devCentroidMass, 0, sizeof(int) * filterCount * clusterCount * afCopies);
    cudaMemset(devCentroidCount, 0, sizeof(int) * filterCount * clusterCount * afCopies);
    cudaMemset(devChanges, 0, sizeof(int));
    {
        dim3 afGrid = dim3(afCopies, clusterCount);
        dim3 afBlock = dim3(filterCount);
        findCentroidsAtomicFreeLocal<<<afGrid, afBlock>>>(afLocal, devIntResponses, nPixels, devClusters, devCentroidMass, devCentroidCount);
        findCentroidsAtomicFreeReduce<<<afGrid, afBlock>>>(afLocal, devIntResponses, nPixels, devClusters, devCentroidMass, devCentroidCount);
    }
    finishCentroids<<<clusterGrid, clusterBlock>>>(devCentroidMass, devCentroidCount, devCentroids);
    findSgemmLabels(devResponses, nPixels, nPixels, devCentroids, clusterCount, clusterCount, filterCount, devPointsDots, devCentroidsDots, devDist, devDistPitchInFloats, devClusters, devChanges);
    // yunsup fixed
    cudaMemcpy(&hostChanges, devChanges, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\tChanges: %d\n", hostChanges);
//    if (hostChanges <= convThresh) {
//      break;
//    }

    hostClusters = new int[10];
    cudaMemcpy(hostClusters, devCentroidMass, 10*sizeof(int), cudaMemcpyDeviceToHost);
    int sum=0;
    for(int k=0;k<10;k++)
    {
        printf("%d: %d\n", k, hostClusters[k]);
        //sum += hostClusters[k];
    }
    printf("count = %u\n", sum);
    delete [] hostClusters;
  }
  printf("\t%i iterations until termination\n", i);
  cudaFree(devDist);
  cudaFree(devCentroidsDots);
  cudaFree(devPointsDots);
  cudaFree(devChanges);
  cudaFree(devCentroids);
  cudaFree(devCentroidCount);
  cudaFree(devCentroidMass);
  cudaFree(devIntResponses);
  printf("Kmeans completed\n");
  return hostChanges;
}
  
  
