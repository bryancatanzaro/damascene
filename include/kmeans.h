#ifndef KMEANS
#define KMEANS

#define XBLOCK 16
#define YBLOCK 16
#define LINBLOCK 512
#define IMUL(a, b) __mul24(a, b)

__global__ void findCentroids(int* responses, int nPixels, int* cluster, int* centroidMass, unsigned int* centroidCount);
__global__ void assignInitialClusters(int width, int height, int nPixels, int clusterCount, int* cluster, int filterCount, float* responses, int* intResponses);
__global__ void finishCentroids(int* centroidMass, unsigned int* centroidCount, float* centroids);
__global__ void findLabels(int nPixels, int filterCount, int clusterCount, float* responses, float* centroids, int* clusters, int* changes);
void findSgemmLabels(float* devA, int nA, int aPitchInFloats, float* devB, int nB, int bPitchInFloats, int nDimension, float* devADots, float* devBDots, float* devDiff, int diffPitchInFloats, int* devClusters, int* devChanges);
__global__ void makeSelfDots(float* devSource, int devSourcePitchInFloats, float* devDest, int sourceCount, int sourceLength);

#define INTCONFACTOR 100000

int kmeans(int textonChoice, int nPixels, int width, int height, int clusterCount, int filterCount, float* devResponses, int** p_devClusters, int maxIter = 15, int convThresh = 0);


#endif
