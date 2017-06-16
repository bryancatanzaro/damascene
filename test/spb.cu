// vim: ts=4 syntax=cpp comments=

#include "spectralPb.h"
#include <stdio.h>
#include <cutil.h>
#include <cuda.h>

void chooseLargestGPU(bool verbose) {
	int cudaDeviceCount;
	cudaGetDeviceCount(&cudaDeviceCount);
	int cudaDevice = 0;
	int maxSps = 0;
	struct cudaDeviceProp dp;
	for (int i = 0; i < cudaDeviceCount; i++) {
		cudaGetDeviceProperties(&dp, i);
		if (dp.multiProcessorCount > maxSps) {
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


int main(int argc, char** argv)
{
	chooseLargestGPU(false);
	int nvec = 9;
	int width = 321;
	int height = 481;
	float* eigvalue = (float*) malloc(sizeof(float)*nvec);
	float* eigvector = (float*) malloc(sizeof(float)*nvec*width*height);
	int size = width*height;
	FILE* fvalue = fopen("eigenValues.txt", "r");
	for (int i = 0; i < nvec; i++)
	{
		fscanf(fvalue, "%e", eigvalue+i);
	}
    fclose(fvalue);
	FILE* fvector = fopen("NormEigenVectors.txt", "r");
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < nvec; j++)
		{
			fscanf(fvector, "%f", eigvector + j*size + i);
		}
	}
	fclose(fvector);

	float* devEigVec = 0;
	CUDA_SAFE_CALL(cudaMalloc((void**)&devEigVec, width*height*nvec*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(devEigVec, eigvector, width*height*nvec*sizeof(float), cudaMemcpyHostToDevice));

	float* devResult = 0;
	size_t pitch = 0;
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&devResult, &pitch, size *  sizeof(float), 8));
	pitch/=sizeof(float);
	printf("pitch value %d", pitch);
	spectralPb(eigvalue, devEigVec, width, height, nvec, devResult, pitch);
	float* hostResult = (float*) malloc(sizeof(float)*pitch*8);
	CUDA_SAFE_CALL(cudaMemcpy(hostResult, devResult, sizeof(float)*pitch*8, cudaMemcpyDeviceToHost) );

	for (int i = 0; i < 8 ; i++)
	{
		for (int j = 0; j < size; j++)
		{
			//printf("%f ", hostResult[i*pitch+j]);
		}
		printf("\n");
	}

	CUDA_SAFE_CALL(cudaFree(devResult));
	free(eigvector);
	free(eigvalue);
}

