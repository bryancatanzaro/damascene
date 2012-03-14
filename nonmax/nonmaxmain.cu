
// vim: ts=4 syntax=cpp comments=

#include <cutil.h>
#include <cuda.h>
#include "nonmax.h"
#include <stdio.h>


void chooseLargestGPU(bool verbose) 
{
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


void dummy()
{
	float* test;
	CUDA_SAFE_CALL(cudaMalloc((void**)&test, 100 * sizeof(float)));
	CUDA_SAFE_CALL(cudaFree(test));
}

void PrintMatrix(char* filename, int p_nWidth, int p_nHeight, float* p_aaMatrix)
{
	FILE* outfile = fopen(filename, "w");
	for (int i = 0; i < p_nHeight; i++)
	{
		for (int j = 0; j < p_nWidth; j++)
		{
			fprintf(outfile, "%f ", p_aaMatrix[i*p_nWidth+j]);
		}
		fprintf(outfile, "\n");
	}
	fclose(outfile);
}

void ReadPB(char* filename, int* p_nHeight, int* p_nWidth, int* p_nOrien, float** p_aafPB)
{
	FILE* infile = fopen(filename, "r");
	fscanf(infile, "%d %d %d", p_nHeight, p_nWidth, p_nOrien);
	(*p_aafPB) = (float*)malloc(sizeof(float)*(*p_nOrien)*(*p_nHeight)*(*p_nWidth));
	//int n = 0;
	for (int i = 0; i < (*p_nHeight); i++)
	{
		for (int j = 0; j < (*p_nWidth); j++)
		{
			for (int k = 0; k < (*p_nOrien); k++)
			{
				int offset = k * (*p_nHeight) * (*p_nWidth) + i * (*p_nWidth) + j;
				fscanf(infile, "%f", (*p_aafPB)+offset);
				//n++;
			}
		}
	}
	fclose(infile);
}

int main(int argc, char** argv)
{
	
	chooseLargestGPU(false);
	dummy();


	char * filename = "pb.txt";
	int width = 0; 
	int height = 0; 
	int orient = 0;
	float* pb = 0;
	ReadPB(filename, &height, &width, &orient, &pb);
	float* devpb = 0;
	size_t pitch = 0;
	int size = width * height;
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&devpb, &pitch, size *  sizeof(float), 8));
	pitch/=sizeof(float);
	for (int i = 0; i < 8; i++)
	{
		CUDA_SAFE_CALL(cudaMemcpy((devpb)+i*pitch, pb+i*size, size * sizeof(float), cudaMemcpyHostToDevice));
	}
	float* devNMax = 0;
	CUDA_SAFE_CALL(cudaMalloc((void**)&devNMax, size * sizeof(float)));

	nonMaxSuppression(width, height, devpb, pitch, devNMax);

	float* nmax = 0;
	nmax = (float*)malloc(sizeof(float)*size);
	CUDA_SAFE_CALL(cudaMemcpy(nmax, devNMax, size * sizeof(float), cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL(cudaFree(devNMax));
	CUDA_SAFE_CALL(cudaFree(devpb));

	PrintMatrix("nmax.txt", width, height, nmax);
	
	free(pb);
	free(nmax);
	return 0;
}
