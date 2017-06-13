// vim: ts=4 syntax=cpp comments=

#include <helper_cuda.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "globalPb.h"

using namespace std;

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
	checkCudaErrors(cudaMalloc((void**)&test, 100 * sizeof(float)));
	checkCudaErrors(cudaFree(test));
}

size_t ReadOneMatrix(FILE* infile, float** devArray, int orient)
{
	float* tempArray = (float*)malloc(154401*orient*sizeof(float));
	for (int i = 0; i < orient; i++)
	{
		for (int j = 0; j < 154401; j++)
		{
			float temp = 1;
			fscanf(infile, "%f", &temp);
			tempArray[i*154401+j] = temp;
		}
	}

	size_t pitch = 0;
	if (orient > 1)
	{
		checkCudaErrors(cudaMallocPitch((void**)devArray, &pitch, 154401 *  sizeof(float), orient));
		pitch/=sizeof(float);
		for (int i = 0; i < orient; i++)
		{
			checkCudaErrors(cudaMemcpy((*devArray)+i*pitch, tempArray+i*154401, 154401 * sizeof(float), cudaMemcpyHostToDevice));
		}
	}
	else
	{
		checkCudaErrors(cudaMalloc((void**)devArray, 154401 * sizeof(float)));
		checkCudaErrors(cudaMemcpy((*devArray), tempArray, 154401 * sizeof(float), cudaMemcpyHostToDevice));
	}
	//printf("\n Pitch %d ", pitch);
	return pitch;
}

void ReadFromFile(int* p_nMatrixPitch,
		float** devbg1, float** devbg2, float** devbg3,
		float** devcga1, float** devcga2, float** devcga3,
		float** devcgb1, float** devcgb2, float** devcgb3,
		float** devtg1, float** devtg2, float** devtg3,
		float** devspb,
		float** devmpb)
{
	FILE* infile = fopen("gpb.txt", "r");
	int pitch = ReadOneMatrix(infile, devbg1, 8);
	int pitch1 = ReadOneMatrix(infile, devbg2, 8);
	assert(pitch==pitch1);
	pitch1 = ReadOneMatrix(infile, devbg3, 8);
	assert(pitch==pitch1);
	pitch1 = ReadOneMatrix(infile, devcga1, 8);
	assert(pitch==pitch1);
	pitch1 = ReadOneMatrix(infile, devcga2, 8);
	assert(pitch==pitch1);
	pitch1 = ReadOneMatrix(infile, devcga3, 8);
	assert(pitch==pitch1);
	pitch1 = ReadOneMatrix(infile, devcgb1, 8);
	assert(pitch==pitch1);
	pitch1 = ReadOneMatrix(infile, devcgb2, 8);
	assert(pitch==pitch1);
	pitch1 = ReadOneMatrix(infile, devcgb3, 8);
	assert(pitch==pitch1);
	pitch1 = ReadOneMatrix(infile, devtg1, 8);
	assert(pitch==pitch1);
	pitch1 = ReadOneMatrix(infile, devtg2, 8);
	assert(pitch==pitch1);
	pitch1 = ReadOneMatrix(infile, devtg3, 8);
	assert(pitch==pitch1);
	pitch1 = ReadOneMatrix(infile, devspb, 8);
	assert(pitch==pitch1);
	pitch1 = ReadOneMatrix(infile, devmpb, 1);
	*p_nMatrixPitch = pitch;
	fclose(infile);
}

void ClearAllMemory(
		float* devbg1, float* devbg2, float* devbg3,
		float* devcga1, float* devcga2, float* devcga3,
		float* devcgb1, float* devcgb2, float* devcgb3,
		float* devtg1, float* devtg2, float* devtg3,
		float* devspb,
		float* devmpb)
{
	checkCudaErrors(cudaFree(devbg1));
	checkCudaErrors(cudaFree(devbg2));
	checkCudaErrors(cudaFree(devbg3));
	checkCudaErrors(cudaFree(devcga1));
	checkCudaErrors(cudaFree(devcga2));
	checkCudaErrors(cudaFree(devcga3));
	checkCudaErrors(cudaFree(devcgb1));
	checkCudaErrors(cudaFree(devcgb2));
	checkCudaErrors(cudaFree(devcgb3));
	checkCudaErrors(cudaFree(devtg1));
	checkCudaErrors(cudaFree(devtg2));
	checkCudaErrors(cudaFree(devtg3));
	checkCudaErrors(cudaFree(devspb));
	checkCudaErrors(cudaFree(devmpb));
}

void StartGlobalPb()
{
	float* bg1 = 0;
	float* bg2 = 0;
	float* bg3 = 0;
	float* cga1 = 0;
	float* cga2 = 0;
	float* cga3 = 0;
	float* cgb1 = 0;
	float* cgb2 = 0;
	float* cgb3 = 0;
	float* tg1 = 0;
	float* tg2 = 0;
	float* tg3 = 0;
	float* spb = 0;
	float* mpb = 0;

	float* result = 0;

	float* hostResult = (float*)malloc(154401*sizeof(float));
	int pitch = 0;

	checkCudaErrors(cudaMalloc((void**)&result, 154401 * sizeof(float)));

	ReadFromFile(&pitch, &bg1, &bg2, &bg3, &cga1, &cga2, &cga3, &cgb1, &cgb2, &cgb3, &tg1, &tg2, &tg3, &spb, &mpb);
	//printf("\nPitch = %d\n", pitch);
	//StartCalcGPb(154401, pitch, 8, bg1, bg2, bg3, cga1, cga2, cga3, cgb1, cgb2, cgb3, tg1, tg2, tg3, spb, mpb, result);

	checkCudaErrors(cudaMemcpy(hostResult, result, 154401 * sizeof(float), cudaMemcpyDeviceToHost));

	ClearAllMemory(bg1, bg2, bg3, cga1, cga2, cga3, cgb1, cgb2, cgb3, tg1, tg2, tg3, spb, mpb);
	checkCudaErrors(cudaFree(result));

	for (int i = 0; i < 481; i++)
	{
		for (int j = 0; j < 321; j++)
		{
			printf("%f ", hostResult[i*321+j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

int main(int argc, char** argv)
{

	chooseLargestGPU(false);
	dummy();
	
	StartGlobalPb();

	return 0;
}


