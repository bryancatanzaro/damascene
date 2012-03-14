// vim: ts=4 syntax=cpp comments=

#include <cutil.h>
#include <cuda.h>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "nonmax.h"

#define M_PIl           3.1415926535897932384626433832795029L  /* pi */
#define M_PI_2l         1.5707963267948966192313216916397514L  /* pi/2 */
#define M_PI_4l         0.7853981633974483096156608458198757L  /* pi/4 */

#define IMUL(a, b) __mul24(a, b)
#define XSIZE 256

using namespace std;


__global__ void devFindMax(int p_nSize, size_t p_nPitch, float* p_aafPB, float* p_aafMaxPb, int* p_aafOri)
{
	int index = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	//float oris[] = {0,   0.39269908169872,   0.78539816339745,   1.17809724509617,   1.57079632679490,
	//	1.96349540849362,   2.35619449019234,   2.748894};

	if (index < p_nSize)
	{
		float max = -200;
		int maxori = 0;
		//for (int i = 0; i < 8; i++)
		//{
		int oriIndex = index;
		if (p_aafPB[oriIndex] > max)
		{
			max = p_aafPB[oriIndex];
			maxori = 0;
		}
		oriIndex = p_nPitch+index;
		if (p_aafPB[oriIndex] > max)
		{
			max = p_aafPB[oriIndex];
			maxori = 1;
		}
		oriIndex = p_nPitch*2+index;
		if (p_aafPB[oriIndex] > max)
		{
			max = p_aafPB[oriIndex];
			maxori = 2;
		}
		oriIndex = p_nPitch*3+index;
		if (p_aafPB[oriIndex] > max)
		{
			max = p_aafPB[oriIndex];
			maxori = 3;
		}
		oriIndex = p_nPitch*4+index;
		if (p_aafPB[oriIndex] > max)
		{
			max = p_aafPB[oriIndex];
			maxori = 4;
		}
		oriIndex = p_nPitch*5+index;
		if (p_aafPB[oriIndex] > max)
		{
			max = p_aafPB[oriIndex];
			maxori = 5;
		}
		oriIndex = p_nPitch*6+index;
		if (p_aafPB[oriIndex] > max)
		{
			max = p_aafPB[oriIndex];
			maxori = 6;
		}
		oriIndex = p_nPitch*7+index;
		if (p_aafPB[oriIndex] > max)
		{
			max = p_aafPB[oriIndex];
			maxori = 7;
		}
		//}
		if (max > 0)
		{
			p_aafMaxPb[index] = max;
		}
		else
		{
			p_aafMaxPb[index] = 0;
		}
		p_aafOri[index] = maxori;
	}
}

void initCudaArrays(int p_nSize, float** p_devMaxPB, int** p_devOri)
{
	CUDA_SAFE_CALL(cudaMalloc((void**)p_devMaxPB, p_nSize * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)p_devOri, p_nSize * sizeof(int)));
}

void freeCudaArrays(float* p_devMaxPB, int* p_devOri)
{
	CUDA_SAFE_CALL(cudaFree(p_devMaxPB));
	CUDA_SAFE_CALL(cudaFree(p_devOri));
}


void hostFindMax(int p_nHeight, int p_nWidth, int p_nOrien, float* p_aafPB, float* p_aafMaxPb, int* p_aafOri)
{
	//float oris[] = {0,   0.39269908169872,   0.78539816339745,   1.17809724509617,   1.57079632679490,
	//	1.96349540849362,   2.35619449019234,   2.748894};

	int n = 0;
	for (int i = 0; i < p_nHeight; i++)
	{
		for (int j = 0; j < p_nWidth; j++)
		{
			float max = -200;
			int maxori = 0;

			for (int k =0; k < p_nOrien; k++)
			{
				if (p_aafPB[n] > max)
				{
					max = p_aafPB[n];
					maxori = k;
				}
				n++;
			}
			if (max > 0)
			{
				p_aafMaxPb[i*p_nWidth+j]=max;
			}
			else
			{
				p_aafMaxPb[i*p_nWidth+j]=0;
			}
			p_aafOri[i*p_nWidth+j] = maxori;
		}
	}
}

texture<float, 1, cudaReadModeElementType> texMax;
texture<int, 1, cudaReadModeElementType> texOrient;

__global__ void devOriented_2D(int p_nHeight, int p_nWidth, float* nmax)
{
	int n = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	//Constants
	
	// get matrix size 
	int size_x = p_nHeight;
	int size_y = p_nWidth;
	//unsigned long size = size_x*size_y;
	int x = (int)(n / p_nWidth);
	int y = n - p_nWidth * x;
	// perform oriented non-max suppression at each element 
	if (n < size_x * size_y)
	{
		nmax[n] = 0;
		/* compute direction (in [0,pi)) along which to suppress */
		int ori = tex1Dfetch(texOrient, n);
		int theta = (ori + 4)%8;
		/* check nonnegativity */
		float v = tex1Dfetch(texMax, n);
		/* initialize indices of values in local neighborhood */
		int ind0a = 0, ind0b = 0, ind1a = 0, ind1b = 0;
		/* initialize distance weighting */
		/* initialize boundary flags */
		bool valid0 = false, valid1 = false;
		/* compute interpolation indicies */
		if (theta == 0) {
			valid0 = (x > 0); valid1 = (x < (size_x-1));
			if (valid0) { ind0a = n-size_y; ind0b = ind0a; }
			if (valid1) { ind1a = n+size_y; ind1b = ind1a; }
		} else if (theta == 1) {
			valid0 = ((x > 0) && (y > 0));
			valid1 = ((x < (size_x-1)) && (y < (size_y-1)));
			if (valid0) { ind0a = n-size_y; ind0b = ind0a-1; }
			if (valid1) { ind1a = n+size_y; ind1b = ind1a+1; }
		} else if (theta <=3) {
			valid0 = ((x > 0) && (y > 0));
			valid1 = ((x < (size_x-1)) && (y < (size_y-1)));
			if (valid0) { ind0a = n-1; ind0b = ind0a-size_y; }
			if (valid1) { ind1a = n+1; ind1b = ind1a+size_y; }
		} else if (theta <= 5) {
			valid0 = ((x < (size_x-1)) && (y > 0));
			valid1 = ((x > 0) && (y < (size_y-1)));
			if (valid0) { ind0a = n-1; ind0b = ind0a+size_y; }
			if (valid1) { ind1a = n+1; ind1b = ind1a-size_y; }
		} else /* (theta < M_PIl) */ {
			valid0 = ((x < (size_x-1)) && (y > 0));
			valid1 = ((x > 0) && (y < (size_y-1)));
			if (valid0) { ind0a = n+size_y; ind0b = ind0a-1; }
			if (valid1) { ind1a = n-size_y; ind1b = ind1a+1; }
		}
		/* check boundary conditions */
		if (valid0 && valid1) {
			float tand[] = {0, 0.414214, 1, 0.414214, 0, 0.414214, 1, 0.414214};
			float d = tand[theta];
			/* initialize values in local neighborhood */
			float v0a = 0,   v0b = 0,   v1a = 0,   v1b = 0;
			/* initialize orientations in local neighborhood */
			int ori0a = 0, ori0b = 0, ori1a = 0, ori1b = 0;
			/* grab values and orientations */
			v0a = tex1Dfetch(texMax, ind0a);
			v0b = tex1Dfetch(texMax, ind0b);

			ori0a = tex1Dfetch(texOrient, ind0a) - ori;
			if (ori0a < 0)
				ori0a*=(-1);
			ori0b = tex1Dfetch(texOrient, ind0b) - ori;
			if (ori0b < 0)
				ori0b*=(-1);

			v1a =  tex1Dfetch(texMax, ind1a);
			v1b =  tex1Dfetch(texMax, ind1b);
			ori1a = tex1Dfetch(texOrient, ind1a) - ori;
			if (ori1a < 0)
				ori1a*=(-1);
			ori1b = tex1Dfetch(texOrient, ind1b) - ori;
			if (ori1b < 0)
				ori1b*=(-1);
			/* place orientation difference in [0,pi/2) range */

			float cosori[]={1, 1, 0.923879, 0.707107, 0.382684, 0.707107, 0.923879, 1};

			/* interpolate */
			float v0 =
				(1.0-d)*v0a*cosori[ori0a] + d*v0b*cosori[ori0b];
			float v1 =
				(1.0-d)*v1a*cosori[ori1a] + d*v1b*cosori[ori1b];
			/* suppress non-max */
			if ((v > v0) && (v > v1))
			{
				//nmax[n] = v;
				float temp = 1.2*v;
				if (temp >= 1)
					nmax[n] = 1;
				else
					nmax[n] = temp;
			}
		}
	}
}



void Oriented_2D(int p_nHeight, int p_nWidth, float* m, int* m_ori, float* nmax)
{
	/*Constants*/
	float tand[] = {0, 0.414214, 1, 0.414214, 0, 0.414214, 1, 0.414214};
	//float o_tol = 0.39269908169872; 
	/* get matrix size */
	unsigned long size_x = p_nHeight;
	unsigned long size_y = p_nWidth;
	/* perform oriented non-max suppression at each element */
	unsigned long n = 0;
	for (unsigned long x = 0; x < size_x; x++) {
		for (unsigned long y = 0; y < size_y; y++) {
			/* compute direction (in [0,pi)) along which to suppress */
			int ori = m_ori[n];
			int theta = (ori + 4)%8;
			/* check nonnegativity */
			float v = m[n];
			if (v < 0)
				printf("matrix must be nonnegative");
			/* initialize indices of values in local neighborhood */
			unsigned long ind0a = 0, ind0b = 0, ind1a = 0, ind1b = 0;
			/* initialize distance weighting */
			/* initialize boundary flags */
			bool valid0 = false, valid1 = false;
			/* compute interpolation indicies */
			if (theta == 0) {
				valid0 = (x > 0); valid1 = (x < (size_x-1));
				if (valid0) { ind0a = n-size_y; ind0b = ind0a; }
				if (valid1) { ind1a = n+size_y; ind1b = ind1a; }
			} else if (theta == 1) {
				valid0 = ((x > 0) && (y > 0));
				valid1 = ((x < (size_x-1)) && (y < (size_y-1)));
				if (valid0) { ind0a = n-size_y; ind0b = ind0a-1; }
				if (valid1) { ind1a = n+size_y; ind1b = ind1a+1; }
			} else if (theta <=3) {
				valid0 = ((x > 0) && (y > 0));
				valid1 = ((x < (size_x-1)) && (y < (size_y-1)));
				if (valid0) { ind0a = n-1; ind0b = ind0a-size_y; }
				if (valid1) { ind1a = n+1; ind1b = ind1a+size_y; }
			//} else if (theta == 4) {
			//	valid0 = (y > 0); valid1 = (y < (size_y-1));
			//	if (valid0) { ind0a = n-1; ind0b = ind0a; }
			//	if (valid1) { ind1a = n+1; ind1b = ind1a; }
			} else if (theta <= 5) {
				valid0 = ((x < (size_x-1)) && (y > 0));
				valid1 = ((x > 0) && (y < (size_y-1)));
				if (valid0) { ind0a = n-1; ind0b = ind0a+size_y; }
				if (valid1) { ind1a = n+1; ind1b = ind1a-size_y; }
			} else /* (theta < M_PIl) */ {
				valid0 = ((x < (size_x-1)) && (y > 0));
				valid1 = ((x > 0) && (y < (size_y-1)));
				if (valid0) { ind0a = n+size_y; ind0b = ind0a-1; }
				if (valid1) { ind1a = n-size_y; ind1b = ind1a+1; }
			}
			/* check boundary conditions */
			if (valid0 && valid1) {
				float d = tand[theta];
				/* initialize values in local neighborhood */
				float v0a = 0,   v0b = 0,   v1a = 0,   v1b = 0;
				/* initialize orientations in local neighborhood */
				int ori0a = 0, ori0b = 0, ori1a = 0, ori1b = 0;
				/* grab values and orientations */
				v0a = m[ind0a];
				v0b = m[ind0b];

				ori0a = m_ori[ind0a] - ori;
				if (ori0a < 0)
					ori0a*=(-1);
				ori0b = m_ori[ind0b] - ori;
				if (ori0b < 0)
					ori0b*=(-1);

				v1a = m[ind1a];
				v1b = m[ind1b];
				ori1a = m_ori[ind1a] - ori;
				if (ori1a < 0)
					ori1a*=(-1);
				ori1b = m_ori[ind1b] - ori;
				if (ori1b < 0)
					ori1b*=(-1);
				/* place orientation difference in [0,pi/2) range */

				//if (ori0a >= 4) { ori0a = 7 - ori0a; }
				//if (ori0b >= 4) { ori0b = 7 - ori0b; }
				//if (ori1a >= 4) { ori1a = 7 - ori1a; }
				//if (ori1b >= 4) { ori1b = 7 - ori1b; }

				float cosori[]={1, 1, 0.923879, 0.707107, 0.382684, 0.707107, 0.923879, 1};

				//ori0a = (ori0a <= 1) ? 0 : (ori0a - 1);
				//ori0b = (ori0b <= 1) ? 0 : (ori0b - 1);
				//ori1a = (ori1a <= 1) ? 0 : (ori1a - 1);
				//ori1b = (ori1b <= 1) ? 0 : (ori1b - 1);

				//printf("\n 0a %f 0b %f 1a %f 1b %f x %d y %d theta %d", cosori[ori0a], cosori[ori0b], cosori[ori1a], cosori[ori1b], x, y, theta);

				/* interpolate */
				float v0 =
					(1.0-d)*v0a*cosori[ori0a] + d*v0b*cosori[ori0b];
				float v1 =
					(1.0-d)*v1a*cosori[ori1a] + d*v1b*cosori[ori1b];
				/* suppress non-max */
				if ((v > v0) && (v > v1))
					nmax[n] = v;
				else
					nmax[n] = 0;
			}
			/* increment linear coordinate */
			n++;
		}
	}
}


void nonMaxSuppression(int p_nWidth, int p_nHeight, float* p_devPB, int p_nDevPBPitch, float* devNMax)
{
	struct timeval start;
	gettimeofday(&start, 0);

	int size = p_nWidth*p_nHeight;

	struct timeval startMax;
	gettimeofday(&startMax, 0);

	float* devMaxPB = 0;
	int* devOri = 0;

	dim3 blockDim(XSIZE, 1);
	dim3 gridDim((p_nWidth * p_nHeight - 1)/XSIZE + 1, 1);


	//size_t pbPitch;
	initCudaArrays(size, &devMaxPB, &devOri);
	//gettimeofday(&start, 0);
        //printf("\nNow size %d Pitch %d", size, pbPitch);
	//printf("\nNow Pitch %d size =%d", p_nDevPBPitch, size);
	devFindMax<<<gridDim, blockDim>>>(size, p_nDevPBPitch, p_devPB, devMaxPB, devOri);

	//CUDA_SAFE_CALL(cudaMemcpy(maxpb, devMaxPB, size * sizeof(float), cudaMemcpyDeviceToHost));
	//CUDA_SAFE_CALL(cudaMemcpy(ori, devOri, size * sizeof(int), cudaMemcpyDeviceToHost));


	struct timeval stopMax;
	gettimeofday(&stopMax, 0);
	float solveTime = (float)(stopMax.tv_sec - startMax.tv_sec)  + ((float)(stopMax.tv_usec - startMax.tv_usec))*1e-6f;
	printf("\nMax time: %f seconds", solveTime);


	struct timeval startNoMax;
	gettimeofday(&startNoMax, 0);

	cudaChannelFormatDesc channelMax = cudaCreateChannelDesc<float>();
	size_t offset = 0;
	cudaBindTexture(&offset, &texMax, devMaxPB, &channelMax, size * sizeof(float));
	cudaChannelFormatDesc channelOri = cudaCreateChannelDesc<int>();
	size_t offset2 = 0;
	cudaBindTexture(&offset2, &texOrient, devOri, &channelOri, size * sizeof(int));

	devOriented_2D<<<gridDim, blockDim>>>(p_nHeight, p_nWidth, devNMax);

	CUDA_SAFE_CALL(cudaUnbindTexture(texMax));
	CUDA_SAFE_CALL(cudaUnbindTexture(texOrient));

	freeCudaArrays(devMaxPB, devOri);

	struct timeval stopNoMax;
	gettimeofday(&stopNoMax, 0);
	solveTime = (float)(stopNoMax.tv_sec - startNoMax.tv_sec)  + ((float)(stopNoMax.tv_usec - startNoMax.tv_usec))*1e-6f;
	printf("\nOriented Max time: %f seconds", solveTime);


	struct timeval stop;
	gettimeofday(&stop, 0);
	solveTime = (float)(stop.tv_sec - start.tv_sec)  + ((float)(stop.tv_usec - start.tv_usec))*1e-6f;

	printf("\nSolve time: %f seconds\n", solveTime);

}


