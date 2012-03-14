// vim: ts=4 syntax=cpp comments=

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include "convert.h"

#include "filters.h"

//Fast integer multiplication macro
#define IMUL(a, b) __mul24(a, b)

//Input data texture reference
texture<float, 2, cudaReadModeElementType> texData;

#define NUM_ORIENT 8
#define KERNEL_RADIUS 3
#define KERNEL_DIAMETER (KERNEL_RADIUS*2+1)
#define KERNEL_SIZE (KERNEL_DIAMETER*KERNEL_DIAMETER)

/*
static inline int timestamp()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec*1000000+tv.tv_usec;
}
*/


////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void mirror_pixels(float* original, float* mirrored, int width, int height, int border)
{
    int i, j;
    int border_width = width + 2*border;

    for (i=0; i<height; i++)
    {
        for (j=-border; j<width+border; j++)
        {
            if (0 <= j && j < width)
            {
                mirrored[(i+border)*border_width+(j+border)] = original[i*width+j];
            }
            else if (0 > j)
            {
                mirrored[(i+border)*border_width+(j+border)] = original[i*width+(-j-1)];
            }
            else
            {
                mirrored[(i+border)*border_width+(j+border)] = original[i*width+(2*width-j-1)];
            }
        }
    }

    for (i=0; i<border; i++)
    {
        for (j=0; j<width+border*2; j++)
        {
            mirrored[i*border_width+j] = mirrored[(2*border-i-1)*border_width+j];
            mirrored[(i+height+border)*border_width+j] = mirrored[(height+border-i-1)*border_width+j];
        }
    }
}

#define KERNEL_FILE "oe_filters.dat"


__device__ __constant__ float d_Kernels[KERNEL_SIZE*NUM_ORIENT];

__global__ void spectralPb_kernel(
	float *d_Result,
	int dataW,
	int dataH,
	float scale)
{
	 const   int ix = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const   int iy = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    int u,v,imagesize;
    imagesize = dataW*dataH;
    float val;
	 float sum[NUM_ORIENT];
	 sum[0] = 0; sum[1] = 0; sum[2] = 0; sum[3] = 0; sum[4] = 0; sum[5] = 0; sum[6] = 0; sum[7] = 0;

    if(ix < dataW && iy < dataH){
    			
    			u = -3; v = -3;
          val = tex2D(texData, ix+u, iy+v);
     		  sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = -3; v = -2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = -3; v = -1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = -3; v = 0;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = -3; v = 1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = -3; v = 2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = -3; v = 3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
						
//

    			u = -2; v = -3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = -2; v = -2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = -2; v = -1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = -2; v = 0;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = -2; v = 1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = -2; v = 2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = -2; v = 3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
						
//

    			u = -1; v = -3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = -1; v = -2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = -1; v = -1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = -1; v = 0;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = -1; v = 1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = -1; v = 2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = -1; v = 3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
						
//

    			u = 0; v = -3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = 0; v = -2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = 0; v = -1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = 0; v = 0;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = 0; v = 1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = 0; v = 2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = 0; v = 3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
						
//

    			u = 1; v = -3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = 1; v = -2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = 1; v = -1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = 1; v = 0;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = 1; v = 1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = 1; v = 2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = 1; v = 3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
						
//

    			u = 2; v = -3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = 2; v = -2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = 2; v = -1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = 2; v = 0;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = 2; v = 1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = 2; v = 2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = 2; v = 3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
						
//

    			u = 3; v = -3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = 3; v = -2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = 3; v = -1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = 3; v = 0;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = 3; v = 1;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];

    			u = 3; v = 2;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				
    			u = 3; v = 3;
     			val = tex2D(texData, ix+u, iy+v);
				sum[0] += val * d_Kernels[(0*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[1] += val * d_Kernels[(1*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[2] += val * d_Kernels[(2*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[3] += val * d_Kernels[(3*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[4] += val * d_Kernels[(4*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[5] += val * d_Kernels[(5*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[6] += val * d_Kernels[(6*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
				sum[7] += val * d_Kernels[(7*KERNEL_SIZE)+((KERNEL_RADIUS+u)+(KERNEL_RADIUS+v)*KERNEL_DIAMETER)];
						
//

        //d_Result[(IMUL(iy, dataW) + ix)] += 1.0f;
        d_Result[(IMUL(iy, dataW) + ix)] += abs(sum[0])*scale;
			d_Result[(1*imagesize)+(IMUL(iy, dataW) + ix)] += abs(sum[1])*scale;
			d_Result[(2*imagesize)+(IMUL(iy, dataW) + ix)] += abs(sum[2])*scale;
			d_Result[(3*imagesize)+(IMUL(iy, dataW) + ix)] += abs(sum[3])*scale;
			d_Result[(4*imagesize)+(IMUL(iy, dataW) + ix)] += abs(sum[4])*scale;
			d_Result[(5*imagesize)+(IMUL(iy, dataW) + ix)] += abs(sum[5])*scale;
			d_Result[(6*imagesize)+(IMUL(iy, dataW) + ix)] += abs(sum[6])*scale;
			d_Result[(7*imagesize)+(IMUL(iy, dataW) + ix)] += abs(sum[7])*scale;
	 }
}

/*
void PrintArray(int size, float* devArray)
{
	float* temp = (float*)malloc(sizeof(float)*size);

	CUDA_SAFE_CALL( cudaMemcpy(resu, devArray, xdim*ydim, cudaMemcpyDeviceToDevice) );

}
*/

void spectralPb(float *eigenvalues, float *devEigVec, int xdim, int ydim, int nvec, float* results, int res_pitch)
{
	int imagesize_mirrored, xdim_mirrored, ydim_mirrored;
	float *eigenvector;
	float *h_Kernels;
	cudaArray *a_Data;
	float *d_Result;
	int fd,val;
	
	// read in convolution kernels
	
	/* fd = open(KERNEL_FILE,O_RDONLY);
	if (fd == -1) {
		perror("couldn't open kernel file");
		exit(-1);
	} */
	
	h_Kernels = (float *) malloc(KERNEL_SIZE*NUM_ORIENT*sizeof(float));
	/* val = read(fd, h_Kernels, KERNEL_SIZE*NUM_ORIENT*sizeof(float));
	close(fd);	
	
	if (val != KERNEL_SIZE*NUM_ORIENT*sizeof(float)) {
		printf("Error reading kernel file\n");
		return;
	}

        for(int i=0;i<8*KERNEL_SIZE;i++)
        {
            printf("%f ",h_Kernels[i]);
            if((i+1)%KERNEL_DIAMETER == 0) printf("\n");
            if((i+1)%KERNEL_SIZE == 0) printf("\n");
        }
        printf("\n");*/ 
         //float* f = new float[KERNEL_SIZE];
         
        for(int ori=NUM_ORIENT-1;ori>=0;ori--)
        {
            gaussian_2D(h_Kernels+(NUM_ORIENT-1-ori)*KERNEL_SIZE, 1, 1, M_PI/2+ori*M_PI/NUM_ORIENT, 1, false, 3, 3 );
//            for(int i=0;i<KERNEL_SIZE;i++)
//            {
//                printf("%f ",h_Kernels[i+(NUM_ORIENT-1-ori)*KERNEL_SIZE]);
//                if((i+1)%KERNEL_DIAMETER == 0) printf("\n");
//                if((i+1)%KERNEL_SIZE == 0) printf("\n");
//            } 
        }

	// add a border of width KERNEL_RADIUS around eigenvector matrices
	
	xdim_mirrored = xdim + 2*KERNEL_RADIUS;
	ydim_mirrored = ydim + 2*KERNEL_RADIUS;
	imagesize_mirrored = xdim_mirrored * ydim_mirrored;
	
	float* devEigVecMirror = 0;
	cudaMalloc((void**)&devEigVecMirror, imagesize_mirrored*(nvec-1)*sizeof(float));
	
	for (int i=1;i<nvec;i++) 
	{
		mirrorImage(xdim, ydim, KERNEL_RADIUS, devEigVec+(xdim*ydim*i), devEigVecMirror+imagesize_mirrored*(i-1));
	}
		
	imagesize_mirrored = imagesize_mirrored * sizeof(float); // value is in bytes
		
	// set up texture memory
	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();
	CUDA_SAFE_CALL( cudaMallocArray(&a_Data, &floatTex, xdim_mirrored, ydim_mirrored) );
	CUDA_SAFE_CALL( cudaBindTextureToArray(texData, a_Data) );		
	
	//ts1 = timestamp();	
	
	// allocate space for result
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Result, imagesize_mirrored*NUM_ORIENT) );
	CUDA_SAFE_CALL( cudaMemset( d_Result, 0, imagesize_mirrored*NUM_ORIENT) );
	//h_Result = (float *) malloc(imagesize_mirrored*NUM_ORIENT);	
	
	// copy kernels to constant memory space
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_Kernels, h_Kernels, KERNEL_SIZE*NUM_ORIENT*sizeof(float)) );	

	dim3 threadBlock(16, 12);
  dim3 blockGrid(iDivUp(xdim_mirrored, threadBlock.x), iDivUp(ydim_mirrored, threadBlock.y));
		
   for (int i=1;i<nvec;i++) {
		eigenvector = devEigVecMirror + (i-1)*(imagesize_mirrored/sizeof(float));
		// copy image to texture memory
		CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, eigenvector, imagesize_mirrored, cudaMemcpyDeviceToDevice) );
				
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
		  spectralPb_kernel<<<blockGrid, threadBlock>>>(d_Result, xdim_mirrored, ydim_mirrored, 1/sqrt(eigenvalues[i]));
        CUT_CHECK_ERROR("spectralPb_kernel execution failed\n");
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
	}
	
	
	for (int i = 0; i < NUM_ORIENT; i++)
	{
		unMirrorImage(xdim_mirrored, ydim_mirrored, KERNEL_RADIUS, d_Result+i*ydim_mirrored*xdim_mirrored, results+i*res_pitch);
	}
	
/*
	CUDA_SAFE_CALL( cudaMemcpy(h_Result, d_Result, imagesize_mirrored*NUM_ORIENT, cudaMemcpyDeviceToHost) );
   for(int i=0;i<NUM_ORIENT;i++) {
	   int y;
	   float *result;
	   for (y=KERNEL_RADIUS;y<ydim_mirrored-KERNEL_RADIUS;y++) {
		   result = h_Result + KERNEL_RADIUS + y*xdim_mirrored;
		   result = result + i*ydim_mirrored*xdim_mirrored;
		   for (int j = 0; j < xdim_mirrored-2*KERNEL_RADIUS; j++)
		   {
				printf("%f ", *(result+j));
			}
	   }
	   printf("\n");
   }
*/
   
	//ts2 = timestamp();   
	
	//printf("spectralPb time = %fms\n", ((double)ts2-(double)ts1)/1000);
   
   CUDA_SAFE_CALL( cudaUnbindTexture(texData) );
   CUDA_SAFE_CALL( cudaFree(d_Result)   );
   CUDA_SAFE_CALL( cudaFreeArray(a_Data)   );
   free(h_Kernels);
}  


