#include "skeleton.h"
#define IMUL(a, b) __mul24(a, b)

texture<float, 2, cudaReadModeElementType> image;
__device__ __constant__ int devLut[8*512];


void __global__ applyLut(int width, int height, unsigned int matrixPitchInFloats, float* devPb_thin, int lutid, int* changed)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x; 
    int row = blockIdx.y*blockDim.y + threadIdx.y; 

    __shared__ int weights[3][3];
    if(threadIdx.x+threadIdx.y == 0)
    {
        weights[0][0] = 1;weights[0][1] =  8;weights[0][2] =  64;
        weights[1][0] = 2;weights[1][1] = 16;weights[1][2] = 128;
        weights[2][0] = 4;weights[2][1] = 32;weights[2][2] = 256;
    }
    __syncthreads();
    
    int minRow, minCol;
    int maxRow, maxCol;
   
    if(row<height && col<width)
    {
        minRow = (row==0)?1:0;
        maxRow = (row==height-1)?1:2;
        minCol = (col==0)?1:0;
        maxCol = (col==width-1)?1:2;
   

        int result=0;
        for(int r=minRow;r<=maxRow;r++)
        {
            for(int c=minCol;c<=maxCol;c++)
            {
                if(tex2D(image, (col+c-1), (row+r-1)) != 0)
                {
                    //printf("add : %d \n",weights[r][c]);
                    result += weights[r][c];
                }
            }
        }
        if(devLut[lutid*512 + result]==1)
        {
            //printf("%d %d\n", row, col);
            devPb_thin[row*matrixPitchInFloats + col] =0;
            *changed = 1;
        }
    }
}

void __global__ normalizewrtmpb(int width, int height, unsigned int matrixPitchInFloats, float* devGPb, float *devMPb, float* devGPb_thin)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;

    if(idx < width && idy < height)
    {
        devGPb_thin[idy*matrixPitchInFloats + idx] = (devMPb[idy*matrixPitchInFloats + idx] > 0.05)?1:0;
    }
}

void __global__ mask(int width, int height, unsigned int matrixPitchInFloats, float* devGPb, float* devGPb_thin)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;

    if(idx < width && idy < height)
    {
        if( devGPb_thin[idy*matrixPitchInFloats + idx] != 0)
        {
            devGPb_thin[idy*matrixPitchInFloats + idx] = (devGPb[idy*matrixPitchInFloats + idx]) ;
        }
    }
}

void __global__ normalizeSigmoid(int width, int height, unsigned int matrixPitchInFloats, float* devGPb)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int idy = blockDim.y*blockIdx.y + threadIdx.y;

    if(idx < width && idy < height)
    {
        float val = devGPb[idy*matrixPitchInFloats + idx] ;
        
        //val = max(0, min(1, 1.2*val));
        val = (val<=0) ? 0 : ( (val*1.2>1) ? 1 : (1.2*val) );

        val = 1.0f/(1.0 + expf(2.6433 -10.7998*val));
        val = (val - 0.0667)/0.9333;
        devGPb[idy*matrixPitchInFloats + idx] = val;
    }
}

void skeletonize(int width, int height, int matrixPitchInFloats, float* devGPb_thin)
{
    //initialize luts
    int* lutc= new int[NUM_LUTS*512];
    memset(lutc, 0, NUM_LUTS*512*sizeof(int));

    int* plut = lutc;
    //lut 1
    plut[89] = plut[91] = plut[217] = plut[219] = 1;
    //lut 2
    plut = plut + 512;
    plut[152] = plut[153] = plut[216] = plut[217] = plut[408] = plut[409] = plut[472] = plut[473] = 1;
    //lut 3
    plut = plut + 512;
    plut[23] = plut[31] = plut[55]= plut[63] = 1;
    //lut 4
    plut = plut + 512;
    plut[26] = plut[27] = plut[30] = plut[31] = plut[90] = plut[91] = plut[94] = plut[95] = 1;
    //lut 5
    plut = plut + 512;
    plut[464] = plut[472] = plut[496]= plut[504] = 1;
    //lut 6
    plut = plut + 512;
    plut[50] = plut[51] = plut[54] = plut[55] = plut[306] = plut[307] = plut[310] = plut[311] = 1;
    //lut 7
    plut = plut + 512;
    plut[308] = plut[310] = plut[436]= plut[438] = 1;
    //lut 8
    plut = plut + 512;
    plut[176] = plut[180] = plut[240] = plut[244] = plut[432] = plut[436] = plut[496] = plut[500] = 1;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(devLut, lutc, NUM_LUTS*512*sizeof(int)));
    
    //bind image to texture
      
    cudaArray* imageArray;
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();
    CUDA_SAFE_CALL(cudaMallocArray(&imageArray, &floatTex, matrixPitchInFloats , height));
    CUDA_SAFE_CALL(cudaMemcpyToArray(imageArray, 0, 0, devGPb_thin, matrixPitchInFloats * height * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaBindTextureToArray(image, imageArray));

    // skeletonize ; copy input image -> output image - till convergence

    int change;

    int* devchange;
    CUDA_SAFE_CALL(cudaMalloc((void**)&devchange, sizeof(int)));

    dim3 block(16,16);
    dim3 grid( (width+block.x-1)/block.x, (height+block.y-1)/block.y );
    
    int iter = 0;
    printf("Skeletonizing ... \n");
    do 
    {
        CUDA_SAFE_CALL(cudaMemset(devchange, 0, sizeof(int)));
        for(int lutid = 0; lutid < NUM_LUTS; lutid++)
        {
            applyLut<<<grid, block>>> (width, height, matrixPitchInFloats, devGPb_thin, lutid, devchange);
            CUDA_SAFE_CALL(cudaMemcpyToArray(imageArray, 0, 0, devGPb_thin, matrixPitchInFloats * height * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        CUDA_SAFE_CALL(cudaMemcpy(&change, devchange, sizeof(int), cudaMemcpyDeviceToHost));
        iter++;

        printf("\tIteration = %d, %s\n", iter, (change > 0) ? "Image changed":"Image unchanged");
    }while(change != 0);

    CUDA_SAFE_CALL(cudaFree(devchange));
    CUDA_SAFE_CALL(cudaFreeArray(imageArray));

}

void PostProcess(int width, int height, unsigned int matrixPitchInFloats,float* devGPb, float* devMPb, float* devGPb_thin)
{
    dim3 block(16,16);
    dim3 grid( (width+block.x-1)/block.x, (height+block.y-1)/block.y );

    normalizewrtmpb<<<grid,block>>>(width, height, matrixPitchInFloats, devGPb, devMPb, devGPb_thin);
    
    skeletonize(width, height, matrixPitchInFloats, devGPb_thin);

    mask<<<grid,block>>>(width, height, matrixPitchInFloats, devGPb, devGPb_thin);

    normalizeSigmoid<<<grid,block>>>(width, height, matrixPitchInFloats, devGPb);
    normalizeSigmoid<<<grid,block>>>(width, height, matrixPitchInFloats, devGPb_thin);
}

int Nhood3(float* image, int width, int height, int col, int row)
{
    int weights[3][3] = {{1,8,64}, {2,16,128}, {4,32,256}};
    int minRow, minCol;
    int maxRow, maxCol;
    
    minRow = (row==0)?1:0;
    maxRow = (row==height-1)?1:2;
    minCol = (col==0)?1:0;
    maxCol = (col==width-1)?1:2;

    int result=0;
    for(int r=minRow;r<=maxRow;r++)
    {
        for(int c=minCol;c<=maxCol;c++)
        {
            if(image[(row+r-1)*width+(col+c-1)] != 0)
            {
                //printf("add : %d \n",weights[r][c]);
                result += weights[r][c];
            }
        }
    }
    
    //printf("%d %d result : %d \n",row,col,result);

    return result;
}

void CPU_skeletonize(int width, int height, float* in_image, float* out_image, int* plut)
{
    for(int y=0;y<height;y++)
    {
        for(int x=0;x<width;x++)
        {
            out_image[y*width+x] = plut[Nhood3(in_image, width, height, x, y)];
        }
    }
}

__global__ void SigmoidGpbAll(int p_nPixels, int p_nOrient, int matrixPitchInFloats, float* devGpball)
{
    int index = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index < p_nPixels)
    {
	for (int i = 0; i < p_nOrient; i++)
	{
	    int oindex = index + i*matrixPitchInFloats;
	    float value = devGpball[oindex];
	    if (value*1.2 > 1)
		value = 1;
	    else
		value *= 1.2;
	    if (value < 0)
		value = 0;
	    value = 1.0f / (1.0f + expf(2.6433-10.7998*value));
	    value = (value - 0.0667)/0.9333;
	    devGpball[oindex] = value;
	}
    }
}

void NormalizeGpbAll(int p_nPixels, int p_nOrient, int matrixPitchInFloats, float* devGpball)
{
    dim3 blockDim(256, 1);
    dim3 gridDim((p_nPixels - 1)/256 + 1, 1);
    SigmoidGpbAll<<<gridDim, blockDim>>>(p_nPixels, p_nOrient, matrixPitchInFloats, devGpball);
}

