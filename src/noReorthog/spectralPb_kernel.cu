//Fast integer multiplication macro
#define IMUL(a, b) __mul24(a, b)

//Input data texture reference
texture<float, 2, cudaReadModeElementType> texData;

#define NUM_ORIENT 8
#define KERNEL_RADIUS 3
#define KERNEL_DIAMETER (KERNEL_RADIUS*2+1)
#define KERNEL_SIZE (KERNEL_DIAMETER*KERNEL_DIAMETER)

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
