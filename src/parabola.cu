#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <damascene/util.h>

#include <damascene/spec.h>

cudaArray* cuda_parabola_pixels;
texture<float, 2, cudaReadModeElementType> tex_parabola_pixels;

__constant__ float const_parabola_filters[MAX_FILTER_LENGTH*MAX_FILTER_LENGTH*MAX_FILTER_ORIENTATION];

//float* cuda_parabola_filters;
//texture<float, 1, cudaReadModeElementType> tex_parabola_filters;

float* cuda_parabola_trace;

__global__ void
parabolaKernel(float* trace, int width, int height, int limit, int border, int border_height, int filter_radius, int filter_length, int filter_width)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int ay = blockIdx.y*blockDim.y + threadIdx.y;

    if (x < width && ay < limit)
    {
        int ori = ay / height;
        int y = ay % height;

        float val = 0;

        for (int v=-filter_radius; v<=filter_radius; v++)
        {
            int cy = y + border + v + ori*border_height;
            int fidx = (v+filter_radius)*filter_length+ori*filter_width;

            for (int u=-filter_radius; u<=filter_radius; u++)
            {
                int cx = x + border + u;
                val += (tex2D(tex_parabola_pixels, cx, cy) * const_parabola_filters[fidx+u+filter_radius]);
                //val += (tex2D(tex_parabola_pixels, cx, cy) * tex1Dfetch(tex_parabola_filters, fidx+u+filter_radius));//[fidx+u+filter_radius]);
            }
        }

        trace[x+y*width+ori*width*height] = val;
        //trace[x+y*width+ori*width*height] = tex2D(tex_parabola_pixels, x+border, y+border);
    }
}


static inline void cuda_parabola_allocate(int norients, int width, int height, int border)
{
    int border_width = width+2*border;
    int border_height = height+2*border;

    cudaChannelFormatDesc ch;
    ch = cudaCreateChannelDesc<float>();

    CUDA_SAFE_CALL(
        cudaMallocArray(&cuda_parabola_pixels, &ch, border_width, border_height*norients) );

    tex_parabola_pixels.addressMode[0] = cudaAddressModeClamp;
    tex_parabola_pixels.addressMode[1] = cudaAddressModeClamp;
    tex_parabola_pixels.filterMode = cudaFilterModePoint;
    tex_parabola_pixels.normalized = 0;

   /*  CUDA_SAFE_CALL( */
/*         cudaBindTextureToArray(tex_parabola_pixels, cuda_parabola_pixels) ); */

    CUDA_SAFE_CALL(
      cudaMalloc((void**)&cuda_parabola_trace, width*height*norients*sizeof(float)) );
}

static inline void cuda_parabola_free()
{
  //    CUDA_SAFE_CALL(cudaUnbindTexture(tex_parabola_pixels));
    CUDA_SAFE_CALL(cudaFreeArray(cuda_parabola_pixels));
    CUDA_SAFE_CALL(cudaFree(cuda_parabola_trace));
    //CUDA_SAFE_CALL(cudaFree(cuda_parabola_filters));
}

static inline void copy_cuda_parabola_buffers(int norients, int width, int height, int border, float *devPixels, int filter_radius, int filter_length, float* host_filters)
{
    int border_width = width+2*border;
    int border_height = height+2*border;

    // copy pixels
    CUDA_SAFE_CALL(
        cudaMemcpy2DToArray(cuda_parabola_pixels, 0, 0, devPixels, border_width*sizeof(int), border_width*sizeof(int), border_height*norients, cudaMemcpyDeviceToDevice) );

    // copy const buffers (filters)
    CUDA_SAFE_CALL(
       cudaMemcpyToSymbol(const_parabola_filters, host_filters, norients*filter_length*filter_length*sizeof(float)) );


    //CUDA_SAFE_CALL(cudaMalloc((void**)&cuda_parabola_filters, sizeof(float)*filter_length*filter_length*norients));
    //CUDA_SAFE_CALL(cudaMemcpy(cuda_parabola_filters, host_filters, filter_length*filter_length*norients*sizeof(float), cudaMemcpyHostToDevice));

	//cudaChannelFormatDesc channelMax = cudaCreateChannelDesc<float>();
	//size_t offset = 0;
	//cudaBindTexture(&offset, &tex_parabola_filters, cuda_parabola_filters, &channelMax, filter_length*filter_length*norients* sizeof(float));
    
         
}

static inline void cuda_parabola_kernel(int norients, int width, int height, int border, int filter_radius, int filter_length, float* devResult, int cuePitchInFloats)
{
    cudaError_t err;

    CUDA_SAFE_CALL(cudaBindTextureToArray(tex_parabola_pixels, cuda_parabola_pixels) );
    dim3 grid(width/16+1, height*norients/16+1, 1);
    dim3 threads(16, 16, 1);

    parabolaKernel<<<grid, threads>>>(cuda_parabola_trace, width, height, height*norients, border, height+2*border, filter_radius, filter_length, filter_length*filter_length);

    if (cudaSuccess != (err = cudaThreadSynchronize()))
    {
        fprintf(stderr, "TB ERROR at %s:%d \"%s\"\n",
            __FILE__, __LINE__,  cudaGetErrorString(err));
    }

    if (cudaSuccess != (err = cudaGetLastError()))
    {
        fprintf(stderr, "TB ERROR at %s:%d \"%s\"\n",
            __FILE__, __LINE__, cudaGetErrorString(err));
    }

    int nPixels = width * height;
    for(int i = 0; i < norients; i++) {
      cudaMemcpy(devResult + cuePitchInFloats * i, cuda_parabola_trace + nPixels * i, nPixels*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_parabola_pixels));
    //CUDA_SAFE_CALL(cudaUnbindTexture(tex_parabola_filters));
 
/*     CUDA_SAFE_CALL( */
/*         cudaMemcpy(host_gradient, cuda_parabola_trace, width*height*norients*sizeof(float), cudaMemcpyDeviceToHost) ); */

#if 0
    int i, j, k;

    printf("gpu\n");
    for (i=0; i<norients; i++)
    {
        printf("%d orientation\n", i+1);
        for (j=0; j<height; j++)
        {
            for (k=0; k<width; k++)
            {
                printf("%9.6f ", host_gradient[k+j*width+i*width*height]);
            }
            printf("\n");
        }
    }
#endif
}

void gpu_parabola_init(int norients, int width, int height, int border)
{
    cuda_parabola_allocate(norients, width, height, border);
}

void gpu_parabola_cleanup()
{
    cuda_parabola_free();
}

void gpu_parabola(int norients, int width, int height, int border, float* devPixels, int filter_radius, int filter_length, float* filters, float* devResult, int cuePitchInFloats)
{
    copy_cuda_parabola_buffers(norients, width, height, border, devPixels, filter_radius, filter_length, filters);
    cuda_parabola_kernel(norients, width, height, border, filter_radius, filter_length, devResult, cuePitchInFloats);

#if 0
    for (int o=0; o<norients; o++)
    {
        int fd;
        char file[1024];
        sprintf(file, "bcg_%d_%d.dat", radius, o+1);
        fd = open(file, O_CREAT|O_WRONLY, 0666);
        write(fd, &host_gradient[o*width*height], width*height*sizeof(float));
        close(fd);
    }
#endif
}
