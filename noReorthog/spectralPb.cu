#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

static inline int timestamp()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec*1000000+tv.tv_usec;
}


////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#include "spectralPb_kernel.cu"

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

void spectralPb(float *eigenvalues, float *eigenvectors, int xdim, int ydim, int nvec)
{
	int imagesize_mirrored, xdim_mirrored, ydim_mirrored;
	float *eigenvectors_mirrored;
	float *eigenvector;
	float *h_Result;	
	float *h_Kernels;
	cudaArray *a_Data;
	float *d_Result;
	char fname[80];	
	int fd,val;
	int ts1, ts2;
	
	// read in convolution kernels
	
	fd = open(KERNEL_FILE,O_RDONLY);
	if (fd == -1) {
		perror("couldn't open kernel file");
		exit(-1);
	}
	
	h_Kernels = (float *) malloc(KERNEL_SIZE*NUM_ORIENT*sizeof(float));
	val = read(fd, h_Kernels, KERNEL_SIZE*NUM_ORIENT*sizeof(float));
	close(fd);	
	
	if (val != KERNEL_SIZE*NUM_ORIENT*sizeof(float)) {
		printf("Error reading kernel file\n");
		return;
	}	
	
	// add a border of width KERNEL_RADIUS around eigenvector matrices
	
	xdim_mirrored = xdim + 2*KERNEL_RADIUS;
	ydim_mirrored = ydim + 2*KERNEL_RADIUS;
	imagesize_mirrored = xdim_mirrored * ydim_mirrored;
	
	eigenvectors_mirrored = (float *) malloc(imagesize_mirrored*nvec*sizeof(float));
	
	for (int i=0;i<nvec;i++) 
		mirror_pixels(eigenvectors+(xdim*ydim*i), eigenvectors_mirrored+(xdim_mirrored*ydim_mirrored*i), xdim, ydim, KERNEL_RADIUS);
		
	imagesize_mirrored = imagesize_mirrored * sizeof(float); // value is in bytes
		
	// set up texture memory
	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();
	CUDA_SAFE_CALL( cudaMallocArray(&a_Data, &floatTex, xdim_mirrored, ydim_mirrored) );
	CUDA_SAFE_CALL( cudaBindTextureToArray(texData, a_Data) );		
	
	ts1 = timestamp();	
	
	// allocate space for result
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Result, imagesize_mirrored*NUM_ORIENT) );
	CUDA_SAFE_CALL( cudaMemset( d_Result, 0, imagesize_mirrored*NUM_ORIENT) );
	h_Result = (float *) malloc(imagesize_mirrored*NUM_ORIENT);	
	
	// copy kernels to constant memory space
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_Kernels, h_Kernels, KERNEL_SIZE*NUM_ORIENT*sizeof(float)) );	
	
	dim3 threadBlock(16, 12);
   dim3 blockGrid(iDivUp(xdim_mirrored, threadBlock.x), iDivUp(ydim_mirrored, threadBlock.y));
		
   for (int i=0;i<nvec;i++) {	
		eigenvector = eigenvectors_mirrored + i*(imagesize_mirrored/4);
		// copy image to texture memory
		CUDA_SAFE_CALL( cudaMemcpyToArray(a_Data, 0, 0, eigenvector, imagesize_mirrored, cudaMemcpyHostToDevice) );			
				
//  	printf("cspectralPb_kernel()\n");
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
//        CUT_SAFE_CALL( cutResetTimer(hTimer) );
//        CUT_SAFE_CALL( cutStartTimer(hTimer) );
		  spectralPb_kernel<<<blockGrid, threadBlock>>>(d_Result, xdim_mirrored, ydim_mirrored, 1/sqrt(eigenvalues[i]));
        CUT_CHECK_ERROR("spectralPb_kernel execution failed\n");
        CUDA_SAFE_CALL( cudaThreadSynchronize() );
//        CUT_SAFE_CALL( cutStopTimer(hTimer) );
//        gpuTime = cutGetTimerValue(hTimer);
//    printf("...convolutionTex2D() time: %f msecs; //%f Mpix/s\n", gpuTime, xdim * ydim * 1e-6 / (0.001 * gpuTime));
	}
	
//	printf("Reading back GPU results...\n");
   CUDA_SAFE_CALL( cudaMemcpy(h_Result, d_Result, imagesize_mirrored*NUM_ORIENT, cudaMemcpyDeviceToHost) );
   
	ts2 = timestamp();   
	
	printf("spectralPb time = %fms\n", ((double)ts2-(double)ts1)/1000);
   
	// write out sPb file, stripping away border   
   
	fd=open("spb_out.dat", O_CREAT|O_WRONLY,0666);
	if (fd == -1) {
		perror("Couldn't open spb_out.dat file for writing!");
		exit(-1);
	}   
   
   // write out dimensions and number of orientations at beginning of file
	write(fd, &xdim, sizeof(float));
	write(fd, &ydim, sizeof(float));
	val = NUM_ORIENT;	
	write(fd, &val, sizeof(float));   
      
	for(int i=0;i<NUM_ORIENT;i++) {
		int y;
		float *result;
		for (y=KERNEL_RADIUS;y<ydim_mirrored-KERNEL_RADIUS;y++) {
			result = h_Result + KERNEL_RADIUS + y*xdim_mirrored;
			result = result + i*ydim_mirrored*xdim_mirrored;
			val=write(fd, result, (xdim_mirrored-2*KERNEL_RADIUS)*sizeof(float));
			if (val != (xdim_mirrored-2*KERNEL_RADIUS)*sizeof(float)) {
				perror("Error writing to spb_out.dat!");
				close(fd);
				exit(-1);
			}
		}
	}      
	
   close(fd);
   
	// clean up   
   
   CUDA_SAFE_CALL( cudaUnbindTexture(texData) );
   CUDA_SAFE_CALL( cudaFree(d_Result)   );
   CUDA_SAFE_CALL( cudaFreeArray(a_Data)   );
   free(eigenvectors_mirrored);
}   
   
/*
int main(int argc, char **argv) {
	
	float *eigenvectors, *eigenvalues;
	
	struct stat status;
	int fd,val,filesize;	
	int 	xdim, ydim, nvec;

//   CUT_DEVICE_INIT(argc, argv);

	if (argc != 3) {
	   printf("Must specify filenames for eigenvector and eigenvalue files\n");
	   exit(-1);
	}
	
	fd = stat(argv[1], &status);
	if (fd == -1) {
		perror("couldn't stat file");
		exit(-1);
	}	

	filesize = status.st_size;
	
	fd = open(argv[1],O_RDONLY);
	if (fd == -1) {
		perror("couldn't open image file");
		exit(-1);
	}

	read(fd, &xdim, 4);
	read(fd, &ydim, 4);
	read(fd, &nvec, 4);

	printf("Image dimensions: %d x %d\n",xdim,ydim);	

	eigenvectors = (float *) malloc(filesize-12);
	val=read(fd, eigenvectors, filesize-12);
	close(fd);
	
	if (val != filesize-12) {
		printf("Error reading eigenvector file %d\n",val);
		exit(-1);
	}

	fd = open(argv[2],O_RDONLY);
	if (fd == -1) {
		perror("couldn't open eigenvalue file");
		exit(-1);
	}	
	
	eigenvalues = (float *) malloc(nvec*sizeof(float));
	read(fd,eigenvalues,nvec*sizeof(float));
	close(fd);
	
	spectralPb(eigenvalues, eigenvectors, xdim, ydim, nvec);

   printf("Shutting down...\n");
   free(eigenvectors);
   free(eigenvalues);

//   CUT_EXIT(argc, argv);
}

*/

				
	
