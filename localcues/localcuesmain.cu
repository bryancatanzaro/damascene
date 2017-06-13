#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>


#include "spec.h"
#include "convert.h"
#include "localcues.h"
#include "stencilMVM.h"
#include "texton.h"
#include <helper_cuda.h>
#include <helper_timer.h>

void writeGra(char* file, int width, int height, int norients, int nscale, int cuePitchInFloats, float* hostGradient)
{
    int fd;

    fd = open(file, O_CREAT|O_WRONLY|O_TRUNC, 0666);
    write(fd, &width, sizeof(int));
    write(fd, &height, sizeof(int));
    write(fd, &norients, sizeof(int));
    write(fd, &nscale, sizeof(int));
    for(int scale = 0; scale < nscale; scale++) {
      for(int orient = 0; orient < norients; orient++) {
        write(fd, &hostGradient[(scale*norients + orient) * cuePitchInFloats], width*height*sizeof(float));
      }
    }
    close(fd);
}


int main(int argc, char** argv)
{
  cuInit(0);
  chooseLargestGPU(true);
  size_t total, free;
  cuMemGetInfo(&free, &total);
  printf("This GPU has %zu bytes of memory\n", total);
  printf("This GPU has %zu bytes of free memory\n", free);
  if (argc != 2)
  {
    printf("give me a file!\n");
    return 0;
  }
  char* filename = argv[1];
  uint width;
  uint height;
  uint* devRgbU;
  loadPPM_rgbU(filename, &width, &height, &devRgbU);
  
  cuMemGetInfo(&free, &total);
  printf("After loading the image, there are %zu bytes of free memory\n", free);
  float* devGreyscale;
  rgbUtoGreyF(width, height, devRgbU, &devGreyscale);
  cuMemGetInfo(&free, &total);
  printf("After converting to greyscale, there are %zu bytes of free memory\n", free);
  int* devTextons;
  int textonChoice = 1;
  findTextons(width, height, devGreyscale, &devTextons, textonChoice);
  cudaFree(devGreyscale);
  cuMemGetInfo(&free, &total);
  printf("After finding textons, there are %zu bytes of free memory\n", free);
 
  
  float* devL;
  float* devA;
  float* devB;
  rgbUtoLab3F(width, height, 2.5, devRgbU, &devL, &devA, &devB);
  cudaFree(devRgbU);
  normalizeLab(width, height, devL, devA, devB);
  cuMemGetInfo(&free, &total);
  printf("After converting to normalized LAB, there are %zu bytes of free memory\n", free);
 
  
  float* devBg;
  float* devCga;
  float* devCgb;
  float* devTg;
  int cuePitchInFloats;
  StopWatchInterface *timer=NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  localCues(width, height, devL, devA, devB, devTextons, &devBg, &devCga, &devCgb, &devTg, &cuePitchInFloats, textonChoice);
  sdkStopTimer(&timer);
  printf("Local cues time: %f ms\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);
  cuMemGetInfo(&free, &total);
  printf("After local cues, there are %zu bytes of free memory\n", free);
  int size = sizeof(float) * cuePitchInFloats * 8 * 3;
  float* hostBg = (float*)malloc(size);
  float* hostCga = (float*)malloc(size);
  float* hostCgb = (float*)malloc(size);
  float* hostTg = (float*)malloc(size);

  
  cudaMemcpy(hostBg, devBg, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCga, devCga, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostCgb, devCgb, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostTg, devTg, size, cudaMemcpyDeviceToHost);

  int norients = 8;
  int nscale = 3;
  writeGra("bg.gra", width, height, norients, nscale, cuePitchInFloats, hostBg);
  writeGra("cga.gra", width, height, norients, nscale, cuePitchInFloats, hostCga);
  writeGra("cgb.gra", width, height, norients, nscale, cuePitchInFloats, hostCgb);
  writeGra("tg.gra", width, height, norients, nscale, cuePitchInFloats, hostTg);
  
  
  
  //file = "data/L.dat";
  //read_dims();
  //read_parabola_filters();
  
  /* gpu_gradient_init(norients, width, height, border); */
/*   gpu_parabola_init(norients, width, height, border); */
  
/*   int ts1, ts2; */
  
/*   ts1 = timestamp(); */
/*   bg(); */
/*   cga(); */
/*   cgb(); */
/*   tg(); */
/*   ts2 = timestamp(); */
  
/*   printf("gpu_time = %fms\n", ((double)ts2-(double)ts1)/1000); */
  
/*   gpu_gradient_cleanup(); */
/*   gpu_parabola_cleanup(); */
}
