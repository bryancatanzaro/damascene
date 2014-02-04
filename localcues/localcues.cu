#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <acml.h>

#include "spec.h"
#include "gradient.h"
#include "convert.h"
#include "stencilMVM.h"

#define TEXTON32 1
#define TEXTON64 2

size_t radii[4]={3,5,10,20};

float** filters;

void savgol_filter(float* filt, int d, float inra, float inrb, float theta)
{
    int k=1;

    assert(d>0);
    assert(k>=1 && k<=d+1);

    float ra=max(inra,1.5);
    float rb=max(inrb,1.5);
    float ira2=1.0/(ra*ra);
    float irb2 = 1.0/(rb*rb);
    int wr = int(floor(max(ra,rb))+0.5);
    int wd = 2*wr+1;
    float sint = sin(theta);
    float cost = cos(theta);

    //float* filt = new float[wd*wd];
    memset(filt,0,sizeof(float)*wd*wd);

    float* xx= new float[2*d+1];
    memset(xx,0,sizeof(float)*(2*d+1));

    for(int u=-wr;u<=wr;u++)
    {
        for(int v=-wr;v<=wr;v++)
        {
            float ai=-u*sint+v*cost;
            float bi=u*cost+v*sint;

            if(ai*ai*ira2 + bi*bi*irb2 > 1)
            {
                continue;
            }
            for(int i=0;i<2*d+1;i++)
            {
                xx[i]=xx[i]+pow(ai,i);
            }
        }
    }
    float *A=new float[(d+1)*(d+1)];
    for(int i=0;i<d+1;i++)
    {
        for(int j=0;j<d+1;j++)
        {
            A[i*(d+1)+j] = xx[i+j];
        }
    }
    //float *invA=new float[(d+1)*(d+1)];
    //A = inv(A)
    int info;
    int* ipiv = new int[d+1];
    float* work=new float[d+1];
    sgetrf(d+1, d+1, A, d+1, ipiv, &info);
    sgetri(d+1, A, d+1, ipiv, &info);

    for(int u=-wr;u<=wr;u++)
    {
        for(int v=-wr;v<=wr;v++)
        {
            
            float ai=-u*sint+v*cost;
            float bi=u*cost+v*sint;

            if(ai*ai*ira2 + bi*bi*irb2 > 1)
            {
                continue;
            }
            for(int i=0;i<d+1;i++)
            {
                filt[(v+wr)*wd+u+wr] += A[i]*pow(ai,i); //doing only k=1
            }
        }
    }

    delete [] A;
    delete [] ipiv;
    delete [] work;
    delete [] xx;

}

void construct_parabola_filters(size_t number, size_t* radii, size_t norients)
{
  filters = (float**)malloc(sizeof(float*) * number);


  for(int filter = 0; filter < number; filter++) {
    int filter_radius;
    int filter_length;

    //filter_radius = 3;
    filter_radius = radii[filter];
    filter_length = 2*filter_radius+1;
  
    filters[filter] = (float*)malloc(filter_length*filter_length*norients*sizeof(float));
    float* currentFilter = filters[filter];
    
    for (int o=0; o<norients; o++)
      {
      
      savgol_filter(currentFilter+filter_length*filter_length*o, 2, filter_radius, float(filter_radius)/4.0f, M_PI/2-o*M_PI/8);

      /*if(o==1) 
      printf("Filter # %d\n", o);
      for(int x=0;x<filter_length*filter_length;x++) {
            printf("%1.4f ", filters3[filter_length*filter_length*o+x]);
            if((x+1)%filter_length==0) printf("\n");
      }
      printf("\n");
      float *f=savgol_filter(f, 2, 3, 0.75, M_PI/2-o*M_PI/8);
      for(int x=0;x<filter_length*filter_length;x++) {
            printf("%1.4f ", f[x]);
            if((x+1)%filter_length==0) printf("\n");
      }*/

    }

  }

}

void writeGrad(char* file, int width, int height, int norients, int nscale, int cuePitchInFloats, float* hostGradient)
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

void writeHist(char* file, int width, int height, int norients, int nscale, int x, int y, int orient, int scale, int nbins, float* hostDebug)
{
    int fd;
    int outWidth = nbins;
    int outHeight = 2;
    fd = open(file, O_CREAT|O_WRONLY|O_TRUNC, 0666);
    write(fd, &outWidth, sizeof(int));
    write(fd, &outHeight, sizeof(int));

    write(fd, &hostDebug[(y*width + x)*nbins*2], nbins*2*sizeof(float));

    close(fd);
}

void bg(size_t width, size_t height, size_t norients, size_t nscale, size_t* bgRadii, float** filters, float* devL, float** p_devBg, int cuePitchInFloats, int textonChoice)
{
  cudaMalloc((void**)p_devBg, sizeof(float) * cuePitchInFloats * norients * nscale);
  float* devBg = *p_devBg;


  size_t nbins = 25;
  float sigma = 0.1;
  bool blur = true;
  size_t border = 30;

  size_t borderWidth = width + 2 * border;
  size_t borderHeight = height + 2 * border;
  float* devGradients = gradients(devL, nbins, blur, sigma, bgRadii, textonChoice);
  uint cueTimer;
  cutCreateTimer(&cueTimer);
  cutStartTimer(cueTimer);
  for(int scale = 0; scale < nscale; scale++) {
    int radius = bgRadii[scale];
    int length = 2*radius + 1;
    gpu_parabola(norients, width, height, border, &devGradients[borderWidth * borderHeight * norients * scale], radius, length, filters[scale], devBg + cuePitchInFloats * norients * scale, cuePitchInFloats);
  }
  cutStopTimer(cueTimer);
  printf(">+< \tBgsmooth: | %f | ms\n", cutGetTimerValue(cueTimer));
}

void cg(size_t width, size_t height, size_t norients, size_t nscale, size_t* cgRadii, float** filters, float* devInput, float** p_devCg, int cuePitchInFloats, int textonChoice)
{
  cudaMalloc((void**)p_devCg, sizeof(float) * cuePitchInFloats * norients * nscale);
  float* devCg = *p_devCg;


  size_t nbins = 25;

  float sigma = 0.05;
  bool blur = true;
  size_t border = 30;
  size_t borderWidth = width + 2 * border;
  size_t borderHeight = height + 2 * border;
  
  float* devGradients = gradients(devInput, nbins, blur, sigma, cgRadii, textonChoice);
  uint cueTimer;
  cutCreateTimer(&cueTimer);
  cutStartTimer(cueTimer);
  for(int scale = 0; scale < nscale; scale++) {
    int radius = cgRadii[scale];
    int length = 2*radius + 1;
    gpu_parabola(norients, width, height, border, &devGradients[borderWidth * borderHeight * norients * scale], radius, length, filters[scale], devCg + cuePitchInFloats * norients * scale, cuePitchInFloats);
  }
  cutStopTimer(cueTimer);
  printf(">+< \tCgsmooth: | %f | ms\n", cutGetTimerValue(cueTimer));
}

void tg(size_t width, size_t height, size_t norients, size_t nscale, size_t* tgRadii, float** filters, int* devTextons, float** p_devTg, int cuePitchInFloats, int textonChoice)
{
  cudaMalloc((void**)p_devTg, sizeof(float) * cuePitchInFloats * norients * nscale);
  float* devTg = *p_devTg;


  size_t nbins = 32;
  if (TEXTON64 == textonChoice)
      nbins = 64;

  float sigma = 0;
  bool blur = false;
  size_t border = 30;
 
  size_t borderWidth = width + 2 * border;
  size_t borderHeight = height + 2 * border;
  
  float* devGradients = gradients(devTextons, nbins, blur, sigma, tgRadii, textonChoice);  
  uint cueTimer;
  cutCreateTimer(&cueTimer);
  cutStartTimer(cueTimer);
  for(int scale = 0; scale < nscale; scale++) {
    int radius = tgRadii[scale];
    int length = 2*radius + 1;
    gpu_parabola(norients, width, height, border, &devGradients[borderWidth * borderHeight * norients * scale], radius, length, filters[scale], devTg + cuePitchInFloats * norients * scale, cuePitchInFloats);
  }
  cutStopTimer(cueTimer);
  printf(">+< \tTgsmooth: | %f | ms\n", cutGetTimerValue(cueTimer));
}


void localCues(int width, int height, float* devL, float* devA, float* devB, int* devTextons, float** devBg, float** devCga, float** devCgb, float** devTg, int* p_cuePitchInFloats, int p_nTextonChoice) {
  printf("Beginning Local cues computation\n");
  size_t norients = 8;
  size_t nscale = 3;
  size_t border = 30;
  size_t maxbins = 64;
  construct_parabola_filters(4, radii, norients);
  int nPixels = width * height;
  int cuePitchInFloats = findPitchInFloats(nPixels);
  *p_cuePitchInFloats = cuePitchInFloats;

  
  initializeGradients(width, height, border, maxbins, norients, nscale, p_nTextonChoice);
  gpu_parabola_init(norients, width, height, border);

  
  uint cueTimer;
  cutCreateTimer(&cueTimer);
  cutStartTimer(cueTimer);
  bg(width, height, norients, nscale, &radii[0], &filters[0], devL, devBg, cuePitchInFloats, p_nTextonChoice);
  cutStopTimer(cueTimer);
  printf(">+< \tBg: | %f | ms\n", cutGetTimerValue(cueTimer));
  cutResetTimer(cueTimer);
  cutStartTimer(cueTimer);
  cg(width, height, norients, nscale, &radii[1], &filters[1], devA, devCga, cuePitchInFloats, p_nTextonChoice);
  cutStopTimer(cueTimer);
  printf(">+< \tCga: | %f | ms\n", cutGetTimerValue(cueTimer));
  cutResetTimer(cueTimer);
  cutStartTimer(cueTimer);
  cg(width, height, norients, nscale, &radii[1], &filters[1], devB, devCgb, cuePitchInFloats, p_nTextonChoice);
  cutStopTimer(cueTimer);
  printf(">+< \tCgb: | %f | ms\n", cutGetTimerValue(cueTimer));
  cutResetTimer(cueTimer);
  cutStartTimer(cueTimer);
  tg(width, height, norients, nscale, &radii[1], &filters[1], devTextons, devTg, cuePitchInFloats, p_nTextonChoice);
  cutStopTimer(cueTimer);
  printf(">+< \tTg: | %f | ms\n", cutGetTimerValue(cueTimer));
  cutResetTimer(cueTimer);
 
  cudaThreadSynchronize();
  finalizeGradients();
  gpu_parabola_cleanup();
  printf("Completed Local cues\n");
}

/* int main(int argc, char** argv) */
/* { */
/*     norients = 8; */
/*     nbins = 25; */

/*     file = "data/L.dat"; */
/*     read_dims(); */
/*     read_parabola_filters(); */

/*     gpu_gradient_init(norients, width, height, border); */
/*     gpu_parabola_init(norients, width, height, border); */

/*     int ts1, ts2; */

/*     ts1 = timestamp(); */
/*     bg(); */
/*     cga(); */
/*     cgb(); */
/*     tg(); */
/*     ts2 = timestamp(); */

/*     printf("gpu_time = %fms\n", ((double)ts2-(double)ts1)/1000); */

/*     gpu_gradient_cleanup(); */
/*     gpu_parabola_cleanup(); */
/* } */
