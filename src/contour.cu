// vim: ts=4 syntax=cpp comments=

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <fcntl.h>
#include <float.h>
#include <unistd.h>
#include <damascene/texton.h>
#include <damascene/convert.h>
#include <damascene/intervening.h>
#include <damascene/lanczos.h>
#include <damascene/stencilMVM.h>

#include <damascene/localcues.h>
#include <damascene/combine.h>
#include <damascene/nonmax.h>
#include <damascene/spectralPb.h>
#include <damascene/globalPb.h>
#include <damascene/skeleton.h>
#include <damascene/util.h>
#include <damascene/ppm_util.h>

#define __TIMER_SPECFIC

#define TEXTON64 2
#define TEXTON32 1

float* loadArray(char* filename, uint& width, uint& height) {
  FILE* fp;
  fp = fopen(filename, "r");
  int dim;
  size_t b = fread(&dim, sizeof(int), 1, fp);
  assert(dim == 2);
  b = fread(&width, sizeof(int), 1, fp);
  b = fread(&height, sizeof(int), 1, fp);
  float* buffer = (float*)malloc(sizeof(float) * width * height);
  int counter = 0;
  for(uint col = 0; col < width; col++) {
    for(uint row = 0; row < height; row++) {
      float element;
      b = fread(&element, sizeof(float), 1, fp);
      counter++;
      buffer[row * width + col] = element;
    }
  }
  assert(b != 0);
 /*  for(int row = 0; row < height; row++) { */
/*     for(int col = 0; col < width; col++) { */
/*       printf("%f ", buffer[row*width + col]); */
/*     } */
/*     printf("\n"); */
/*   } */
  return buffer;
}

void writeTextImage(const char* filename, uint width, uint height, float* image) {
  FILE* fp = fopen(filename, "w");
  for(uint row = 0; row < height; row++) {
    for(uint col = 0; col < width; col++) {
      fprintf(fp, "%f ", image[row * width + col]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void writeFile(char* file, int width, int height, int* input)
{
    int fd;
    float* pb = (float*)malloc(sizeof(float)*width*height);
    for(int i = 0; i < width * height; i++) {
      pb[i] = (float)input[i];
    }
    fd = open(file, O_CREAT|O_WRONLY, 0666);
    size_t b = write(fd, &width, sizeof(int));
    b = write(fd, &height, sizeof(int));
    b = write(fd, pb, width*height*sizeof(float));
    assert(b != 0);
    close(fd);
}

void writeFile(char* file, int width, int height, float* pb)
{
    int fd;

    fd = open(file, O_CREAT|O_WRONLY, 0666);
    size_t b = write(fd, &width, sizeof(int));
    b = write(fd, &height, sizeof(int));
    b = write(fd, pb, width*height*sizeof(float));
    assert(b != 0);
    close(fd);
}

void writeGradients(char* file, int width, int height, int pitchInFloats, int norients, int scales, float* pb)
{
    int fd;

    fd = open(file, O_CREAT|O_WRONLY, 0666);
    size_t b = write(fd, &width, sizeof(int));
    b = write(fd, &height, sizeof(int));
    b = write(fd, &norients, sizeof(int));
    b = write(fd, &scales, sizeof(int));
    for(int scale = 0; scale < scales; scale++) {
      for(int orient = 0; orient < norients; orient++) {
        float* currentPointer = &pb[pitchInFloats * orient + pitchInFloats * scale * norients];
        b = write(fd, currentPointer, width*height*sizeof(float));
      }
    }
    assert(b != 0);
    close(fd);
}

void writeArray(char* file, int ndim, int* dim, float* input) {
  int fd;
  fd = open(file, O_CREAT|O_WRONLY|O_TRUNC, 0666);
  int size = 1;
  for(int i = 0; i < ndim; i++) {
    size *= dim[i];
  }
  size_t b = write(fd, &ndim, sizeof(int));
  b = write(fd, dim, sizeof(int) * ndim);
  b = write(fd, input, sizeof(float) * size);
  assert(b != 0);
  close(fd);
}

void transpose(int width, int height, float* input, float* output) {
  for(int row = 0; row < height; row++) {
    for(int col = 0; col < width; col++) {
      output[col * height + row] = input[row * width + col];
    }
  }                                         
}

void checkInputValue(int& nEigNum, float& fEigTolerance, int& nTextonChoice)
{
	if (nEigNum > 25)
	{
		printf("\nException: Do not support for more than 25 eigen vectors.\n");
		nEigNum = 25;
	}
	if (nEigNum < 2)
	{
		printf("\nException: Do not support for less than 2 eigen vectors.\n");
		nEigNum = 9;
	}
	if  (fEigTolerance < 1e-5)
	{
		printf("\nException: Do not support for accuracy below 1e-5.\n");
		fEigTolerance = 1e-4;
	}
	if  (fEigTolerance > 1e-1)
	{
		printf("\nException: Do not support for accuracy above 1e-1.\n");
		fEigTolerance = 1e-3;
	}
	if (nTextonChoice > 2 || nTextonChoice < 1)
	{
		printf("\nException: Only support choice 1 (32 bins) and choice 2 (64 bins)\n");
	}
}

void parsingCommand(int argc, char** argv, int& nEigNum, float& fEigTolerance, int& nTextonChoice)
{
	if (argc < 3)
	{
		nEigNum = 9;
		fEigTolerance = 1e-3;
		nTextonChoice = TEXTON32;
		return;
	}
	if (argc < 4)
	{
		nEigNum = atoi(argv[2]);
		fEigTolerance = 1e-3;
		nTextonChoice = TEXTON32;
		checkInputValue(nEigNum, fEigTolerance, nTextonChoice);
		return;
	}
	if (argc < 5)
	{
		nEigNum = atoi(argv[2]);
		fEigTolerance = atof(argv[3]);
		nTextonChoice = TEXTON32;
		checkInputValue(nEigNum, fEigTolerance, nTextonChoice);
		return;
	}

	if (argc < 6)
	{
		nEigNum = atoi(argv[2]);
		fEigTolerance = atof(argv[3]);
		nTextonChoice = atoi(argv[4]);
		checkInputValue(nEigNum, fEigTolerance, nTextonChoice);
		return;
	}

}


int main(int argc, char** argv) {
  cuInit(0);
  chooseLargestGPU(true);

  if (argc < 2) {
	printf("\nUsage: damascene input_image.ppm eigenvector_num eigenvector_tolerance texton_choice");
	printf("\nInput image should be in ppm format");
	printf("\nThe number of eigenvectors is from 2 to 25");
	printf("\nThe eigenvector tolerance is from 1e-2 to 1e-5");
	printf("\nFor the texton choice parameter, 1 for 32 bins, 2 for 64 bins\n");
    exit(1);
  }



  char* filename = argv[1];
  char outputPGMfilename[1000];
  char outputthinPGMfilename[1000];
  char outputPBfilename[1000];
  char outputthinPBfilename[1000];
  char outputgpbAllfilename[1000];
  printf("Processing: %s, output in ", filename);
  char* period = strrchr(filename, '.');
  if (period == 0) {
    period = strrchr(filename, 0);
  }
  strncpy(outputPGMfilename, filename, period - filename);
  sprintf(&outputPGMfilename[0] + (period - filename) , "Pb.pgm");
  strncpy(outputthinPGMfilename, filename, period - filename);
  sprintf(&outputthinPGMfilename[0] + (period - filename) , "Pbthin.pgm");
  
  strncpy(outputPBfilename, filename, period - filename);
  sprintf(&outputPBfilename[0] + (period - filename), ".pb");
  strncpy(outputthinPBfilename, filename, period - filename);
  sprintf(&outputthinPBfilename[0] + (period - filename), ".thin.pb");
  
  printf("%s and %s\n", outputPGMfilename, outputPBfilename);
  strncpy(outputgpbAllfilename, filename, period - filename);
  sprintf(&outputgpbAllfilename[0] + (period - filename), "GpbAll.ary");

  int nEigNum = 9;
  float fEigTolerance = 1e-3;
  int nTextonChoice = TEXTON32;

  parsingCommand(argc, argv, nEigNum, fEigTolerance, nTextonChoice);
  printf("\n Eig %d Tol %f Texton %d", nEigNum, fEigTolerance, nTextonChoice);

  
  uint width;
  uint height;
  uint* devRgbU;
  loadPPM_rgbU(filename, &width, &height, &devRgbU);
  int nPixels = width * height;
  printf("Image found: %i x %i pixels\n", width, height);
  assert(width > 0);
  assert(height > 0);

  size_t totalMemory, availableMemory;
  cuMemGetInfo(&availableMemory,&totalMemory );
  printf("Available %zu bytes on GPU\n", availableMemory);

  cuda_timer timer;
  timer.start();
#ifdef __TIMER_SPECFIC
  cuda_timer timer_specific;
  timer_specific.start();
#endif

  float* devGreyscale;
  rgbUtoGreyF(width, height, devRgbU, &devGreyscale);
  float time = 0;
#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< rgbUtoGrayF | %f | ms\n", time);
  timer_specific.start();
#endif

//   float* hostG = (float*)malloc(sizeof(float) * nPixels); 
//   CUDA_SAFE_CALL(cudaMemcpy(hostG, devGreyscale, height*width*sizeof(float),cudaMemcpyDeviceToHost));
//   cutSavePGMf("grey.pgm", hostG, width, height);
//   free(hostG);

  int* devTextons;
  findTextons(width, height, devGreyscale, &devTextons, nTextonChoice);
/*   int* hostTextons = (int*)malloc(sizeof(int)*width*height); */
/*   cudaMemcpy(hostTextons, devTextons, sizeof(int)*width*height, cudaMemcpyDeviceToHost); */
/*   writeFile("textons.pb", width, height, hostTextons); */

/*   float* hostFTextons = loadArray("goodTextons.dat", width, height); */
/*   printf("Host textons found %i width, %i height\n", width, height); */
/*   int * hostTextons = (int*)malloc(sizeof(float)*width*height); */
/*   for(int i = 0; i < width * height; i++) { */
/*     hostTextons[i] = (float)hostFTextons[i]; */
/*   } */
/*   int* devTextons; */
/*   cudaMalloc((void**)&devTextons, sizeof(int) * width * height); */
/*   cudaMemcpy(devTextons, hostTextons, sizeof(int) * width * height, cudaMemcpyHostToDevice); */
#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< texton | %f | ms\n", time);
  timer_specific.start();
#endif

  float* devL;
  float* devA;
  float* devB;
  rgbUtoLab3F(width, height, 2.5, devRgbU, &devL, &devA, &devB);

#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< rgbUtoLab3F | %f | ms\n", time);
  timer_specific.start();
#endif
  normalizeLab(width, height, devL, devA, devB);
#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< normalizeLab | %f | ms\n", time);
  timer_specific.start();
#endif
  int border = 30;
  float* devLMirrored;
  mirrorImage(width, height, border, devL, &devLMirrored);
/*   float* hostLMirrored = (float*)malloc(borderWidth * borderHeight * sizeof(float)); */
/*   cudaMemcpy(hostLMirrored, devLMirrored, borderWidth * borderHeight * sizeof(float), cudaMemcpyDeviceToHost); */
/*   writeFile("L.pb", borderWidth, borderHeight, hostLMirrored); */
 
  cudaThreadSynchronize();
  cudaFree(devRgbU);
  cudaFree(devGreyscale);
#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< mirrorImage | %f | ms\n", time);
  timer_specific.start();
#endif
  float* devBg;
  float* devCga;
  float* devCgb;
  float* devTg;
  int matrixPitchInFloats;
  cuda_timer local_cues_timer;
  local_cues_timer.start();

  localCues(width, height, devL, devA, devB, devTextons, &devBg, &devCga, &devCgb, &devTg, &matrixPitchInFloats, nTextonChoice);

  time = local_cues_timer.stop();
  printf("localcues time: %f seconds\n", time/1000.0);

#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< localcues | %f | ms\n", time);
  timer_specific.start();
#endif
   //float* hostG = (float*)malloc(sizeof(float) * nPixels); 
   //CUDA_SAFE_CALL(cudaMemcpy(hostG, devBg, height*width*sizeof(float),cudaMemcpyDeviceToHost));
   //cutSavePGMf("Bg.pgm", hostG, width, height);
   //free(hostG);

  cudaFree(devTextons);
  cudaFree(devL);
  cudaFree(devA);
  cudaFree(devB);
  
/*   int size = matrixPitchInFloats * 8 * 3 * sizeof(float); */
/*   float* hostBg = (float*)malloc(size); */
/*   float* hostCga = (float*)malloc(size); */
/*   float* hostCgb = (float*)malloc(size); */
/*   float* hostTg = (float*)malloc(size); */
/*   cudaMemcpy(hostBg, devBg, size, cudaMemcpyDeviceToHost); */
/*   cudaMemcpy(hostCga, devCga, size, cudaMemcpyDeviceToHost); */
/*   cudaMemcpy(hostCgb, devCgb, size, cudaMemcpyDeviceToHost); */
/*   cudaMemcpy(hostTg, devTg, size, cudaMemcpyDeviceToHost); */
/*   writeGradients("bg.gra", width, height, matrixPitchInFloats, 8, 3, hostBg); */
/*   writeGradients("cga.gra", width, height, matrixPitchInFloats, 8, 3, hostCga); */
/*   writeGradients("cgb.gra", width, height, matrixPitchInFloats, 8, 3, hostCgb); */
/*   writeGradients("tg.gra", width, height, matrixPitchInFloats, 8, 3, hostTg); */
  float* devMPbO;
  float *devCombinedGradient;
  combine(width, height, matrixPitchInFloats, devBg, devCga, devCgb, devTg, &devMPbO, &devCombinedGradient, nTextonChoice);

#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< combine | %f | ms\n", time);
  timer_specific.start();
#endif

  CUDA_SAFE_CALL(cudaFree(devBg));
  CUDA_SAFE_CALL(cudaFree(devCga));
  CUDA_SAFE_CALL(cudaFree(devCgb));
  CUDA_SAFE_CALL(cudaFree(devTg));

  float* devMPb;
  cudaMalloc((void**)&devMPb, sizeof(float) * nPixels);
  nonMaxSuppression(width, height, devMPbO, matrixPitchInFloats, devMPb);

#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< nonmaxsuppression | %f | ms\n", time);
  timer_specific.start();
#endif
  
  //int devMatrixPitch = matrixPitchInFloats * sizeof(float);
  int radius = 5;
  //int radius = 10;

  Stencil theStencil(radius, width, height, matrixPitchInFloats);
  float* devMatrix;
  intervene(theStencil, devMPb, &devMatrix);
  printf("Intervening contour completed\n");
 
#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< intervene | %f | ms\n", time);
  timer_specific.start();
#endif

  float* eigenvalues;
  float* devEigenvectors;
  //int nEigNum = 17;
  generalizedEigensolve(theStencil, devMatrix, matrixPitchInFloats, nEigNum, &eigenvalues, &devEigenvectors, fEigTolerance);

#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< generalizedEigensolve | %f | ms\n", time);
  timer_specific.start();
#endif
  float* devSPb = 0;
  size_t devSPb_pitch = 0;
  CUDA_SAFE_CALL(cudaMallocPitch((void**)&devSPb, &devSPb_pitch, nPixels *  sizeof(float), 8));
  cudaMemset(devSPb, 0, matrixPitchInFloats * sizeof(float) * 8);

  spectralPb(eigenvalues, devEigenvectors, width, height, nEigNum, devSPb, matrixPitchInFloats);

#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< spectralPb | %f | ms\n", time);
  timer_specific.start();
#endif
  float* devGPb = 0;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devGPb, sizeof(float) * nPixels));
  float* devGPball = 0;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devGPball, sizeof(float) * matrixPitchInFloats * 8));
  //StartCalcGPb(nPixels, matrixPitchInFloats, 8, devbg1, devbg2, devbg3, devcga1, devcga2, devcga3, devcgb1, devcgb2, devcgb3, devtg1, devtg2, devtg3, devSPb, devMPb, devGPball, devGPb);
  StartCalcGPb(nPixels, matrixPitchInFloats, 8, devCombinedGradient, devSPb, devMPb, devGPball, devGPb);
 
#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< StartCalcGpb | %f | ms\n", time);
  timer_specific.start();
#endif
  float* devGPb_thin = 0;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devGPb_thin, nPixels * sizeof(float) ));
  PostProcess(width, height, width, devGPb, devMPb, devGPb_thin); //note: 3rd param width is the actual pitch of the image
  NormalizeGpbAll(nPixels, 8, matrixPitchInFloats, devGPball);
  
  cudaThreadSynchronize();
  float all_time = timer.stop();
  printf("CUDA Status : %s\n", cudaGetErrorString(cudaGetLastError()));

#ifdef __TIMER_SPECFIC
  time = timer_specific.stop();
  printf(">+< PostProcess | %f | ms\n", time);
#endif
  printf(">+< Computation time: | %f | seconds\n", all_time/1000.0);
  float* hostGPb = (float*)malloc(sizeof(float)*nPixels);
  memset(hostGPb, 0, sizeof(float) * nPixels);
  cudaMemcpy(hostGPb, devGPb, sizeof(float)*nPixels, cudaMemcpyDeviceToHost);
  
  savePGMf(outputPGMfilename, hostGPb, width, height);
  writeFile(outputPBfilename, width, height, hostGPb);

  /* thin image */
  float* hostGPb_thin = (float*)malloc(sizeof(float)*nPixels);
  memset(hostGPb_thin, 0, sizeof(float) * nPixels);
  cudaMemcpy(hostGPb_thin, devGPb_thin, sizeof(float)*nPixels, cudaMemcpyDeviceToHost);
  savePGMf(outputthinPGMfilename, hostGPb_thin, width, height);
  writeFile(outputthinPBfilename, width, height, hostGPb);
  free(hostGPb_thin);
  /* end thin image */

  float* hostGPbAll = (float*)malloc(sizeof(float) * matrixPitchInFloats * 8);
  cudaMemcpy(hostGPbAll, devGPball, sizeof(float) * matrixPitchInFloats * 8, cudaMemcpyDeviceToHost);
  //int oriMap[] = {0, 1, 2, 3, 4, 5, 6, 7};
  //int oriMap[] = {4, 5, 6, 7, 0, 1, 2, 3};
  int oriMap[] = {3, 2, 1, 0, 7, 6, 5, 4};
  float* hostGPbAllConcat = (float*)malloc(sizeof(float) * width * height * 8);
  for(int i = 0; i < 8; i++) {
    transpose(width, height, hostGPbAll + matrixPitchInFloats * oriMap[i], hostGPbAllConcat + width * height * i);
  }
  int dim[3];
  dim[0] = 8; 
  dim[1] = width;
  dim[2] = height;
  writeArray(outputgpbAllfilename, 3, dim, hostGPbAllConcat);
  
  /*
  for(int orientation = 0; orientation < 8; orientation++) {
    sprintf(orientationIndicator, "_%i_Pb.pgm", orientation);
    cutSavePGMf(outputPGMAllfilename, hostGPbAll + matrixPitchInFloats * orientation, width, height);
  }
  */

  free(hostGPb);
  free(hostGPbAll);
  free(hostGPbAllConcat);
  
/*   filename = "polynesiaPb.txt"; */
/*   writeTextImage(filename, width, height, hostGPb);  */
/*   int getNEigs = 9; */
/*   FILE* fp; */
/*   fp = fopen("eigenVectors.txt", "w"); */
/* 	//Print out the eigenvectors */
/*   for (int j = 0; j < nPixels; j++) { */
/*     for (int i = 0; i < getNEigs; i++) { */
/*       fprintf(fp, "%f ", eigenvectors[i*nPixels+j]); */
/*     } */
/*     fprintf(fp, "\n"); */
/*   } */
/*   fclose(fp); */

/*   fp = fopen("eigenValues.txt", "w"); */
/* 	for (int i = 0; i < getNEigs; i++) { */
/* 		fprintf(fp, "%e\n", eigenvalues[i]); */
/* 	} */
/* 	fclose(fp); */

/*  CUDA_SAFE_CALL(cudaFree(devBgcombined));
  CUDA_SAFE_CALL(cudaFree(devCgacombined));
  CUDA_SAFE_CALL(cudaFree(devCgbcombined));
  CUDA_SAFE_CALL(cudaFree(devTgcombined));*/

  CUDA_SAFE_CALL(cudaFree(devEigenvectors));
  CUDA_SAFE_CALL(cudaFree(devCombinedGradient));
  CUDA_SAFE_CALL(cudaFree(devSPb));
  CUDA_SAFE_CALL(cudaFree(devGPb));
  CUDA_SAFE_CALL(cudaFree(devGPb_thin));
  CUDA_SAFE_CALL(cudaFree(devGPball));

}
