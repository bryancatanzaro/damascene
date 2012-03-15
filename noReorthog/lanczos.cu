// vim: ts=4 syntax=cpp comments=


#define MAXITER 6000
#define CHECKITER 500
#define LUMPTOL 1e-5
#define TOLERANCE 1e-3
//#define TOLERANCE 3.46e-4
//#define TOLERANCE 1e-5
#define SPURTOLERANCE 1e-10
//#define EIGNUM 9 
#define MAXEIGNUM 25

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <getopt.h>
#include <time.h>
#include <math.h>
#include "cublas.h"
#include <acml.h>
#include <vector>
#include "stencilMVM.h"


typedef std::vector<float> floatVector;
typedef std::vector<double> doubleVector;
typedef std::vector<bool> boolVector;

float getTimeUs(struct timeval start, struct timeval stop) {
  return (stop.tv_sec - start.tv_sec) * 1e6f + ((float)(stop.tv_usec - start.tv_usec));
}

void PrintVectorOnFile(int p_nSize, float* vec, char* filename)
{
	FILE* fo = fopen(filename, "w");
	float* tempVec = (float*) malloc(p_nSize*sizeof(float));
	CUDA_SAFE_CALL(cudaMemcpy(tempVec, vec, p_nSize*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < p_nSize; i++)
		fprintf(fo, "%f\n", tempVec[i]);
	fclose(fo);
	free(tempVec);
}

void clearTestMatrix(float* sMatrixValues) {
  free(sMatrixValues);
}

void initEigs(int p_nEigNum, int p_nMatrixDimension, float** p_eigenValues, float** devEigVectors)
{
  (*p_eigenValues) = (float*) malloc(p_nEigNum * sizeof(float));
  CUDA_SAFE_CALL(cudaMalloc((void**)devEigVectors, p_nMatrixDimension * sizeof(float)*p_nEigNum);)
  //memset(*p_eigenValues, 0, p_nEigNum * sizeof(float));
  //memset(*p_eigenVectors, 0, p_nEigNum * p_nMatrixDimension * sizeof(float));
}

void clearEigs(float* p_eigenValues, float* p_eigenVectors)
{
  free(p_eigenValues);
  free(p_eigenVectors);
}

void initStartingVector(int p_nMatrixDimension, float* p_aaDMatrixV)
{
	// Use [1 1 ... 1] as the starting vector
	float fValue = 1/sqrt(p_nMatrixDimension);
	for (int i = 0; i < p_nMatrixDimension; i++)
	{
		p_aaDMatrixV[i] = fValue;
	}	
}

void lanczosInit(int p_nEigNum, int p_nMatrixDimension, float** p_aInitVector, float** p_aBeta, float** p_aAlpha,
                 float** p_aTEigVals, float** p_aaTEigVecs)
{
	cublasStatus status;
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("!!!! CUBLAS initialization error\n");
		return;
	}

	(*p_aInitVector) = (float*) malloc(p_nMatrixDimension * sizeof(float));
	(*p_aBeta) = (float*) malloc(MAXITER * sizeof(float));
	(*p_aAlpha)= (float*) malloc(MAXITER * sizeof(float));
	(*p_aTEigVals) = (float*) malloc(MAXITER * sizeof(float));
	(*p_aaTEigVecs)= (float*) malloc(MAXITER * MAXITER * sizeof(float));

	initStartingVector(p_nMatrixDimension, (*p_aInitVector));

}

void lanczosClear(float* p_aInitVector, float* p_aBeta, float* p_aAlpha, float* p_aTEigVals, float* p_aaTEigVecs)
{

  free(p_aInitVector);
  free(p_aBeta);
  free(p_aAlpha);
  free(p_aTEigVals);
  free(p_aaTEigVecs);

  cublasStatus status;
  status = cublasShutdown();
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf ("!!!! shutdown error (A)\n");
    return;
  }

}


bool TestForConvergence(int p_nEigNum, int p_nMatrixDimension, int p_nIter, 
                        float* p_aBeta, float* p_aaTEigVecs, 
						float* p_daaDMatrixV, float* p_daVectorS, float* p_daVectorX, float p_fTolerance)
{
	// Test whether the residual of the first p_nEigNum eigenvalues are all <= TOLERANCE
	for (int i = p_nEigNum - 1; i >=0 ; i--) {
    float absoluteResidual = abs(p_aBeta[p_nIter]*p_aaTEigVecs[i * (p_nIter + 1) + p_nIter]);
		if (absoluteResidual > p_fTolerance) {
      printf("Eigenvalue: %d has too large a residual %e\n", i, absoluteResidual);
                                                                            
			return false;
    }
  }
 /*  for (int i = p_nEigNum - 1; i >= 0; i--) { */
/* 		cublasSetVector(p_nIter+1, sizeof(float), p_aaTEigVecs+i*(p_nIter+1), 1, p_daVectorS, 1); */
/* 		cublasSgemv('n', p_nMatrixDimension, p_nIter+1, 1, p_daaDMatrixV, p_nMatrixDimension, p_daVectorS, 1, */
/* 				0, p_daVectorX, 1); */

/* 		float fNorm = cublasSnrm2(p_nMatrixDimension, p_daVectorX, 1); */

/* 		if (abs(p_aBeta[p_nIter]*p_aaTEigVecs[i * (p_nIter + 1) + p_nIter])/fNorm > TOLERANCE) */
/* 			return false; */
/* 	} */
	

	return true;
}



void calcEigs(int width, int height, int p_nEigNum, int p_nMatrixDimension, int p_nIter, 
              float* p_eigenValues, float* p_eigenVectors, 
              float* p_aTEigVals, float* p_aaTEigVecs, 
              float* p_daaDMatrixV, float* devRSqrtSum)
{
	int tVecLength = p_nIter + 1;
	size_t tEigenVectorPitch;
	float* devTEigVecs;
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&devTEigVecs, &tEigenVectorPitch, tVecLength * sizeof(float), p_nEigNum));

	float* devEigVecs;
	size_t eigenVectorPitch;
	cudaError_t cuerror;
	cuerror = cudaMallocPitch((void**)&devEigVecs, &eigenVectorPitch, p_nMatrixDimension * sizeof(float), p_nEigNum);
	if (cuerror != cudaSuccess)
	{
		printf("\nAlloc failed!");

	}
	//cudaMalloc((void**)&devEigVecs,p_nMatrixDimension * sizeof(float)* p_nEigNum);

	cudaMemcpy2D(devTEigVecs, tEigenVectorPitch, p_aaTEigVecs, tVecLength * sizeof(float), tVecLength * sizeof(float), p_nEigNum, cudaMemcpyHostToDevice);
	cublasSgemm('n', 'n', p_nMatrixDimension, p_nEigNum, tVecLength, 1.0f, p_daaDMatrixV, p_nMatrixDimension, devTEigVecs, tEigenVectorPitch/sizeof(float), 0.0f, devEigVecs, p_nMatrixDimension);

        //printf("multiplied %d vectors \n", tVecLength);
	//dim3 blockDim = dim3(256, 1);
	//dim3 gridDim = dim3((p_nMatrixDimension - 1)/256 + 1, 1);
	
	dim3 blockDim(XBLOCK, YBLOCK);
	dim3 gridDim((width - 1)/XBLOCK + 1, (height - 1)/(YBLOCK) + 1);
	//generalizeVectors<<<gridDim, blockDim>>>(p_nMatrixDimension, p_nEigNum, devEigVecs, eigenVectorPitch/sizeof(float), devRSqrtSum);
	scaleEigByD<<<gridDim, blockDim>>>(width, height, devRSqrtSum, devEigVecs, p_nEigNum);
	//cudaMemcpy2D(p_eigenVectors, p_nMatrixDimension * sizeof(float), devEigVecs, eigenVectorPitch, p_nMatrixDimension * sizeof(float), p_nEigNum, cudaMemcpyDeviceToHost);

	cudaMemcpy(p_eigenVectors, devEigVecs,  p_nMatrixDimension * sizeof(float)*p_nEigNum, cudaMemcpyDeviceToHost);

}


/// function lanczos iteration does one iteration
/// of computation in the lanczos algorithm
/// @param d_aaDMatrixV the matrix of lanczos vectors 

void lanczosIteration(float* d_aaDMatrixV, int k, int i, float* d_aVectorQQ, 
                      float* d_aVectorQQPrev, float* d_aVectorZ, float* aBeta, float* aAlpha,
                      int p_nMatrixDimension, int width, int height, int nPixels, int nDiags,
                      int nDimUnroll, float *devVector, float* devMatrix, int matrixPitchInFloats,
                      dim3 gridDim, dim3 blockDim, int maxIterationsThatFitGPU,  int storeVectors=0, float*RitzVectors=0, int p_nEigNum=0, float* p_eigenVectors=0, size_t eigenVectorPitch=0, int nIterations=MAXITER)
{
        int iteration= i+k*(maxIterationsThatFitGPU-1); // actual iteration number

        int read, write;
///        if(k)
///        {
            read = (i+maxIterationsThatFitGPU)%(maxIterationsThatFitGPU+1);;
            write = (i+1)%(maxIterationsThatFitGPU+1);

///        }
///        else
///        {
///            if(i<=maxIterationsThatFitGPU)
///            {
///                read = (i+maxIterationsThatFitGPU)%(maxIterationsThatFitGPU+1);
///                write = (i+1);
///            }
///            else
///            {
///                read = maxIterationsThatFitGPU + ((i+1)%2);
///                write = maxIterationsThatFitGPU + ((i+1)%2);
///            }
///        }
///
	//printf("Iteration %d : Read from %d -- Write to %d storing = %d\n", iteration, read, write, storeVectors );
        CUDA_SAFE_CALL(cudaMemcpy(devVector, d_aVectorQQ, nPixels * sizeof(float), cudaMemcpyDeviceToDevice));

        //r= A*qq 
	stencilMVM<<<gridDim, blockDim>>>(width, height, nPixels, nDiags, nDimUnroll, 
        devMatrix, matrixPitchInFloats, d_aVectorZ);

	if (iteration > 0)
	{
		//cublasScopy(p_nMatrixDimension, d_aaDMatrixV + (i-1)*p_nMatrixDimension, 1, d_aVectorQQPrev, 1);
		//cublasScopy(p_nMatrixDimension, d_aaDMatrixV + p_nMatrixDimension, 1, d_aVectorQQPrev, 1);
		//r = r - v(i-1)*beta(i-1)
		//cublasSaxpy(p_nMatrixDimension, (-1) * aBeta[i-1], d_aVectorQQPrev, 1, d_aVectorZ, 1);

		cublasSaxpy(p_nMatrixDimension, (-1) * aBeta[iteration-1], d_aaDMatrixV + read*p_nMatrixDimension, 1, d_aVectorZ, 1);
	}
	//alpha(i) = v(i) * r
        float oldalpha=aAlpha[iteration];
	aAlpha[iteration] = cublasSdot(p_nMatrixDimension, d_aVectorQQ, 1, d_aVectorZ, 1);
        if(storeVectors && iteration<nIterations) assert(oldalpha == aAlpha[iteration]);
	//r = r - v(j) * alpha(j)
	cublasSaxpy(p_nMatrixDimension, (-1) * aAlpha[iteration], d_aVectorQQ, 1, d_aVectorZ, 1);

	//Reorthogonalization goes here, but we're not doing it
	//beta(j) = norm2(r)
	aBeta[iteration] = cublasSnrm2(p_nMatrixDimension, d_aVectorZ, 1);
	//v(j+1) = r / beta(j)
	cublasScopy(p_nMatrixDimension, d_aVectorZ, 1, d_aVectorQQ, 1);
	cublasSscal(p_nMatrixDimension, 1/aBeta[iteration], d_aVectorQQ, 1);

	//cublasScopy(p_nMatrixDimension, d_aVectorQQ, 1, d_aaDMatrixV + ((i+1)%2)*p_nMatrixDimension, 1);
	cublasScopy(p_nMatrixDimension, d_aVectorQQ, 1, d_aaDMatrixV + write*p_nMatrixDimension, 1);

	if(i==maxIterationsThatFitGPU-1 || iteration>=nIterations-1)
	{
		//cudamemcpy to CPU --all the lanczos vectors
                if(storeVectors)
                {
                    assert(RitzVectors != NULL);
                    assert(p_eigenVectors != NULL);
                    int IterationsToDo = (i==maxIterationsThatFitGPU-1)?(maxIterationsThatFitGPU-1):(nIterations+1-k*(maxIterationsThatFitGPU-1));
                    
                    float *RitzGPU=0;
                    size_t RitzGPUPitch;
                    CUDA_SAFE_CALL(cudaMallocPitch((void**)&RitzGPU, &RitzGPUPitch, sizeof(float)*IterationsToDo, p_nEigNum));
                    CUDA_SAFE_CALL(cudaMemcpy2D(RitzGPU, RitzGPUPitch, RitzVectors+k*(maxIterationsThatFitGPU-1), (nIterations+1)*sizeof(float), IterationsToDo*sizeof(float), p_nEigNum, cudaMemcpyHostToDevice));
                    assert(RitzGPU != NULL); 
                    
                    cublasSgemm('n','n',p_nMatrixDimension, p_nEigNum, IterationsToDo,  1.0, d_aaDMatrixV, p_nMatrixDimension, RitzGPU, RitzGPUPitch/sizeof(float), 1.0, p_eigenVectors, p_nMatrixDimension );
                   
                    
                    CUDA_SAFE_CALL(cudaFree(RitzGPU));
                    //printf("multiplied 0-%d with ritz vectors %d-%d \n", IterationsToDo-1, k*(maxIterationsThatFitGPU-1),k*(maxIterationsThatFitGPU-1)+IterationsToDo-1);
                
                }

		CUDA_SAFE_CALL(cudaMemcpy( d_aaDMatrixV ,  d_aaDMatrixV + (i)*p_nMatrixDimension, sizeof(float)*p_nMatrixDimension*2, cudaMemcpyDeviceToDevice));
		//printf("Copied %d %d to 0 1\n", i, i+1);
	}


}

bool CullumDevice(int i, float* aAlpha, float*aBeta, double* tempAlpha, double* tempBeta, int eigCheck, float* aTEigVals, float* aaTEigVecs, int p_nEigNum, float* p_eigenValues, double *tvectors, char range, char order, double vl, double vu, int il, int iu, double abstol, int nsplit, double* w, int *iblock, int* isplit, double* work, int* iwork, int* ifail )
{
	int m, info;
	int tempn = i;
	for(int j = 0; j < tempn; j++) {
		tempAlpha[j] = (double)aAlpha[j + 1];
	}
	for(int j = 0; j < tempn - 1; j++) {
		tempBeta[j] = (double)aBeta[j + 1];
	}

	doubleVector* currentCullum = new doubleVector();

	dstebz_(&range, &order, &tempn, &vl, &vu, &il, &iu, &abstol, tempAlpha, tempBeta, &m, &nsplit, w, iblock, isplit, work, iwork, &info, 1, 1); 
	for (int j = 0; j < eigCheck; j++) {
		currentCullum->push_back(w[j]);
	}
	//cullumValues.push_back(currentCullum);


	tempn = i+1;
	for(int j = 0; j < tempn; j++) {
		tempAlpha[j] = (double)aAlpha[j];
	}
	for(int j = 0; j < tempn - 1; j++) {
		tempBeta[j] = (double)aBeta[j];
	}

	dstebz_(&range, &order, &tempn, &vl, &vu, &il, &iu, &abstol, tempAlpha, tempBeta, &m, &nsplit, w, iblock, isplit, work, iwork, &info, 1, 1); 
	doubleVector* currentRitz = new doubleVector();
	doubleVector acceptedEigVals;
	boolVector duplicates;
	for (int j = 0; j < eigCheck; j++) {
		currentRitz->push_back(w[j]);
		bool accept = true;
		if (j > 0) {
			if (currentRitz->operator[](j) - currentRitz->operator[](j-1) < LUMPTOL) {
				accept = false;
				boolVector::reverse_iterator lastDuplicate = duplicates.rbegin();
				*lastDuplicate = true;
			}
		}

		if (accept) {
			acceptedEigVals.push_back(w[j]);
			duplicates.push_back(false);
		}
	}


	doubleVector screenedEigVals;
	for (int j = 0; j < acceptedEigVals.size(); j++) {
		double candidateValue = acceptedEigVals[j];
		bool accept = true;
		if (!duplicates[j]) {
			for (doubleVector::iterator kt = currentCullum->begin(); kt != currentCullum->end(); kt++) {
				double closeness = abs((candidateValue - *kt)/candidateValue);
				if (closeness <= SPURTOLERANCE) {
					accept = false;
				}
			}
		}
		if (accept) {
			screenedEigVals.push_back(candidateValue);
		}
	}


	//ritzValues.push_back(currentRitz);



	printf("Screened Eigenvalues: \n");
	int j = 0;
	for(doubleVector::iterator jt = screenedEigVals.begin(); j < p_nEigNum&&jt!=screenedEigVals.end(); jt++) {
		printf("%e ", *jt);
		w[j] = *jt;
		p_eigenValues[j] = aTEigVals[j] = *jt;
		j++;
	}
	printf("\n");

	if (screenedEigVals.size() < p_nEigNum)
		return false;
	assert (screenedEigVals.size() >= p_nEigNum); //--uncomment later 
	int getNEig = p_nEigNum;
	for(int j = 0; j < tempn; j++) {
		tempAlpha[j] = (double)aAlpha[j];
	}
	for(int j = 0; j < tempn - 1; j++) {
		tempBeta[j] = (double)aBeta[j];
	}

	assert(w!=NULL);
	dstein_(&tempn, tempAlpha, tempBeta, &getNEig, w, iblock, isplit, tvectors, &tempn, work, iwork, ifail, &info);
	for(int j = 0; j < getNEig; j++) {
		for(int k = 0; k < tempn; k++) {
			aaTEigVecs[j * tempn + k] = (float)tvectors[j * tempn + k];
		}
	}
	delete currentCullum;
	delete currentRitz;

	return true;

}

void lanczos(int p_nMatrixDimension, dim3 gridDim, dim3 blockDim,
             Stencil* theStencil, float* devMatrix,
             int p_nEigNum, float* p_eigenValues, float* devEigVectors, int p_nOrthoChoice, float* devRSqrtSum, float p_fTolerance)
{
	float* aInitVector;
	float* aBeta;
	float* aAlpha;
	float* aTEigVals;
	float* aaTEigVecs;
	int nIter = 0;

	float* d_aVectorZ = 0;
	float* d_aVectorQQ = 0;
	float* d_aVectorQQPrev = 0;
	float* d_aaDMatrixV = 0;
	float* d_aVectorT1 = 0;
	float* d_aVectorT2 = 0;
        float* devVector = 0;

	lanczosInit(p_nEigNum, p_nMatrixDimension, &aInitVector, &aBeta, &aAlpha, &aTEigVals, &aaTEigVecs);
	cudaError_t ce;
	ce = cudaGetLastError();
	if(ce != cudaSuccess)
	{
		printf("Error in line %d in %s : %s\n",__LINE__,__FILE__, cudaGetErrorString(ce));
		//return;
	}
        size_t totalMemory, availableMemory;
        cuMemGetInfo(&availableMemory,&totalMemory );
        printf("Available %u bytes on GPU\n", availableMemory);

        float margin = 0.9;
        int maxIterationsThatFitGPU;
        do {
      
            maxIterationsThatFitGPU = int(margin* (float(availableMemory/sizeof(float)-p_nMatrixDimension*(p_nEigNum+6)))/(p_nEigNum+p_nMatrixDimension));
            printf("Can fit %d iterations on GPU\n", maxIterationsThatFitGPU);
       
            // to do: write code to free memory before allocating for the next iterations.. 

            cublasAlloc(p_nMatrixDimension * (maxIterationsThatFitGPU+ 1), sizeof(float), (void**)&d_aaDMatrixV);
            cublasAlloc(p_nMatrixDimension, sizeof(float), (void**)&d_aVectorZ);
            cublasAlloc(p_nMatrixDimension, sizeof(float), (void**)&d_aVectorQQ);
            cublasAlloc(p_nMatrixDimension, sizeof(float), (void**)&d_aVectorQQPrev);
            cublasAlloc(p_nMatrixDimension, sizeof(float), (void**)&d_aVectorT1);
            cublasAlloc(p_nMatrixDimension, sizeof(float), (void**)&d_aVectorT2);

            cudaMalloc((void**)&devVector, p_nMatrixDimension * sizeof(float));

            ce = cudaGetLastError();
            if(ce != cudaSuccess)
            {
                    //printf("Error in line %d in %s : %s\n",__LINE__,__FILE__,cudaGetErrorString(ce));
                    printf("Cuda alloc failed -- trying to make do with less memory \n");
		
            }
            margin = margin-0.1;

        } while(ce != cudaSuccess && margin>0);

        if(margin <= 0)
        {
            printf("Aborted due to insufficient memory \n");
            exit(-1);
        }

        //float *h_aaDMatrixV = malloc(sizeof(float)*p_nMatrixDimension*(MAXITER+1));

	cublasSetVector(p_nMatrixDimension, sizeof(float), aInitVector, 1, d_aaDMatrixV, 1);
	cublasScopy(p_nMatrixDimension, d_aaDMatrixV, 1, d_aVectorQQ, 1);
	int eigCheck = p_nEigNum + 20;//p_nEigNum * 5;
	int n = MAXITER + 1;
	char range = 'I';
	char order = 'E';
	double vl;
	double vu;
	int il = 1;
	int iu = eigCheck;
	double abstol = 0.0;
	int m;
	int nsplit;
	double* w = (double*)malloc(sizeof(double) * n);
	int* iblock = (int*)malloc(sizeof(int) * n);
	int* isplit = (int*)malloc(sizeof(int) * n);
	double* work = (double*)malloc(sizeof(double) * 5 * n);
	int* iwork = (int*)malloc(sizeof(int) * 3 * n);
	int* ifail = (int*)malloc(sizeof(int) * eigCheck);
	int info;

	double* tempAlpha = (double*)malloc(sizeof(double) * n);
	double* tempBeta = (double*)malloc(sizeof(double) * n);
	double* tvectors = (double*)malloc(sizeof(double) * n * p_nEigNum);

	int width = theStencil->getWidth();
	int height = theStencil->getHeight();
	int nDiags = theStencil->getStencilArea();
	int nPixels = width * height;
	int radius = theStencil->getRadius();
	int nDimUnroll = findNDimUnroll(nDiags);
	int matrixPitchInFloats = findPitchInFloats(nPixels);
/* 	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); */
		
/* 	size_t offset = 0; */
/* 	cudaBindTexture(&offset, &texVector, devVector, &channelDesc, nPixels * sizeof(float)); */

  bindTexture(devVector, nPixels);

	std::vector<floatVector*> times;
	std::vector<doubleVector*> ritzValues;
	std::vector<doubleVector*> cullumValues;
        
        
        struct timeval lanczosTimeStart;
	gettimeofday(&lanczosTimeStart, 0);
	int i;

	floatVector* currentIteration = new floatVector();

	for (int iter = 0; iter < MAXITER; iter++)
	{
		i = iter;
		if ((i % 100) == 0) {
			printf("lanczos iteration: %d\n", i);
		}
		struct timeval start;
		/*cudaThreadSynchronize();*/ gettimeofday(&start, 0);
		
		lanczosIteration(d_aaDMatrixV, 0, iter, d_aVectorQQ, d_aVectorQQPrev, 
                                 d_aVectorZ, aBeta, aAlpha, p_nMatrixDimension, width, height,
                                 nPixels, nDiags, nDimUnroll, devVector, devMatrix, 
                                 matrixPitchInFloats, gridDim, blockDim, maxIterationsThatFitGPU, 0);


		if (((i+1) % CHECKITER == 0) || (i == MAXITER - 1) || (i == maxIterationsThatFitGPU-2))
		{


			while (1)
			{
				bool success = CullumDevice(i, aAlpha, aBeta, tempAlpha, tempBeta, eigCheck, aTEigVals, aaTEigVecs, p_nEigNum, p_eigenValues, tvectors, range, order, vl,  vu, il, iu, abstol, nsplit, w, iblock, isplit, work, iwork, ifail );
				if (success)
					break;
				eigCheck += 20;
				iu = eigCheck;
				ifail = (int*) realloc(ifail, sizeof(int) * eigCheck);
				assert(eigCheck <= 1000);
				printf("Screened Eig number too small, enlarge as %d\n", eigCheck);

			}
                        
			//Test bounds for convergence
			if (TestForConvergence(p_nEigNum, p_nMatrixDimension, i, aBeta, aaTEigVecs, d_aaDMatrixV, d_aVectorT1, d_aVectorT2, p_fTolerance)) {
				printf("Converged\n");
				break;
			}

		}

		struct timeval convergence;
		/*cudaThreadSynchronize();*/ gettimeofday(&convergence, 0);
		//currentIteration->push_back(getTimeUs(four64, convergence));

		//cublasScopy(p_nMatrixDimension, d_aVectorQQ, 1, d_aaDMatrixV + (i+1)*p_nMatrixDimension, 1);
		struct timeval stop;
		/*cudaThreadSynchronize();*/ gettimeofday(&stop, 0);
		currentIteration->push_back(getTimeUs(convergence, stop));
		currentIteration->push_back(getTimeUs(start, stop));
		times.push_back(currentIteration);
                ce = cudaGetLastError();
                if(ce != cudaSuccess)
                {
                    printf("Error %d in %s : %s\n",__LINE__,__FILE__, cudaGetErrorString(ce));
                    //return;
                }
                

	}


        printf("nIterations = %d\n", i+1);

        size_t eigenVectorPitch;
        CUDA_SAFE_CALL(cudaMemset(devEigVectors, 0, p_nMatrixDimension * sizeof(float)*p_nEigNum));
        //CUDA_SAFE_CALL(cudaMallocPitch((void**)&devEigVectors, &eigenVectorPitch, p_nMatrixDimension * sizeof(float), p_nEigNum));
        //CUDA_SAFE_CALL(cudaMemset(devEigVectors, 0, eigenVectorPitch*p_nEigNum));

        if(i < maxIterationsThatFitGPU)
        {
            lanczosIteration(d_aaDMatrixV, 0 , i+1, d_aVectorQQ, d_aVectorQQPrev, 
                                 d_aVectorZ, aBeta, aAlpha, p_nMatrixDimension, width, height,
                                 nPixels, nDiags, nDimUnroll, devVector, devMatrix, 
                                 matrixPitchInFloats, gridDim, blockDim, maxIterationsThatFitGPU, 1, 
                                 aaTEigVecs , p_nEigNum, devEigVectors, eigenVectorPitch, i);

        }
        else
        {
        
            cublasSetVector(p_nMatrixDimension, sizeof(float), aInitVector, 1, d_aaDMatrixV, 1);
            cublasScopy(p_nMatrixDimension, d_aaDMatrixV, 1, d_aVectorQQ, 1);
    
            int nIterations = i;
            int cycle;
            int done=0;
            int iter;
    
            for(cycle = 0; cycle <MAXITER/maxIterationsThatFitGPU; cycle ++)
            {
                iter=(cycle==0)?0:1;
                for( ;iter<maxIterationsThatFitGPU; iter++)
                {
                    i = cycle*(maxIterationsThatFitGPU-1)+iter;
                    if(i < nIterations)
                    {
                        lanczosIteration(d_aaDMatrixV, cycle, iter, d_aVectorQQ, d_aVectorQQPrev, 
                                    d_aVectorZ, aBeta, aAlpha, p_nMatrixDimension, width, height,
                                    nPixels, nDiags, nDimUnroll, devVector, devMatrix, 
                                    matrixPitchInFloats, gridDim, blockDim, maxIterationsThatFitGPU, 1, 
                                    aaTEigVecs , p_nEigNum, devEigVectors, eigenVectorPitch, nIterations);
                                    //aaTEigVecs , p_nEigNum, 0, eigenVectorPitch, nIterations);
                    }
                    else
                    {
                        done=1;
                        break;
                    }

                }
                if(done)
                {
                    break;
                }
            }
        }
        
        cudaThreadSynchronize();
        struct timeval lanczosTimeStop;
	gettimeofday(&lanczosTimeStop, 0);
        printf("lanczos Iterations : %f seconds\n", getTimeUs(lanczosTimeStart, lanczosTimeStop)/1e6);
        
        struct timeval eigCalcStart;
	/*cudaThreadSynchronize();*/
        gettimeofday(&eigCalcStart, 0);

        //printf("End : cycle = %d i = %d iter = %d\n", cycle, i , iter);


        //calcEigs(width, height, p_nEigNum, p_nMatrixDimension, nIterations, p_eigenValues, p_eigenVectors, aTEigVals, aaTEigVecs, d_aaDMatrixV, devRSqrtSum);
        
        /* included for calcEigs  */
        dim3 blockDim2(XBLOCK, YBLOCK);
	dim3 gridDim2((width - 1)/XBLOCK + 1, (height - 1)/(YBLOCK) + 1);
	
        scaleEigByD<<<gridDim2, blockDim2>>>(width, height, devRSqrtSum, devEigVectors, p_nEigNum);

        /* end calcEigs */

	struct timeval eigCalcStop;
	cudaThreadSynchronize();gettimeofday(&eigCalcStop, 0);
	printf("Eigenvector calculation: %f microseconds\n", getTimeUs(eigCalcStart, eigCalcStop));

        //cutSavePGMf("eigvec1.pgm", p_eigenVectors+2*p_nMatrixDimension, width,height);
	/* 	printf("\nTotal Iterations : %d", nIter); */
	/* 	printf("\nRequired Eigen Values\n"); */
	/* 	for (int i = 0; i < REQUIREDEIGS; i++) */
	/* 	{ */
	/* 		//printf("%e ", p_eigenValues[i]); */
	/*     printf("%e ", w[i]); */
	/* 	} */
	/* 	printf("\n"); */

	/*   FILE* fp; */
	/*   fp = fopen("iterationTimes.txt", "w"); */
	/*   for (int j = 0; j < i; j++) { */
	/*     fprintf(fp, "%i ", j); */
	/*     floatVector* currentIteration = times[j]; */
	/*     for(std::vector<float>::iterator it = currentIteration->begin(); it != currentIteration->end(); it++) { */
	/*       fprintf(fp, "%e ", *it); */
	/*     } */
	/*     fprintf(fp, "\n"); */
	/*   } */
	/*   fclose(fp); */

	/*   fp = fopen("ritzValues.bin", "w"); */
	/*   //for (std::vector<floatVector*>::iterator jt = ritzValues.begin(); jt != ritzValues.end(); jt++) { */
	/*   std::vector<doubleVector*>::reverse_iterator jt = ritzValues.rbegin(); */
	/*   doubleVector* currentRitz = *jt; */
	/*   for(std::vector<double>::iterator it = currentRitz->begin(); it != currentRitz->end(); it++) { */
	/*     double currentRitzValue = *it; */
	/*     fwrite(&currentRitzValue, sizeof(double), 1, fp); */
	/*     //fprintf(fp, "%e ", *it); */
	/*   } */
	/*   //fprintf(fp, "\n"); */
	/*     //} */
	/*   fclose(fp); */

	/*   fp = fopen("cullumValues.bin", "w"); */
	/*   //for (std::vector<floatVector*>::iterator jt = cullumValues.begin(); jt != cullumValues.end(); jt++) { */
	/*   jt = cullumValues.rbegin(); */
	/*   doubleVector* currentCullum = *jt; */
	/*   for(std::vector<double>::iterator it = currentCullum->begin(); it != currentCullum->end(); it++) { */
	/*     double currentCullumValue = *it; */
	/*     //fprintf(fp, "%e ", *it); */
	/*     fwrite(&currentCullumValue, sizeof(double), 1, fp); */
	/*   } */
	/*    /\*  fprintf(fp, "\n"); *\/ */
	/* /\*   } *\/ */
	/*   fclose(fp); */




	/*   fp = fopen("alpha.bin", "w"); */
	/*   fwrite(aAlpha, sizeof(float), i, fp); */
	/* /\*   for (int j = 0; j < i; j++) { *\/ */
	/* /\*     fprintf(fp, "%f\n", aAlpha[j]); *\/ */
	/* /\*   } *\/ */
	/*   fclose(fp); */
	/*   fp = fopen("beta.bin", "w"); */
	/*   fwrite(aBeta, sizeof(float), i - 1, fp); */
	/* /\*   for (int j = 0; j < i - 1; j++) { *\/ */
	/* /\*     fprintf(fp, "%f\n", aBeta[j]); *\/ */
	/* /\*   } *\/ */
	/*   fclose(fp); */

	//return;
	// Free memory used by the sstegr subroutine

	free(tempBeta);
	free(tempAlpha);
	free(w);
	//free(z);
	//free(isuppz);
	free(work);
	free(iwork);
        //free(h_aaDMatrixV);
	// End of freeing memory usage
        delete currentIteration;


	cublasFree(d_aVectorT1);
	cublasFree(d_aVectorT2);
	cublasFree(d_aaDMatrixV);
	cublasFree(d_aVectorQQPrev);
	cublasFree(d_aVectorQQ);
	cublasFree(d_aVectorZ);
	lanczosClear(aInitVector, aBeta, aAlpha, aTEigVals, aaTEigVecs);
}

__global__ void FindMaxMinPerBlock(int p_nMatrixDimension, float* p_devEigVecs, int p_nEigNum, float* p_devReduceMax, float* p_devReduceMin, int p_nMaxLevel)
{
	int index = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	__shared__ float MaxReduce[XBLOCK*(MAXEIGNUM - 1)];
	__shared__ float MinReduce[XBLOCK*(MAXEIGNUM - 1)];

	if (index < (p_nMatrixDimension + 1)/2)
	{
		//First Reduction
		for (int i = 0; i < p_nEigNum - 1; i++)
		{
			int eigVecIndex = (i+1)*p_nMatrixDimension+index*2;
			int reduceIndex = threadIdx.x + i * XBLOCK;
			//If p_nMatrixDimension is an odd number
			if ((p_nMatrixDimension % 2 == 1) && (index == (p_nMatrixDimension+1)/2 - 1))
			{
				MaxReduce[reduceIndex] = MinReduce[reduceIndex] = p_devEigVecs[eigVecIndex];
			}
			else 
			{
				if (p_devEigVecs[eigVecIndex] < p_devEigVecs[eigVecIndex+1])
				{
					MaxReduce[reduceIndex] = p_devEigVecs[eigVecIndex+1];
					MinReduce[reduceIndex] = p_devEigVecs[eigVecIndex];
				}
				else
				{
					MaxReduce[reduceIndex] = p_devEigVecs[eigVecIndex];
					MinReduce[reduceIndex] = p_devEigVecs[eigVecIndex+1];
				}
			}
		}
		__syncthreads();

		//The Reductions Thereafter
		int mask = 1; 
		for (int level= 0;level< p_nMaxLevel; level++)
		{
			if ((threadIdx.x & mask) == 0)
			{
				int index1 = threadIdx.x;
				int index2 = (1 << level) + threadIdx.x;
				if (IMUL(blockDim.x, blockIdx.x) + index2 < (p_nMatrixDimension + 1)/2)
				{
					for (int i= 0; i < p_nEigNum - 1; i++)
					{
						if (MaxReduce[i*XBLOCK + index1] < MaxReduce[i*XBLOCK + index2])
						{
							MaxReduce[i*XBLOCK + index1] = MaxReduce[i*XBLOCK + index2];
						}
						if (MinReduce[i*XBLOCK + index1] > MinReduce[i*XBLOCK + index2])
						{
							MinReduce[i*XBLOCK + index1] = MinReduce[i*XBLOCK + index2];
						}
					}
				}

			}
			mask = (mask<<1)|1;
			__syncthreads();
		}

		//Write max and min into global memory
		if (threadIdx.x == 0)
		{
			for (int i = 0; i < p_nEigNum - 1; i++)
			{
				int memIndex = i * gridDim.x + blockIdx.x;
				p_devReduceMax[memIndex] = MaxReduce[i*XBLOCK];
				p_devReduceMin[memIndex] = MinReduce[i*XBLOCK];
			}
		}
	}
}

__global__ void FindMaxMinPerGrid(int p_nGridSize, int p_nEigNum, float* p_devMax, float* p_devMin, float* p_devReduceMax, float* p_devReduceMin, int p_nMaxLevel)
{
	__shared__ float MaxReduce[XBLOCK*(MAXEIGNUM - 1)];
	__shared__ float MinReduce[XBLOCK*(MAXEIGNUM - 1)];

	int taskPerTh = (p_nGridSize + XBLOCK - 1)/XBLOCK;
	// First Assignment

	if (threadIdx.x < p_nGridSize)
	{
		for (int i = 0; i < p_nEigNum - 1; i++)
		{
			MaxReduce[i*XBLOCK + threadIdx.x] = p_devMax[threadIdx.x + i * p_nGridSize];
			MinReduce[i*XBLOCK + threadIdx.x] = p_devMin[threadIdx.x + i * p_nGridSize];
		}
	}

	// First Reduction
	for (int i = 1; i < taskPerTh; i++)
	{
		int curIndex = threadIdx.x + i * XBLOCK;
		if (curIndex < p_nGridSize)
		{
			for (int j = 0; j < p_nEigNum - 1; j++)
			{
				if (MaxReduce[j*XBLOCK + threadIdx.x] < p_devMax[curIndex + j * p_nGridSize])
				{
					MaxReduce[j*XBLOCK + threadIdx.x] = p_devMax[curIndex + j * p_nGridSize];
				}
				if (MinReduce[j*XBLOCK + threadIdx.x] > p_devMin[curIndex + j * p_nGridSize])
				{
					MinReduce[j*XBLOCK + threadIdx.x] = p_devMin[curIndex + j * p_nGridSize];
				}
			}
		}
	}
	__syncthreads();

	//The Reductions Thereafter
	int mask = 1; 
	for (int level = 0; level < p_nMaxLevel; level++)
	{
		if ((threadIdx.x & mask) == 0)
		{
			int index1 = threadIdx.x;
			int index2 = (1 << level) + threadIdx.x;
			if (index2 < p_nGridSize)
			{
				for (int i = 0; i < p_nEigNum - 1; i++)
				{
					if (MaxReduce[i*XBLOCK + index1] < MaxReduce[i*XBLOCK + index2])
					{
						MaxReduce[i*XBLOCK + index1] = MaxReduce[i*XBLOCK + index2];
					}
					if (MinReduce[i*XBLOCK + index1] > MinReduce[i*XBLOCK + index2])
					{
						MinReduce[i*XBLOCK + index1] = MinReduce[i*XBLOCK + index2];
					}
				}
			}
		}
		__syncthreads();
		mask = (mask<<1)|1;
	}

	//Write max and min into global memory
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < p_nEigNum - 1; i++)
		{
			p_devReduceMax[i] = MaxReduce[i*XBLOCK];
			p_devReduceMin[i] = MinReduce[i*XBLOCK];
		}
	}

}

__global__ void NormalizationDev(int p_nMatrixDimension, int p_nEigNum, float* p_devEigVecs, float* p_devMax, float* p_devMin)
{
	int index = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	if (index < p_nMatrixDimension)
	{
		for (int i = 0; i < p_nEigNum - 1; i++)
		{
			int curIndex = index + i*p_nMatrixDimension;
			if ((p_devMax[i]-p_devMin[i]) > 1e-4)
			{
				p_devEigVecs[curIndex] = (p_devEigVecs[curIndex] - p_devMin[i] )/(p_devMax[i]-p_devMin[i]);
			}
		}
	}
}

void PrintCudaVector(int p_nSize, float* p_devVec)
{
	float* vec = (float*) malloc(p_nSize*sizeof(float));
	CUDA_SAFE_CALL(cudaMemcpy(vec, p_devVec, p_nSize*sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < p_nSize; i++)
	{
		printf("\n %d : %f", i, vec[i]);
	}
	free(vec);
}

void FindMaxMin(int p_nStart, int p_nEnd, float* p_Vec)
{
	float min = p_Vec[p_nStart];
	float max = p_Vec[p_nStart];
	for (int i = p_nStart + 1; i < p_nEnd; i++)
	{
		if (p_Vec[i] < min)
			min = p_Vec[i];
		if (p_Vec[i] > max)
			max = p_Vec[i];
	}
	printf("\n Serial Max %f Min %f", max, min);
}

void NormalizeEigVecDev(int p_nMatrixDimension, float* p_devEig, int p_nEigNum)
{

	int blockNum = ((p_nMatrixDimension + 1) / 2 - 1)/XBLOCK + 1;
	dim3 blockDim(XBLOCK, 1);
	dim3 gridDim(blockNum, 1);
	float* devReduceMax = 0;
	float* devReduceMin = 0;
	CUDA_SAFE_CALL(cudaMalloc((void**)&devReduceMin, blockNum*sizeof(float)*(MAXEIGNUM-1)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&devReduceMax, blockNum*sizeof(float)*(MAXEIGNUM-1)));

	int maxLevel = 0;
	int temp = XBLOCK;
	while(temp !=0 )
	{
		maxLevel++;
		temp>>=1;
	}
	maxLevel--;
	FindMaxMinPerBlock<<<gridDim, blockDim>>>(p_nMatrixDimension, p_devEig, p_nEigNum, devReduceMax, devReduceMin, maxLevel);
	float* devFinalMax = 0;
	float* devFinalMin = 0;


	CUDA_SAFE_CALL(cudaMalloc((void**)&devFinalMax, MAXEIGNUM*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&devFinalMin, MAXEIGNUM*sizeof(float)));
	dim3 oneGrid(1,1);
	FindMaxMinPerGrid<<<oneGrid, blockDim>>>(blockNum, p_nEigNum, devReduceMax, devReduceMin,devFinalMax, devFinalMin, maxLevel);


	dim3 gridDim2((p_nMatrixDimension - 1) / XBLOCK + 1, 1);
	NormalizationDev<<<gridDim2, blockDim>>>(p_nMatrixDimension, p_nEigNum, p_devEig + p_nMatrixDimension, devFinalMax, devFinalMin);


	CUDA_SAFE_CALL(cudaFree(devReduceMax));
	CUDA_SAFE_CALL(cudaFree(devReduceMin));
	CUDA_SAFE_CALL(cudaFree(devFinalMin));
	CUDA_SAFE_CALL(cudaFree(devFinalMax));


}

void NormalizeEigVecs(int p_nMatrixDimension, float* p_aaEigVecs, int p_nEigNum)
{
	for (int i = 1; i < p_nEigNum; i++)
	{
		float minValue = 100;
		float maxValue = -100;
		for (int j = 0; j < p_nMatrixDimension; j++)
		{
			float temp = *(p_aaEigVecs+i*p_nMatrixDimension+j);
			if (minValue > temp)
				minValue = temp;
			if (maxValue < temp)
				maxValue = temp;
		}
		float diff = maxValue - minValue;
		for (int j = 0; j < p_nMatrixDimension; j++)
		{
			p_aaEigVecs[i*p_nMatrixDimension+j] =(p_aaEigVecs[i*p_nMatrixDimension+j] - minValue)/diff;
		}
	}
}

/* int main(int argc, char** argv)  */
/* { */
/*   chooseLargestGPU(false); */

/*   int width = 321; */
/*   int height = 481; */
/*   int radius = 5; */
/*   char* filename = "polynesia.sma"; */
/*   int nMatrixDimension = width * height; */

/*   int getNEigs = 9; */
  
/*   int nPixels = width * height; */
/*   assert(nPixels == nMatrixDimension); */
  

/*   dim3 blockDim(XBLOCK, 1); */
/*   dim3 gridDim((width * height - 1)/XBLOCK + 1, 1); */
  
/*   int matrixPitchInFloats = findPitchInFloats(nPixels); */
/*   Stencil myStencil(radius, width, height, matrixPitchInFloats); */

/*   float* devMatrix; */

/*   printf("Reading matrix from file...\n"); */
/*   float* hostMatrix = myStencil.readStencilMatrix(filename); */
/*   printf("Copying matrix to GPU\n"); */

  
/*   uint nDimension = myStencil.getStencilArea(); */
  
/*   cudaMalloc((void**)&devMatrix, nDimension * nPixels * sizeof(float)); */
 
/* 	CUDA_SAFE_CALL(cudaMemcpy(devMatrix, hostMatrix, nPixels * nDimension * sizeof(float), cudaMemcpyHostToDevice)); */
 
/*   struct timeval start; */
/*   gettimeofday(&start, 0); */
  
/*   float* devRSqrtSum = convertMatrix(&myStencil, gridDim, blockDim, nDimension, devMatrix); */

  
  
 
/*   float* eigenValues; */
/*   float* eigenVectors; */
    
/*   initEigs(getNEigs, nMatrixDimension, &eigenValues, &eigenVectors); */




/*   int nOrthoChoice = 1; */
/*   if (argc > 1) */
/*     nOrthoChoice = atoi(argv[1]); */
/*   lanczos(nMatrixDimension, gridDim, blockDim, &myStencil, devMatrix,   */
          
/*           getNEigs, eigenValues, eigenVectors, nOrthoChoice, devRSqrtSum); */
/*   struct timeval stop; */
/*   gettimeofday(&stop, 0); */
/*   float solveTime = (float)(stop.tv_sec - start.tv_sec)  + ((float)(stop.tv_usec - start.tv_usec))*1e-6f; */
  
/*   NormalizeEigVecs(nMatrixDimension, eigenVectors, getNEigs); */
/*   printf("Solve time: %f seconds\n", solveTime); */
/*   FILE* fp; */
/*   fp = fopen("eigenVectors.txt", "w"); */
/* 	//Print out the eigenvectors */
/*   for (int j = 0; j < nMatrixDimension; j++) { */
/*     for (int i = 0; i < getNEigs; i++) { */
/*       fprintf(fp, "%f ", eigenVectors[i*nMatrixDimension+j]); */
/*     } */
/*     fprintf(fp, "\n"); */
/*   } */
/*   fclose(fp); */

/*   fp = fopen("eigenValues.txt", "w"); */
/* 	for (int i = 0; i < getNEigs; i++) { */
/* 		fprintf(fp, "%e\n", eigenValues[i]); */
/* 	} */
/* 	fclose(fp); */

/*   spectralPb(eigenValues, eigenVectors, width, height, getNEigs); */
/*   clearEigs(eigenValues, eigenVectors); */

  
/* } */


void generalizedEigensolve(Stencil& myStencil, float* devMatrix, int matrixPitchInFloats, int getNEigs, float** p_eigenvalues, float** devEigVectors, float fTolerance) {
  int width = myStencil.getWidth();
  int height = myStencil.getHeight();
  int radius = myStencil.getRadius();
  int nDimension = myStencil.getStencilArea();
  //int getNEigs = 9;
  //int getNEigs = 17;
  dim3 blockDim(XBLOCK, 1);
  dim3 gridDim((width * height - 1)/XBLOCK + 1, 1);

  float* devRSqrtSum = convertMatrix(&myStencil, gridDim, blockDim, nDimension, devMatrix);
  int nMatrixDimension = width * height;
  initEigs(getNEigs, nMatrixDimension, p_eigenvalues, devEigVectors);

  lanczos(nMatrixDimension, gridDim, blockDim, &myStencil, devMatrix, getNEigs, *p_eigenvalues, *devEigVectors, 1, devRSqrtSum, fTolerance);
  
  NormalizeEigVecDev(nMatrixDimension, *devEigVectors, getNEigs);

//  float* p_eigenvectors = new float[width*height];
//  CUDA_SAFE_CALL(cudaMemcpy(p_eigenvectors, *devEigVectors+10*nMatrixDimension, width*height*sizeof(float), cudaMemcpyDeviceToHost));
//  cutSavePGMf("eigvec1.pgm", p_eigenvectors, width,height);
//  delete [] p_eigenvectors;
}
