#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <damascene/Stencil.h>

void readCircleStencilMatrix(char* filename, Stencil theStencil, float** p_array) { 
  int nDimension = theStencil.getStencilArea();
  int width = theStencil.getWidth();
  int height = theStencil.getHeight();
  
  FILE* fp = fopen(filename,"r");

  if (fp == NULL) {
    printf("Error: File not found\n");
    return;
  }
  float* array = (float*)malloc(nDimension * theStencil.getMatrixPitch());
  *p_array = array;

  int n = 0;
  size_t b = fread(&n,sizeof(int),1,fp);

  assert(n == (width * height)); 

  int nnz = 0;
  b = fread(&nnz,sizeof(int),1,fp);
  assert(nnz < nDimension * width * height);

  int* x = new int[nDimension]; //col indices;
  double* z = new double[nDimension]; //values
                         
  int nz = 0;
  for (int row = 0; row < n; row++) {
    b = fread(&nz,sizeof(int),1,fp); //number of entries in this row
    assert(nz < nDimension);
    
    b = fread(z,sizeof(double),nz,fp); //value
    b = fread(x,sizeof(int),nz,fp);    //col index
    
    for (int col = 0; col < nz; col++) {
      int index = theStencil.matrixIndex(row, x[col]);
      array[index] = (float)z[col];
    } 
  }
  assert(b != 0);
  fclose(fp);
  
  delete[] x;
  delete[] z;
}
  





