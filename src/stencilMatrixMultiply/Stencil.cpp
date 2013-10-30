#include "Stencil.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

Stencil::Stencil(int radiusIn, int widthIn, int heightIn, int matrixPitchInFloatsIn) : radius(radiusIn), width(widthIn), height(heightIn), matrixPitchInFloats(matrixPitchInFloatsIn) {
  int r2 = radius * radius;
  int previousOffset = 0;
  int nDimension = 0;
  diameter = radius * 2 + 1;
  for (int row = -radius; row <= radius; row++) {
    for (int col = -radius; col <= radius; col++) {
      if (row * row + col * col <= r2) {
        
        int newOffset = row * width + col;
        hostOffsets[nDimension] = newOffset - previousOffset;
       
        previousOffset = newOffset;
        int diagonalOffset = (row + radius) * diameter + col + radius;
        diagonalOffsets[diagonalOffset] = nDimension;
        //printf("DiagonalOffsets: %i = %i\n", diagonalOffset, nDimension);
        
        nDimension++;
      }
    }
  }
  stencilArea = nDimension;
  assert(stencilArea <= STENCILAREAMAX);
  for(int i = nDimension; i < STENCILAREAMAX; i++) {
    hostOffsets[i] = -2000000000;
  }
  // for(int i = 0; i < nDimension; i++) {
//     printf("nDimension: %i, (xOffset, yOffset): (%i, %i)\n", i, hostXOffsets[i], hostYOffsets[i]);
//   }
}

uint Stencil::getStencilArea() {
  return stencilArea;
}

int Stencil::matrixIndex(int row, int col) {
  int differential = col - row;
  int xOffset = differential % width;
  int yOffset = differential / width;
  if (xOffset > radius) {
    xOffset = xOffset - width;
    yOffset++;
  }
  if (xOffset < -radius) {
    xOffset = xOffset + width;
    yOffset--;
  }
 
  int x = row % width;
  int y = row / width;
  int dimension = diagonalOffsets[(yOffset + radius) * diameter + xOffset + radius];
  int index = matrixPitchInFloats * dimension + width * y + x;
  return index;
}

void Stencil::copyOffsets(int* destOffsets) {
  memcpy(destOffsets, hostOffsets, sizeof(int) * STENCILAREAMAX);
}

void Stencil::copyDiagonalOffsets(int* destOffsets) {
  memcpy(destOffsets, diagonalOffsets, sizeof(int) * diameter * diameter);
}

int Stencil::getMatrixPitchInFloats() {
  return matrixPitchInFloats;
}

int Stencil::getMatrixPitch() {
  return matrixPitchInFloats * sizeof(float);
}

int Stencil::getHeight() {
  return height;
}

int Stencil::getWidth() {
  return width;
}
int Stencil::getRadius() {
  return radius;
}

int Stencil::getDiameter() {
  return diameter;
}

float* Stencil::readStencilMatrix(char* filename) { 
 
  
  FILE* fp = fopen(filename,"r");

  assert(fp != NULL);
  float* array = (float*)malloc(stencilArea * matrixPitchInFloats * sizeof(float));
  for(int i = 0; i < stencilArea * matrixPitchInFloats; i++) {
    array[i] = 0.0f;
  }
  int n = 0;
  fread(&n,sizeof(int),1,fp);

  assert(n == (width * height)); 

  int nnz = 0;
  fread(&nnz,sizeof(int),1,fp);
  assert(nnz < stencilArea * width * height);

  int* x = new int[stencilArea]; //col indices;
  double* z = new double[stencilArea]; //values
                         
  int nz = 0;
  for (int row = 0; row < n; row++) {
    fread(&nz,sizeof(int),1,fp); //number of entries in this row
    assert(nz <= stencilArea);
    
    fread(z,sizeof(double),nz,fp); //value
    fread(x,sizeof(int),nz,fp);    //col index
    
    for (int col = 0; col < nz; col++) {
  //     if ((row == 43) && (x[col] == 55)) {
//         printf("Element 43, 55 = %f\n", z[col]);
//       }
      int index = matrixIndex(row, x[col]);
      array[index] = (float)z[col];
    } 
  }
  fclose(fp);
  
  delete[] x;
  delete[] z;

  // for(int i = 0; i < stencilArea; i++) {
//     for(int j = 0; j < height; j++) {
//       for(int k = 0; k < width; k++) {
//         printf("%.2f ", array[i * matrixPitchInFloats + j * widthPitchInFloats + k]);
//       }
//     }
//     printf("\n");
//   }
  
  return array;
}
