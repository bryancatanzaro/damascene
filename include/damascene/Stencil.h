#ifndef STENCIL
#define STENCIL
#include <map>
#include <assert.h>

#define CONSTSPACE 16384
#define STENCILAREAMAX CONSTSPACE


typedef unsigned int uint;

class Stencil {
 public:
  Stencil(int radiusIn, int widthIn, int heightIn, int matrixPitchInFloatsIn);
  uint getStencilArea();
  int matrixIndex(int row, int col);
  void copyOffsets(int* destOffsets);
  int getMatrixPitch();
  int getMatrixPitchInFloats();
  int getHeight();
  int getWidth();
  int getRadius();
  int getDiameter();
  float* readStencilMatrix(char* filename);
  void copyDiagonalOffsets(int* destOffsets);
  
 private:
  uint stencilArea;
  int hostOffsets[STENCILAREAMAX];
  int diagonalOffsets[CONSTSPACE];
  int radius;
  int diameter;
  int width;
  int height;
  int matrixPitchInFloats;
};
#endif
