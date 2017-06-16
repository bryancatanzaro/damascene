#ifndef INTERVENING
#define INTERVENING
#include "Stencil.h"


/**
 * This function computes the affinities for every pixel inside the stencil
 * pattern described by theStencil.  It creates the sparse matrix needed
 * for the spectralPb code
 * @param theStencil the stencil pattern & information about the size of the problem
 * @param devMPb a DEVICE pointer to the MPb information used to derive the matrix.  This is assumed to be in straight scanline order (no padding)
 * @param p_devMatrix a pointer to a DEVICE pointer where the matrix is stored
 * @param sigma a parameter used in the creation of this matrix.  Defaults to 0.1
 */
void intervene(Stencil& theStencil, float* devMPb, float** p_devMatrix, float sigma = 0.1f);

#endif
