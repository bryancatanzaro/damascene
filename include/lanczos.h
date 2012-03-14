#ifndef LANCZOS
#define LANCZOS
#include "Stencil.h"

/**
 * Performs the generalized eigensolve (D-W) v=\lambda D v, for a matrix
 * W, where D is constructed by summing the rows of W.
 * The matrix W is a sparse, multidiagonal matrix.  The diagonal structure
 * is encoded in the myStencil object.
 * @param myStencil the Stencil pattern for the matrix
 * @param devMatrix a DEVICE pointer to the matrix, stored in the correct diagonal format
 * @param matrixPitchInFloats the pitch of each row of the diagonal matrix
 * @param nEigNum number of eigenvalues required
 * @param p_eigenvalues a pointer to an array storing the eigenvalues (on the CPU)
 * @param p_eigenvectors a pointer to the eigenvectors (on the CPU)
 */
void generalizedEigensolve(Stencil& myStencil, float* devMatrix, int matrixPitchInFloats, int nEigNum, float** p_eigenvalues, float** p_eigenvectors, float p_fTolerance);


#endif
