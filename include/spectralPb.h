#ifndef SPECTRALPB
#define SPECTRALPB


/**
 * Given n eigenvectors and eigenvalues, apply 8 filters to each eigenvector, and use the corresponding eigenvalues
 * as the weight of each eigenvector to sum up all the filtered eigenvector elements.
 * 
 * The caller needs to allocate and free the memory storing the results manually.
 * @param eigenvalues: The eigenvalues stored in CPU.
 * @param eigenvectors: The eigenvectors stored in CPU.
 * @param xdim: The width of the image.
 * @param ydim: The height of the image.
 * @param nvec: The number of eigenvalue and eigenvector pairs stored in CPU.
 * @param results: The final spb value for the 8 orientations. The size should be res_pitch*8.
 * @param res_pitch: The pitch of the results array. This value should be divided by sizeof(float) before
 *		     given to this subroutine.
 */
void spectralPb(float *eigenvalues, float *eigenvectors, int xdim, int ydim, int nvec, float* results, int res_pitch);


#endif
