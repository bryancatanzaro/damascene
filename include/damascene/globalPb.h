#ifndef GLOBALPB
#define GLOBALPB

#include <cuda.h>

#define IMUL(a, b) __mul24(a, b)


/**
 * Given all the local cues and the spectral pb, calculate the weighted sum among them and normalize the result.
 * Assuming all the arrays are in GPU, each with size p_nMatrixPitch*p_nOrient
 * @param p_nPixels: Number of pixels in the image
 * @param p_nMatrixPitch: The pitch of each matrix
 * @param p_nOrient: Number of orientations 
 * @param devCombinedGradient: The combined gradient matrix 
 * @param devspb: The spb matrix
 * @param devmpb: The mpb matrix
 * @param devGpball : The final Gpb output, with size p_nPixels * p_nOrient
 * @param devResult: The result matrix, with size p_nPixels * 1 
 * 
 **/

void StartCalcGPb(int p_nPixels, int p_nMatrixPitch, int p_nOrient,
                  float* devCombinedGradient, float* devspb, float* devmpb,
                  float* devGpball, float* devResult);

#endif

