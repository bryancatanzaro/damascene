#ifndef NONMAX
#define NONMAX


/**
 *  Calculate the max suppression of the mpb.
 *  This function assumes that the p_devPB is in GPU, and the caller has already allocated memory for devNMax.
 *  @param p_nWidth: The width of the image.
 *  @param p_nHeight: The height of the image.
 *  @param p_devPB: The mpb_all matrix, the size is p_nDevPBPitch*8
 *  @param p_nDevPBPitch: The pitch of the mpb_all matrix, it should be divided by sizeof(float) before
 *			  putting into this function.
 *  @param devNMax: The matrix after max suppression. The size is p_nWidth*p_nHeight.
 */

void nonMaxSuppression(int p_nWidth, int p_nHeight, float* p_devPB, int p_nDevPBPitch, float* devNMax);

#endif

