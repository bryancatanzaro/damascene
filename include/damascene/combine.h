#ifndef COMBINE
#define COMBINE

/**
 * This function combines Bg, Cga, Cgb and Tg to produce Mpb
 * @param width the width of the image
 * @param height the height of the image
 * @param cuePitchInFloats the pitch of each row of Bg, Cga, etc.
 * @param devBg a DEVICE pointer to the Bg information.  This assumes there are three scales present, and eight orientations
 * @param devCga a DEVICE pointer to the Cga information.
 * @param devCgb a DEVICE pointer to the Cgb information.
 * @param devTg a DEVICE pointer to the Tg information.
 * @param p_devMpb a pointer to a DEVICE pointer where the Mpb information will be stored.  This routine allocates the buffer for the Mpb information.
 */
void combine(int width, int height, int cuePitchInFloats, float* devBg, float* devCga, float* devCgb, float* devTg, float** p_devMpb, float** p_devCombinedg, int p_nTextonChoice);

#endif
