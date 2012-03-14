#ifndef TEXTON
#define TEXTON


/**
 * Finds texton labels for each pixel.  This is done by convolving the image
 * with a filter bank, then performing k-means on the outputs of the filter bank.
 * Right now, it is hard coded with a filter bank of 34 filters
 * (the 17 fundamental textons at 2 different scales)
 * The number of textons is also hard coded at 64.
 * @param width the image width
 * @param height the image height
 * @param devImage a DEVICE pointer where the greyscale image is stored, one float per channel
 * @param p_devTextons a pointer to a DEVICE pointer where the texton labels will be stored, one per pixel
 */
void findTextons(int width, int height, float* devImage, int** p_devTextons, int p_nTextonChoice);

#endif
