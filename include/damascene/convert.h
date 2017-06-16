#ifndef CONVERT
#define CONVERT

/**
 * Loads a color image from a PPM file
 * The image is stored with one 32 bit uint per pixel, where bits 0-7 encode red, 8-15 encode green, and 16-23 encode blue
 * This function allocates the image on the GPU, but the caller must free the image manually.
 * @param filename the filename of the PPM image file
 * @param p_width a pointer to a uint where the width of the image will be stored
 * @param p_height a pointer to a uint where the height of the image will be stoerd
 * @param p_devRgbU a DEVICE pointer to a uint array where the image will be stored
 */
void loadPPM_rgbU(char* filename, uint* p_width, uint* p_height, uint** p_devRgbU);

/**
 * Converts a 32 bit uint per pixel image into the Lab color space, with 3 floats per pixel.
 * Each of the L, a, b channels is stored in a separate image
 * This function allocates the Lab images on the GPU, but the caller must free them manually.
 * @param width the image width
 * @param height the image height
 * @param gamma the gamma correction factor (set to 1.0 to leave unchanged)
 * @param devRgbU a DEVICE pointer to a uint array where the image is stored
 * @param p_devL a pointer to a DEVICE pointer where the L channel will be stored
 * @param p_devA a pointer to a DEVICE pointer where the a channel will be stored
 * @param p_devB a pointer to a DEVICE pointer where the b channel will be stored
 */
void rgbUtoLab3F(uint width, uint height, float gamma, uint* devRgbU, float** p_devL, float** p_devA, float** p_devB);

/**
 * Normalizes the L, a, b channels of an image to lie between [0, 1]
 * Each of the L, a, b channels is stored in a separate image
 * This function is destructive: if you need the unnormalized L, a, b, you
 * should copy them before calling it.
 * @param width the image width
 * @param height the image height
 * @param devL a DEVICE pointer where the L channel will be stored
 * @param devA a DEVICE pointer where the a channel will be stored
 * @param devB a DEVICE pointer where the b channel will be stored
 */
void normalizeLab(uint width, uint height, float* devL, float* devA, float* devB);



/**
 * Converts a 32 bit uint per pixel image into grayscale, with 1 float per pixel (ranging from 0.0-1.0)
 * Each of the L, a, b channels is stored in a separate image
 * This function allocates the Lab images on the GPU, but the caller must free them manually.
 * @param width the image width
 * @param height the image height
 * @param devRgbU a DEVICE pointer to a uint array where the image is stored
 * @param p_devGrey a pointer to a DEVICE pointer where the Grey channel will be stored
 */
void rgbUtoGreyF(uint width, uint height, uint* devRgbU, float** p_devGrey);

/**
 * Mirrors an image.  The borders of the new image are generated from the input image, just mirrored around the edges.
 * @param width the image width
 * @param height the image height
 * @param border the border width (this is symmetric for all edges)
 * @param devInput a DEVICE pointer to the unmirrored input (assumed to be width*height floats, in scanline order)
 * @param p_devOutput a pointer to a DEVICE pointer to the mirrored output (this will be (width + 2 * border) * (height + 2 * border) floats, in scanline order
 */
void mirrorImage(uint width, uint height, uint border, float* devInput, float** p_devOutput);


/**
 * Mirrors an image.  The borders of the new image are generated from the input image, just mirrored around the edges.
 * @param width the image width
 * @param height the image height
 * @param border the border width (this is symmetric for all edges)
 * @param devInput a DEVICE pointer to the unmirrored input (assumed to be width*height floats, in scanline order)
 * @param devOutput a DEVICE pointer to the mirrored output (this will be (width + 2 * border) * (height + 2 * border) floats, in scanline order
 * This version does not allocate an output pointer
 */
void mirrorImage(uint width, uint height, uint border, float* devInput, float* p_devOutput);

/**
 * Mirrors an image.  The borders of the new image are generated from the input image, just mirrored around the edges.
 * @param width the image width
 * @param height the image height
 * @param border the border width (this is symmetric for all edges)
 * @param devInput a DEVICE pointer to the unmirrored input (assumed to be width*height floats, in scanline order)
 * @param p_devOutput a pointer to a DEVICE pointer to the mirrored output (this will be (width + 2 * border) * (height + 2 * border) floats, in scanline order
 */
void mirrorImage(uint width, uint height, uint border, int* devInput, int** p_devOutput);


/**
 * Mirrors an image.  The borders of the new image are generated from the input image, just mirrored around the edges.
 * @param width the image width
 * @param height the image height
 * @param border the border width (this is symmetric for all edges)
 * @param devInput a DEVICE pointer to the unmirrored input (assumed to be width*height floats, in scanline order)
 * @param devOutput a DEVICE pointer to the mirrored output (this will be (width + 2 * border) * (height + 2 * border) floats, in scanline order
 * This version does not allocate an output pointer
 */
void mirrorImage(uint width, uint height, uint border, int* devInput, int* p_devOutput);


/**
 * Unmirrors an image.  The borders of the old image are cut off.
 * This subroutine allocate memory for the results, but the caller should free it manually.
 * @param width the image width
 * @param height the image height
 * @param border the border width (this is symmetric for all edges)
 * @param devInput a DEVICE pointer to the unmirrored input (assumed to be width*height floats, in scanline order)
 * @param p_devOutput a pointer to a DEVICE pointer to the mirrored output (this will be (width - 2 * border) * (height - 2 * border) floats, in scanline order
 */
void unMirrorImage(uint width, uint height, uint border, float* devInput, float** p_devOutput);

/**
 * Unmirrors an image.  The borders of the old image are cut off.
 * This subroutine assumes that the memory for the results has been allocated.
 * @param width the image width
 * @param height the image height
 * @param border the border width (this is symmetric for all edges)
 * @param devInput a DEVICE pointer to the unmirrored input (assumed to be width*height floats, in scanline order)
 * @param p_devOutput a pointer to a DEVICE pointer to the mirrored output (this will be (width - 2 * border) * (height - 2 * border) floats, in scanline order
 */
void unMirrorImage(uint width, uint height, uint border, float* devInput, float* p_devOutput);

/**
 * Quantizes image from float in [0,1] to int in [0, nbins - 1]
 * @param width the image width
 * @param height the image height
 * @param nbins the number of bins to quantize to
 * @param devInput a DEVICE pointer to the float input
 * @param p_devOutput a pointer to a DEVICE pointer to the quantized output
 */
void quantizeImage(uint width, uint height, uint nbins, float* devInput, int** p_devOutput);

/**
 * Quantizes image from float in [0,1] to int in [0, nbins - 1]
 * @param width the image width
 * @param height the image height
 * @param nbins the number of bins to quantize to
 * @param devInput a DEVICE pointer to the float input
 * @param p_devOutput a pointer to a DEVICE pointer to the quantized output
 */
void quantizeImage(uint width, uint height, uint nbins, float* devInput, int* devOutput);

#endif
