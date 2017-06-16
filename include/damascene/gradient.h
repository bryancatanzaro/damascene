#ifndef GRADIENT_H
#define GRADIENT_H

int initializeGradients(uint width, uint height, uint border, uint maxbins, uint norients, uint nscale, uint textonChoice);

float* gradients(float* devImage, uint nbins, bool blur, float sigma, uint* radii, int textonChoice);
float* gradients(int* devImage, uint nbins, bool blur, float sigma, uint* radii, int textonChoice);


void finalizeGradients();

#endif
