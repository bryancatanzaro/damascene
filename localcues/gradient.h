#ifndef GRADIENT_H
#define GRADIENT_H

int initializeGradients(unsigned long width, unsigned long height, unsigned long border, unsigned long maxbins, unsigned long norients, unsigned long nscale, unsigned long textonChoice);

float* gradients(float* devImage, unsigned long nbins, bool blur, float sigma, unsigned long* radii, int textonChoice);
float* gradients(int* devImage, unsigned long nbins, bool blur, float sigma, unsigned long* radii, int textonChoice);


void finalizeGradients();

#endif
