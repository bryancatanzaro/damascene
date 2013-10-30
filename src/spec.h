#ifndef __SPEC_H
#define __SPEC_H

#include <sys/time.h>

#define MAX_KERNEL 32
//#define MAX_LENGTH (100)
#define MAX_LENGTH (400)
#define MAX_ORIENTATIONS 16

#define FILTER_PREFIX "newdata/savgol_"
#define MAX_FILTER_LENGTH 41
//#define MAX_FILTER_LENGTH 81
#define MAX_FILTER_ORIENTATION 8

//#define M_PI 3.14159265 

//#define __NO_ATOMIC


void gpu_parabola_init(int norients, int width, int height, int border);
void gpu_parabola_cleanup();
void gpu_parabola(int norients, int width, int height, int border, float* devPixels, int filter_radius, int filter_length, float* filters, float* devResult, int cuePitchInFloats);

static inline int timestamp()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec*1000000+tv.tv_usec;
}

#endif // __SPEC_H
