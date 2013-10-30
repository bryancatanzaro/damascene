#pragma once

bool loadPPM(const char* file, unsigned char** data, 
             unsigned int *w, unsigned int *h, unsigned int *channels);

bool loadPPM4ub(const char* file, unsigned char** data, 
                unsigned int *w, unsigned int *h);
