#include <damascene/ppm_util.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <fstream>

//! size of PGM file header 
const unsigned int PGMHeaderSize = 0x40;

bool loadPPM(const char* file, unsigned char** data, 
             unsigned int *w, unsigned int *h, unsigned int *channels) {
    FILE *fp = NULL;
    if(NULL == (fp = fopen(file, "rb"))) 
    {
        std::cerr << "loadPPM() : Failed to open file: " << file << std::endl;
        return false;
    }

    // check header
    char header[PGMHeaderSize];
    assert(fgets(header, PGMHeaderSize, fp) != NULL);
    if (strncmp(header, "P5", 2) == 0) {
        *channels = 1;
    }
    else if (strncmp(header, "P6", 2) == 0) {
        *channels = 3;
    } else {
        std::cerr << "loadPPM() : File is not a PPM or PGM image" << std::endl;
        *channels = 0;
        return false;
    }

    // parse header, read maxval, width and height
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int maxval = 0;
    unsigned int i = 0;
    while(i < 3) {
        assert(fgets(header, PGMHeaderSize, fp) != NULL);
        if(header[0] == '#') 
            continue;

        if(i == 0) {
            i += sscanf(header, "%u %u %u", &width, &height, &maxval);
        } else if (i == 1) {
            i += sscanf(header, "%u %u", &height, &maxval);
        } else if (i == 2) {
            i += sscanf(header, "%u", &maxval);
        }
    }

    // check if given handle for the data is initialized
    if(NULL != *data) {
        if (*w != width || *h != height) {
            std::cerr << "loadPPM() : Invalid image dimensions." << std::endl;
            return false;
        }
    } else {
        *data = (unsigned char*) malloc(sizeof(unsigned char) *
                                        width * height * *channels);
        *w = width;
        *h = height;
    }

    // read and close file
    assert(
        fread(*data, sizeof(unsigned char),
              width * height * *channels, fp) ==
        sizeof(unsigned char) * width * height * *channels);
    fclose(fp);

    return true;
}

bool loadPPM4ub(const char* file, unsigned char** data, 
                unsigned int *w, unsigned int *h) {
    unsigned char *idata = 0;
    unsigned int channels;
    
    if (loadPPM(file, &idata, w, h, &channels)) {
        // copy and pad 4th component
        int size = *w * *h;
        // keep the original pointer
        unsigned char* idata_orig = idata;
        *data = (unsigned char*) malloc(sizeof(unsigned char) * size * 4);
        unsigned char *ptr = *data;
        for(int i=0; i<size; i++) {
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = 0;
        }
        free(idata_orig);
        return true;
    } else {
        free(idata);
        return false;
    }
}

struct ConverterToUByte {
    unsigned char operator()(const float& val) {
        return static_cast<unsigned char>(val * 255.0f);
    }
};

bool savePGMf(const char* file, float* data,
              unsigned int w, unsigned int h) {
    unsigned int size = w * h;
    unsigned char* idata = 
        (unsigned char*) malloc( sizeof(unsigned char) * size);
    int channels = 1;
    std::transform( data, data + size, idata, ConverterToUByte());

    assert( NULL != data);
    assert( w > 0);
    assert( h > 0);

    std::fstream fh( file, std::fstream::out | std::fstream::binary );
    if( fh.bad()) 
    {
        std::cerr << "savePPM() : Opening file failed." << std::endl;
        return false;
    }
    fh << "P5\n";
    fh << w << "\n" << h << "\n" << 0xff << std::endl;

    for( unsigned int i = 0; (i < (w*h*channels)) && fh.good(); ++i) 
    {
        fh << data[i];
    }
    fh.flush();

    if( fh.bad()) 
    {
        std::cerr << "savePPM() : Writing data failed." << std::endl;
        return false;
    } 
    fh.close();

    // cleanup
    free( idata);

    return true;
}
