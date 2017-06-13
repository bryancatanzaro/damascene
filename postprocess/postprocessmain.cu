#include "skeleton.h"

void skeletonTest()
{
    int width =3;
    int height = 3;
    int nPixels = width*height;
    
    float *in = new float[width*height];
    memset(in, 0, nPixels*sizeof(float));
    in[0]=in[1]=in[2]=in[4]=1;


    float *out = new float[width*height];

    int *lut = new int[512];
    lut[89]=1; lut[91]=1; lut[217]=1; lut[219]=1;

    CPU_skeletonize(width, height, in, out, lut);

    for(int y=0;y<height;y++)
    {
        for(int x=0;x<width;x++)
        {
            printf("%d ", (in[y*width+x]>0)?1:0 );
        }
        printf("\n");
    }
    for(int y=0;y<height;y++)
    {
        for(int x=0;x<width;x++)
        {
            printf("%d ", (out[y*width+x]>0)?1:0 );
        }
        printf("\n");
    }


    float* devGPb, *devGPb_thin, *devMPb;
    checkCudaErrors(cudaMalloc((void**)&devGPb,nPixels*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&devGPb_thin,nPixels*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&devMPb,nPixels*sizeof(float)));

    checkCudaErrors(cudaMemcpy(devGPb, in, nPixels*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devGPb_thin, devGPb , nPixels*sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(devMPb, devGPb , nPixels*sizeof(float), cudaMemcpyDeviceToDevice));

    PostProcess(width, height, width , devGPb, devMPb , devGPb_thin);

    checkCudaErrors(cudaMemcpy(out, devGPb_thin,  nPixels*sizeof(float), cudaMemcpyDeviceToHost));
    printf("GPU OUT\n");
    for(int y=0;y<height;y++)
    {
        for(int x=0;x<width;x++)
        {
            printf("%d ", (out[y*width+x]>0)?1:0 );
        }
        printf("\n");
    }

    delete [] in;
    delete [] out;
    delete [] lut;

    checkCudaErrors(cudaFree(devGPb));
    checkCudaErrors(cudaFree(devMPb));
    checkCudaErrors(cudaFree(devGPb_thin));
}

int main()
{
    skeletonTest();
    return 0;
}

