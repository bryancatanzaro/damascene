#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cutil.h>
#include <assert.h>
#include "Stencil.h"

#define XBLOCK 16
#define YBLOCK 16



//#define MAXPB 1

/* __device__ int max(int a, int b) { */
/*   return (a > b) ? a : b; */
/* } */

/* __device__ int min(int a, int b) { */
/*   return (a < b) ? a : b; */
/* } */

__constant__ int constDiagonals[CONSTSPACE];
texture<float, 2, cudaReadModeElementType> mPb;



__device__ float affinity(float cicss, float rsigma) {
  return exp (-(1-cicss)*rsigma);
}

__device__ float C_IC_SS(float ic) {
  float val = 1.8682;
  
  val += (ic / 0.3130)*-1.3113;
  
  const float post = 1.0 / (1.0 + exp(-val));
  if (!isfinite(post)) { return 0; }
  if (post < 0) { return 0; }
  if (post > 1) { return 1; }
  return post;
}

__device__ void
ic_walk (const int x0, const int y0,
         const int x1, const int y1,
         const int x2, const int y2,
         const int x3, const int y3,
         const int radius, const int diameter,
         const int width, const int height,
         const float rsigma,
         float* devMatrix, int matrixPitchInFloats/*, float* scratch*/)
{
 
  //constants used in testing whether this is
  //the best path
  const int dx1 = x1 - x0;
  const int dy1 = y1 - y0;
  const int dx2 = x2 - x0;
  const int dy2 = y2 - y0;
  const int dx3 = x3 - x0;
  const int dy3 = y3 - y0;
  const int dot11 = dx1 * dx1 + dy1 * dy1;
  const int dot22 = dx2 * dx2 + dy2 * dy2;
  const int dot33 = dx3 * dx3 + dy3 * dy3;

  // compute dx,dy for the bresenham line
  const int dx = x2 - x0;
  const int dy = y2 - y0;
  const int adx = abs (dx);
  const int ady = abs (dy);

  // figure out what octant we're in for the bresenham algorithm;
  // octant i covers pi/4 * [i,i+1)
  int octant = -1;
  if (dx > 0 && dy >= 0) {           // quadrant 0
    octant = (adx > ady) ? 0 : 1;
  } else if (dx <= 0 && dy > 0) {    // quadrant 1
    octant = (adx < ady) ? 2 : 3;
  } else if (dy <= 0 && dx < 0) {    // quadrant 2
    octant = (adx > ady) ? 4 : 5;
  } else if (dx >= 0 && dy < 0) {    // quadrant 3
    octant = (adx < ady) ? 6 : 7;
  } 

  // t is our bresenham counter
  int t = 0;
  switch (octant)
    {
    case 0: t = -adx; break;
    case 1: t = -ady; break;
    case 2: t = -ady; break;
    case 3: t = -adx; break;
    case 4: t = -adx; break;
    case 5: t = -ady; break;
    case 6: t = -ady; break;
    case 7: t = -adx; break;
    
    }

  // maxpb contains the max-accumulation of pb from (x0,y0) to (x,y)
  // on the bresenham line.
  float maxpb = 0.0f;

  // (xi,yi) is our current location on the bresenham line
  int xi = x0;
  int yi = y0;

  // accumulate the points in the order we find them
  int oldx = xi;
  int oldy = yi;

  //walk the line
  while (xi != x2 || yi != y2)
    {
      // step one pixel on the bresenham line
      switch (octant)
        {
        case 0:
          xi++; t += (ady << 1);
          if (t > 0) { yi++; t -= (adx << 1); }
          break;
        case 1:
          yi++; t += (adx << 1);
          if (t > 0) { xi++; t -= (ady << 1); }
          break;
        case 2:
          yi++; t += (adx << 1);
          if (t > 0) { xi--; t -= (ady << 1); }
          break;
        case 3:
          xi--; t += (ady << 1);
          if (t > 0) { yi++; t -= (adx << 1); }
          break;
        case 4:
          xi--; t += (ady << 1);
          if (t > 0) { yi--; t -= (adx << 1); }
          break;
        case 5:
          yi--; t += (adx << 1);
          if (t > 0) { xi--; t -= (ady << 1); }
          break;
        case 6:
          yi--; t += (adx << 1);
          if (t > 0) { xi++; t -= (ady << 1); }
          break;
        case 7:
          xi++; t += (ady << 1);
          if (t > 0) { yi--; t -= (adx << 1); }
          break;
        }

      // Figure out if the bresenham line from (x0,y0) to (x2,y2) is the
      // best approximant we will see for the line from (x0,y0) to (xi,yi).
      // We need:
      //              T(i,2) < T(i,1) && T(i,2) <= T(i,3)
      // Where T(a,b) is the angle between the two lines (x0,y0)-(xa,ya)
      // and (x0,y0)-(xb,yb).
      // We can compute an exact integer predicate; let C be the square
      // of the cosine of T:
      //              C(i,2) > C(i,1) && C(i,2) >= C(i,3)
      // Use the identity:
      //              cos(t) = a.b/|a||b|
      // Square and cross-multiply to get rid of the divides and square
      // roots.
      // Note that we first check to see if T(i,2) == 0, in which case
      // the line is a perfect approximant.

      const int dxi = xi - x0;
      const int dyi = yi - y0;
      const int dotii = dxi * dxi + dyi * dyi;
      const int doti1 = dxi * dx1 + dyi * dy1;
      const int doti2 = dxi * dx2 + dyi * dy2;
      const int doti3 = dxi * dx3 + dyi * dy3;


      const bool good = (doti2*doti2 == dotii*dot22)
        || (dot11*doti2*doti2 > dot22*doti1*doti1
            && dot33*doti2*doti2 >= dot22*doti3*doti3);


      // otherwise accumulate the pb value if we've crossed an edge
      float intersected = 0.0f;
    /*   intersected = tex2D(mPb, xi, yi); */

/*       if ((oldx != xi) && (oldy != yi)) { */
/*         intersected = fmax(tex2D(mPb, oldx, yi), intersected); */
/*         intersected = fmax(tex2D(mPb, xi, oldy), intersected); */
/*       } */
    
      if (oldx == xi)
        {
          if (yi > oldy)
            {
              intersected = tex2D(mPb, xi, yi-1);//boundaries.H(xi,yi);
            }
          else if (yi < oldy)
            {
              intersected = tex2D(mPb, oldx,oldy-1);//boundaries.H(oldx,oldy);
            }
        }
      else if (oldy == yi)
        {
          if (xi > oldx)
            {
              intersected = tex2D(mPb, xi-1, yi);//boundaries.V(xi,yi);
            }
          else if (xi < oldx)
            {
              intersected = tex2D(mPb, oldx-1, oldy);//boundaries.V(oldx,oldy);
            }
        }
      else
        {
          if ((xi > oldx) && (yi > oldy))  //down to right
            {
              intersected = fmax(tex2D(mPb, oldx, yi-1), intersected);//boundaries.H(oldx,yi),intersected);
              intersected = fmax(tex2D(mPb, xi, yi-1), intersected);//boundaries.H(xi,yi),intersected);
              intersected = fmax(tex2D(mPb, xi-1, oldy), intersected);//boundaries.V(xi,oldy),intersected);
              intersected = fmax(tex2D(mPb, xi-1, yi), intersected);//boundaries.V(xi,yi),intersected);
            }
          else if ((xi > oldx) && (yi < oldy)) //up to right
            {
              intersected = fmax(tex2D(mPb, oldx, oldy-1), intersected);//boundaries.H(oldx,oldy),intersected);
              intersected = fmax(tex2D(mPb, xi, oldy-1), intersected);//boundaries.H(xi,oldy),intersected);
              intersected = fmax(tex2D(mPb, xi-1, oldy), intersected);//boundaries.V(xi,oldy),intersected);
              intersected = fmax(tex2D(mPb, xi-1, yi), intersected);//boundaries.V(xi,yi),intersected);
              /* if ((x0 == 49) && (y0 == 19) && (x2 == 54) && (y2 == 16) */
/*                   && (xi == 50) && (yi == 18)) { */
/*                 scratch[0] = tex2D(mPb, oldx, oldy-1); */
/*                 scratch[1] = tex2D(mPb, xi, oldy-1); */
/*                 scratch[2] = tex2D(mPb, xi-1, oldy); */
/*                 scratch[3] = tex2D(mPb, xi-1, yi); */
/*                 scratch[4] = 10.0f; */
/*               } */
            }
          else if ((xi < oldx) && (yi > oldy)) //down to left
            {
              intersected = fmax(tex2D(mPb, oldx, yi-1), intersected);//boundaries.H(oldx,yi),intersected);
              intersected = fmax(tex2D(mPb, xi, yi-1), intersected);//boundaries.H(xi,yi),intersected);
              intersected = fmax(tex2D(mPb, oldx-1, oldy), intersected);//boundaries.V(oldx,oldy),intersected);
              intersected = fmax(tex2D(mPb, oldx-1, yi), intersected);//boundaries.V(oldx,yi),intersected);
            }
          else if ((xi < oldx) && (yi < oldy)) //up to left
            {
              intersected = fmax(tex2D(mPb, oldx, oldy-1), intersected);//boundaries.H(oldx,oldy),intersected);
              intersected = fmax(tex2D(mPb, xi, oldy-1), intersected);//boundaries.H(xi,oldy),intersected);
              intersected = fmax(tex2D(mPb, oldx-1, oldy), intersected);//boundaries.V(oldx,oldy),intersected);
              intersected = fmax(tex2D(mPb, oldx-1, yi), intersected);//boundaries.V(oldx,yi),intersected);
            }
        }
      maxpb = fmax(maxpb,intersected);
      oldx = xi;
      oldy = yi;
/*       if ((x0 == 49) && (y0 == 19) && (x2 == 54) && (y2 == 16)) { */
/*         int entry = xi - x0 - 1; */
/*         scratch[3*entry] = xi; */
/*         scratch[3*entry+1] = yi; */
/*         scratch[3*entry+2] = maxpb; */
/*       } */
      // if the approximation is not good, then skip this point
      if (!good) { continue; }

      float val = affinity(C_IC_SS(maxpb), rsigma);
      //float val = maxpb;
      //float val = tex2D(mPb, xi, yi);
      int xOffset = xi - x0;
      int yOffset = yi - y0;
      if((xOffset * xOffset + yOffset * yOffset <= radius * radius)) {
        int dimension = constDiagonals[(yOffset + radius) * diameter + xOffset + radius];
        int index = matrixPitchInFloats * dimension + y0 * width + x0;
        #ifdef MAXPB
        devMatrix[index] = maxpb;
        #else
        devMatrix[index] = val;
        #endif

        
      }
    }


}

__device__ void fixEntry(int width, int height, int radius, int diameter, int x0, int y0, int x, int y, float rsigma, float* devMatrix, int matrixPitchInFloats) {
  int xOffset = x - x0;
  int yOffset = y - y0;
  int dimension = constDiagonals[(yOffset + radius) * diameter + xOffset + radius];
  int index = matrixPitchInFloats * dimension + y0 * width + x0;
  float maxPb = 0.0f;
  #ifdef MAXPB
  devMatrix[index] = maxPb;
  #else
  devMatrix[index] = affinity(C_IC_SS(maxPb), rsigma);
  #endif
}

__global__ void findAffinities(int width, int height, int radius, int diameter, float rsigma, float* devMatrix, int matrixPitchInFloats/*, float* scratch*/) {
  
  int x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int y0 = blockDim.y * blockIdx.y + threadIdx.y;

  
  if ((x0 < width) && (y0 < height)) {
    //Self affinity = 1.0
      int dimension = constDiagonals[radius * diameter + radius];
      int index = matrixPitchInFloats * dimension + y0 * width + x0;
      devMatrix[index] = 1.0;

     // the rectangle of interest, a square with edge of length
      // 2*radius+1 clipped to the image dimensions
      const int rxa = max(0,x0-radius);
      const int rya = max(0,y0-radius);
      const int rxb = min(x0+radius,width-1);
      const int ryb = min(y0+radius,height-1);



      
      // walk around the boundary, collecting points in the scanline array
      // first walk around the rectangle boundary clockwise for theta = [pi,0]
      //std::cerr << "[" << x0 << "," << y0 << "]  ";
      //std::cerr << "(" << rxa << "," << rya << ")-(" << rxb << "," << ryb << ")" << std::endl;
      if (x0 > rxa) // left 
      {
        if ((y0 > rya) && (y0 < ryb))
        {
          ic_walk(x0,y0, rxa,y0-1, rxa,y0, rxa,y0+1, radius, diameter, width, height, rsigma, devMatrix, matrixPitchInFloats/*, scratch*/);
        }
        for (int y = y0-1; y > rya; y--)
        {
          ic_walk(x0,y0, rxa,y-1, rxa,y, rxa,y+1, radius, diameter, width, height, rsigma, devMatrix, matrixPitchInFloats/*, scratch*/);
        }
      }
      
      if (x0 > rxa+1 || y0 > rya+1 || ((x0 > rxa) && (y0 > rya)) ) // top-left
      {
        ic_walk(x0,y0, rxa,rya+1, rxa,rya, rxa+1,rya, radius, diameter, width, height, rsigma, devMatrix, matrixPitchInFloats/*, scratch*/);
      }
     /*  if((x0 == rxa) && (y0 == rya + 1)) { */
/*         //fixEntry(width, height, radius, diameter, x0, y0+1, rxa, rya+1, rsigma, devMatrix, matrixPitchInFloats, scratch); */
/*       } */
/*       if ((x0 == rxa+1) && (y0 == rya))  */
/*       { */
/*         //fixEntry(width, height, radius, diameter, x0+1, y0, rxa+1, rya, rsigma, devMatrix, matrixPitchInFloats, scratch); */
/* /\*         PointIC pnt; *\/ */
/* /\*         pnt.x = rxa; *\/ */
/* /\*         pnt.y = rya; *\/ */
/* /\*         pnt.sim = 1.0f; *\/ */
/* /\*         const int yind = pnt.y - y0 + radius; *\/ */
/* /\*         scanLines(yind,scanCount(yind)++) = pnt; *\/ */
/*       } */

      if (y0 > rya) // top
      {
        for (int x = rxa+1; x < rxb; x++)
        {
          ic_walk(x0,y0, x-1,rya, x,rya, x+1,rya, radius, diameter, width, height, rsigma, devMatrix, matrixPitchInFloats/*, scratch*/);
        }
      }

      if ((y0 > rya+1) || (x0 < rxb-1) || ((y0 > rya) && (x0 < rxb)) ) // top-right
      {
        ic_walk(x0,y0, rxb-1,rya, rxb,rya, rxb,rya+1, radius, diameter, width, height, rsigma, devMatrix, matrixPitchInFloats/*, scratch*/);
      }
     /*  if ((x0 == rxb-1) && (y0 == rya)) { */
/*         //fixEntry(width, height, radius, diameter, x0, y0, rxb, rya, rsigma, devMatrix, matrixPitchInFloats, scratch); */
/*       } */
/*       if ((x0 == rxb) && (y0 == rya+1))  { */
/*         //fixEntry(width, height, radius, diameter, x0, y0+1, rxb, rya+1, rsigma, devMatrix, matrixPitchInFloats, scratch); */
/*       } */
/* /\*       { *\/ */
/* /\*         PointIC pnt; *\/ */
/* /\*         pnt.x = rxb; *\/ */
/* /\*         pnt.y = rya; *\/ */
/* /\*         pnt.sim = 1.0f; *\/ */
/* /\*         const int yind = pnt.y - y0 + radius; *\/ */
/* /\*         scanLines(yind,scanCount(yind)++) = pnt; *\/ */
/* /\*       } *\/ */


      if (x0 < rxb) // right
      {
        for (int y = rya+1; y < y0; y++)
        {
          ic_walk(x0,y0, rxb,y-1, rxb,y, rxb,y+1, radius, diameter, width, height, rsigma, devMatrix, matrixPitchInFloats/*, scratch*/);
        }
      }

      // now counterclockwise for theta = (pi,0)
      if (x0 > rxa) // left
      {
        for (int y = y0+1; y < ryb; y++)
        {
          ic_walk(x0,y0, rxa,y-1, rxa,y, rxa,y+1, radius, diameter, width, height, rsigma, devMatrix, matrixPitchInFloats/*, scratch*/);
        }
      }

      if ((x0 > rxa+1) || (y0 < ryb-1) || ((x0 > rxa) && (y0 < ryb))) // bottom-left
      {
        ic_walk(x0,y0, rxa,ryb-1, rxa,ryb, rxa+1,ryb, radius, diameter, width, height, rsigma, devMatrix, matrixPitchInFloats/*, scratch*/);
      }
     /*  if ((x0 == rxa) && (y0 == ryb-1)) { */
/*         //fixEntry(width, height, radius, diameter, x0+1, y0, rxa+1, ryb, rsigma, devMatrix, matrixPitchInFloats, scratch); */
/*       } */
/*       if ((x0 == rxa+1) && (y0 == ryb))  { */
/*         //fixEntry(width, height, radius, diameter, x0, y0, rxa, ryb, rsigma, devMatrix, matrixPitchInFloats, scratch); */
/*       } */

/* /\*       { *\/ */
/* /\*         PointIC pnt; *\/ */
/* /\*         pnt.x = rxa; *\/ */
/* /\*         pnt.y = ryb; *\/ */
/* /\*         pnt.sim = 1.0f; *\/ */
/* /\*         const int yind = pnt.y - y0 + radius; *\/ */
/* /\*         scanLines(yind,scanCount(yind)++) = pnt; *\/ */
/* /\*       } *\/ */
      if (y0 < ryb) // bottom
      {
        for (int x = rxa+1; x < rxb; x++)
        {
          ic_walk(x0,y0, x-1,ryb, x,ryb, x+1,ryb, radius, diameter, width, height, rsigma, devMatrix, matrixPitchInFloats/*, scratch*/);
        }
      }

      if ((y0 < ryb-1) || (x0 < rxb-1) || ((y0 < ryb) && (x0 < rxb))) // bottom-right
      {
        ic_walk(x0,y0, rxb-1,ryb, rxb,ryb, rxb,ryb-1, radius, diameter, width, height, rsigma, devMatrix, matrixPitchInFloats/*, scratch*/);
      }
    /*   if ((x0 == rxb-1) && (y0 == ryb)) { */
/*         //fixEntry(width, height, radius, diameter, x0, y0, rxb, ryb, rsigma, devMatrix, matrixPitchInFloats, scratch); */
/*       } */
/*       if ((x0 == rxb) && (y0 == ryb-1)) { */
/*         //fixEntry(width, height, radius, diameter, x0, y0, rxb, ryb, rsigma, devMatrix, matrixPitchInFloats, scratch); */
/*       } */
      
/* /\*       { *\/ */
/* /\*         PointIC pnt; *\/ */
/* /\*         pnt.x = rxb; *\/ */
/* /\*         pnt.y = ryb; *\/ */
/* /\*         pnt.sim = 1.0f; *\/ */
/* /\*         const int yind = pnt.y - y0 + radius; *\/ */
/* /\*         scanLines(yind,scanCount(yind)++) = pnt; *\/ */
/* /\*       } *\/ */

      if (x0 < rxb) // right
      {
        for (int y = ryb-1; y > y0; y--)
        {
          ic_walk(x0,y0, rxb,y-1, rxb,y, rxb,y+1, radius, diameter, width, height, rsigma, devMatrix, matrixPitchInFloats/*, scratch*/);
        }
        if ((y0 > 0) && (y0 < ryb))
        {
          ic_walk(x0,y0, rxb,y0-1, rxb,y0, rxb,y0+1, radius, diameter, width, height, rsigma, devMatrix, matrixPitchInFloats/*, scratch*/);
        }
      }


      //Fix some corners
      if ((y0 == rya + 1) && ((x0 == rxa) || (x0 == rxb))) {
        fixEntry(width, height, radius, diameter, x0, y0, x0, y0-1, rsigma, devMatrix, matrixPitchInFloats);
      }
      if ((x0 == rxa + 1) && ((y0 == rya) || (y0 == ryb))) {
        fixEntry(width, height, radius, diameter, x0, y0, x0-1, y0, rsigma, devMatrix, matrixPitchInFloats);
      }
      if ((x0 == rxb - 1) && ((y0 == rya) || (y0 == ryb))) {
        fixEntry(width, height, radius, diameter, x0, y0, x0+1, y0, rsigma, devMatrix, matrixPitchInFloats);
      }
      if ((y0 == ryb - 1) && ((x0 == rxa) || (x0 == rxb))) {
        fixEntry(width, height, radius, diameter, x0, y0, x0, y0+1, rsigma, devMatrix, matrixPitchInFloats);
      }
      

  }
}

__global__ void symmetrizeMatrix(int width, int height, int radius, int diameter, int nDimension, float* devMatrix, int matrixPitchInFloats/*, float* devScratch*/) {
  int x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int y0 = blockDim.y * blockIdx.y + threadIdx.y;

 
  if ((x0 < width) && (y0 < height)) {
    for(int yOffset = 0; yOffset <= radius; yOffset++) {
      for(int xOffset = -radius; xOffset <= radius; xOffset++) {
        if ((xOffset * xOffset + yOffset * yOffset <= radius * radius)) {
          if ((yOffset > 0) || (xOffset > 0)) {
            int xx = x0 + xOffset;
            int yy = y0 + yOffset;
            if ((xx >= 0) && (yy >= 0) && (xx < width) && (yy < height)) {
              int dimensionOne = constDiagonals[(yOffset + radius) * diameter + xOffset + radius];
              float* pOne = &devMatrix[dimensionOne * matrixPitchInFloats + y0 * width + x0];
              float one = *pOne;
              int dimensionTwo = constDiagonals[(radius - yOffset) * diameter + radius - xOffset];
              
              float* pTwo = &devMatrix[dimensionTwo * matrixPitchInFloats + yy * width + xx];
            
              
              float two = *pTwo;
              float symmetrized = 0.5*(one + two);
              //float symmetrized = xx;
              //float symmetrized = 10.0f;
              //float symmetrized = dimensionOne * matrixPitchInFloats + y0 * width + x0;
              *pOne = symmetrized;
              *pTwo = symmetrized;
            }
          }
        }
      }
    }
  }
}

__global__ void shapeMatrix(int width, int height, int radius, int diameter, int nDimension, float* devMatrix, int matrixPitchInFloats) {
  int x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int y0 = blockDim.y * blockIdx.y + threadIdx.y;
  if ((x0 < width) && (y0 < height)) {
    for(int yOffset = -radius; yOffset <= radius; yOffset++) {
      for(int xOffset = -radius; xOffset <= radius; xOffset++) {
        int xx = x0 + xOffset;
        int yy = y0 + yOffset;
        if ((xx < 0) || (xx >= width) || (yy < 0) || (yy >= height)) {
          int dimension = constDiagonals[(yOffset + radius) * diameter + xOffset + radius];
          devMatrix[dimension * matrixPitchInFloats + y0 * width + x0] = 0.0f;
        }
      }
    }
  }

}


void intervene(Stencil& theStencil, float* devMPb, float** p_devMatrix, float sigma) {
  int width = theStencil.getWidth();
  int height = theStencil.getHeight();
  int radius = theStencil.getRadius();
  int diameter = theStencil.getDiameter();
  int matrixPitchInFloats = theStencil.getMatrixPitchInFloats();
  int nDimension = theStencil.getStencilArea();
  int nPixels = width * height;

/*   mPb.addressMode[0] = cudaAddressModeClamp; */
/*   mPb.addressMode[1] = cudaAddressModeClamp; */
/*   mPb.filterMode = cudaFilterModePoint; */
/*   mPb.normalized = 0; */
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray* mPbArray;
  CUDA_SAFE_CALL(cudaMallocArray(&mPbArray, &channelDesc, width, height));
  CUDA_SAFE_CALL(cudaBindTextureToArray(mPb, mPbArray));
  CUDA_SAFE_CALL(cudaMemcpy2DToArray(mPbArray, 0, 0, devMPb, sizeof(float) * width, sizeof(float) * width, height, cudaMemcpyDeviceToDevice));
  size_t devMatrixPitch;
  CUDA_SAFE_CALL(cudaMallocPitch((void**)p_devMatrix, &devMatrixPitch, nPixels * sizeof(float), nDimension));
  assert(devMatrixPitch == theStencil.getMatrixPitch());
  float* devMatrix = *p_devMatrix;
  CUDA_SAFE_CALL(cudaMemset(devMatrix, 0, devMatrixPitch * nDimension));
  int hostDiagonalMap[CONSTSPACE];
  theStencil.copyDiagonalOffsets(hostDiagonalMap);
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(constDiagonals, hostDiagonalMap, sizeof(int) * diameter * diameter));
  dim3 gridDim = dim3((width - 1)/XBLOCK + 1, (height - 1)/YBLOCK + 1);
  dim3 blockDim = dim3(XBLOCK, YBLOCK);
  
  findAffinities<<<gridDim, blockDim>>>(width, height, radius, diameter, 1.0f/sigma, devMatrix, matrixPitchInFloats);//, devScratch);
  symmetrizeMatrix<<<gridDim, blockDim>>>(width, height, radius, diameter, nDimension, devMatrix, matrixPitchInFloats);//, devScratch);
  cudaThreadSynchronize();
  CUDA_SAFE_CALL(cudaUnbindTexture(mPb));
  CUDA_SAFE_CALL(cudaFreeArray(mPbArray));
}
