#include <stdio.h>
#include <math.h>
#include <complex>
//#include "filters.h"
#include <assert.h>
#include <cstdlib>
using namespace std;


int sign(int i)
{
    if(i==0) return 0;
    if(i>0) return 1;
    else return (-1);

}
float* gaussian(
   double sigma, unsigned int deriv, bool hlbrt, unsigned long support)
{
   /* enlarge support so that hilbert transform can be done efficiently */
   unsigned long support_big = support;
//   if(hlbrt)
//   {
//       printf("hilebert not supported\n");
//   }
//   if (hlbrt) {
//      support_big = 1;
//      unsigned long temp = support;
//      while (temp > 0) {
//         support_big *= 2;
//         temp /= 2;
//      }
//   }
   /* compute constants */
   const double sigma2_inv = double(1)/(sigma*sigma);
   const double neg_two_sigma2_inv = double(-0.5)*sigma2_inv;
   /* compute gaussian (or gaussian derivative) */
   unsigned long size = 2*support_big + 1;
   float *m = new float[size];
   double x = -(static_cast<double>(support_big)); 
   if (deriv == 0) {
      /* compute guassian */
      for (unsigned long n = 0; n < size; n++, x++)
         m[n] = exp(x*x*neg_two_sigma2_inv);
   } else if (deriv == 1) {
      /* compute gaussian first derivative */
      for (unsigned long n = 0; n < size; n++, x++)
         m[n] = exp(x*x*neg_two_sigma2_inv) * (-x);
   } else if (deriv == 2) {
      /* compute gaussian second derivative */
      for (unsigned long n = 0; n < size; n++, x++) {
         double x2 = x * x;
         m[n] = exp(x2*neg_two_sigma2_inv) * (x2*sigma2_inv - 1);
      }
   } else {
      //throw ex_invalid_argument("only derivatives 0, 1, 2 supported");
      printf("only derivatives 0, 1, 2 supported\n");
   }
   /* take hilbert transform (if requested) */
//   if (hlbrt) {
//      /* grab power of two sized submatrix (ignore last element) */
//      //size--;
//      //m._dims[0]--;
//      /* grab desired submatrix after hilbert transform */
//      unsigned long start[2];
//      unsigned long end[2];
//      start[0] = support_big - support;
//      end[0] = start[0] + support + support;
//      m = (hilbert(m)).submatrix(start, end);
//   }


   float* hil;

   complex<float> dft[size], idft[size];

   if(hlbrt)
   {
       hil = new float[size];
       for(int i=0;i<size;i++)
       {
           hil[i]=0;
           for(int j=0;j<size;j++)
           {
//               if(j%2 != i%2)
//               {
//                   hil[i] += m[j] * 2/tan(M_PI*float(i-j)/float(size))/float(size);
//               }
           complex<float> factor(cos(2*M_PI*i*j/size), -sin(2*M_PI*i*j/size));               
           dft[i] += m[j]*factor;               
           }
           //hil[i] /= float(size);
           //assert(hil[i] == m[i]);
           //printf("%d: %f %f %f \n", i, m[i], real(dft[i]), imag(dft[i]));
       }
       for(int i=0;i<size;i++)
       {
           if(i!=0) // leave dc component unchanged
           {
               if(i<(size+1)/2)
               {
                   dft[i] *=2;
               }
               else if(i>size/2+1)
               {
                   dft[i]=0;
               }
           }

           //printf("%d: %f %f %f \n", i, m[i], real(dft[i]), imag(dft[i]));

       }
       for(int i=0;i<size;i++)
       {
           for(int j=0;j<size;j++)
           {
               complex<float> factor(cos(2*M_PI*i*j/size)/size, sin(2*M_PI*i*j/size)/size);
               idft[i] += dft[j]*factor;
           }
           hil[i] = imag(idft[i]);
           //assert(abs(real(idft[i])-m[i]) < 1e-3);
           //printf("%d: %f %f %f \n", i, m[i], real(idft[i]), imag(idft[i]));
       }
       //printf("Computed hilbert transform\n");
       delete [] m;
       m = hil;
   }
   /* make zero mean (if returning derivative) */
   if (deriv > 0) 
   {
       float sum=0;
       for(int i=0;i<size;i++)
       {
           sum+= m[i];
       }
       float mean=sum/size;
       for(int i=0;i<size;i++)
       {
           m[i] -= mean;
       }
      //m -= mean(m);
   }
   /* make unit L1 norm */
   float sabs=0;
   for(int i=0;i<size;i++)
   {
       sabs += abs(m[i]);
   }
   for(int i=0;i<size;i++)
   {
       m[i] /= sabs;
   }
   //m /= sum(abs(m));
   return m;
}

double support_x_rotated(double support_x, double support_y, double ori) {
   const double sx_cos_ori = support_x * cos(ori);
   const double sy_sin_ori = support_y * sin(ori);
   double x0_mag = abs(sx_cos_ori - sy_sin_ori);
   double x1_mag = abs(sx_cos_ori + sy_sin_ori);
   return (x0_mag > x1_mag) ? x0_mag : x1_mag;
}

double support_y_rotated(double support_x, double support_y, double ori) {
   const double sx_sin_ori = support_x * sin(ori);
   const double sy_cos_ori = support_y * cos(ori);
   double y0_mag = abs(sx_sin_ori - sy_cos_ori);
   double y1_mag = abs(sx_sin_ori + sy_cos_ori);
   return (y0_mag > y1_mag) ? y0_mag : y1_mag;
}
unsigned long support_x_rotated(
   unsigned long support_x, unsigned long support_y, double ori)
{
   return static_cast<unsigned long>(
      ceil(support_x_rotated(
         static_cast<double>(support_x), static_cast<double>(support_y), ori
      ))
   );
}

unsigned long support_y_rotated(
   unsigned long support_x, unsigned long support_y, double ori)
{
   return static_cast<unsigned long>(
      ceil(support_y_rotated(
         static_cast<double>(support_x), static_cast<double>(support_y), ori
      ))
   );
}

template <typename T>
void compute_rotate_2D(
   const T*      m_src,       /* source matrix */
   T*            m_dst,       /* destination matrix */
   unsigned long size_x_src,  /* size of source */
   unsigned long size_y_src,
   unsigned long size_x_dst,  /* size of destination */
   unsigned long size_y_dst,
   double ori)                /* orientation */
{
   /* check that matrices are nonempty */
   if ((size_x_src > 0) && (size_y_src > 0) &&
       (size_x_dst > 0) && (size_y_dst > 0))
   {
      /* compute sin and cos of rotation angle */
      const double cos_ori = cos(ori);
      const double sin_ori = sin(ori);
      /* compute location of origin in src */
      const double origin_x_src = static_cast<double>((size_x_src - 1)) / 2;
      const double origin_y_src = static_cast<double>((size_y_src - 1)) / 2;
      /* rotate */
      double u = -(static_cast<double>((size_x_dst - 1)) / 2);
      unsigned long n = 0;
      for (unsigned long dst_x = 0; dst_x < size_x_dst; dst_x++) {
         double v = -(static_cast<double>((size_y_dst - 1)) / 2);
         for (unsigned long dst_y = 0; dst_y < size_y_dst; dst_y++) {
            /* reverse rotate by orientation and shift by origin offset */
            double x = u * cos_ori + v * sin_ori + origin_x_src;
            double y = v * cos_ori - u * sin_ori + origin_y_src;
            /* check that location is in first quadrant */
            if ((x >= 0) && (y >= 0)) {
               /* compute integer bounds on location */
               unsigned long x0 = static_cast<unsigned long>(floor(x));
               unsigned long x1 = static_cast<unsigned long>(ceil(x));
               unsigned long y0 = static_cast<unsigned long>(floor(y));
               unsigned long y1 = static_cast<unsigned long>(ceil(y));
               /* check that location is within src matrix */
               if ((0 <= x0) && (x1 < size_x_src) &&
                   (0 <= y0) && (y1 < size_y_src))
               {
                  /* compute distances to bounds */
                  double dist_x0 = x - x0;
                  double dist_x1 = x1 - x;
                  double dist_y0 = y - y0;
                  double dist_y1 = y1 - y;
                  /* grab matrix elements */
                  const T& m00 = m_src[x0*size_y_src + y0];
                  const T& m01 = m_src[x0*size_y_src + y1];
                  const T& m10 = m_src[x1*size_y_src + y0];
                  const T& m11 = m_src[x1*size_y_src + y1];
                  /* interpolate in x-direction */
                  const T t0 =
                     (x0 != x1) ? (dist_x1 * m00 + dist_x0 * m10) : m00;
                  const T t1 =
                     (x0 != x1) ? (dist_x1 * m01 + dist_x0 * m11) : m01;
                  /* interpolate in y-direction */
                  m_dst[n] = (y0 != y1) ? (dist_y1 * t0 + dist_y0 * t1) : t0;
               }
            }
            /* increment coordinate */
            n++;
            v++;
         }
         u++;
      }
   }
}


void gaussian_2D(
   float* filter,
   double        sigma_x, 
   double        sigma_y,
   double        ori,
   unsigned int  deriv,
   bool          hlbrt,
   unsigned long support_x,
   unsigned long support_y)
{
   /* compute size of larger grid for axis-aligned gaussian   */
   /* (reverse rotate corners of bounding box by orientation) */
   unsigned long support_x_rot = support_x_rotated(support_x, support_y, -ori);
   unsigned long support_y_rot = support_y_rotated(support_x, support_y, -ori);
   /* compute 1D kernels */
   float* mx = gaussian(sigma_x, 0,     false, support_x_rot);
   float* my = gaussian(sigma_y, deriv, hlbrt, support_y_rot);
   int mx_size = 2*support_x_rot+1;
   int my_size= 2*support_y_rot+1;
   /* compute 2D kernel from product of 1D kernels */
   int m_size = mx_size* my_size;
   float* m=new float[m_size];
   unsigned long n = 0;
   for (unsigned long n_x = 0; n_x < mx_size; n_x++) {
      for (unsigned long n_y = 0; n_y < my_size; n_y++) {
         m[n] = mx[n_x] * my[n_y];
         n++;
      }
   }
   /* rotate 2D kernel by orientation */

   float* mrotate = new float[(2*support_x + 1)*(2*support_y + 1)];
   compute_rotate_2D(m, mrotate, mx_size, my_size, 2*support_x + 1, 2*support_y + 1, ori);

   delete [] m,mx,my;
   m = mrotate;
   int size=(2*support_x + 1)*(2*support_y + 1);
   //m = rotate_2D_crop(m, ori, 2*support_x + 1, 2*support_y + 1);
   /* make zero mean (if returning derivative) */
   //if (deriv > 0)
   //   m -= mean(m);
   /* make unit L1 norm */
   //m /= sum(abs(m));

   if (deriv > 0) 
   {
       float sum=0;
       for(int i=0;i<size;i++)
       {
           sum+= m[i];
       }
       float mean=sum/size;
       for(int i=0;i<size;i++)
       {
           m[i] -= mean;
       }
      //m -= mean(m);
   }
   /* make unit L1 norm */
   float sabs=0;
   for(int i=0;i<size;i++)
   {
       sabs += abs(m[i]);
   }
   for(int i=0;i<size;i++)
   {
       m[i] /= sabs;
   }

   for(int i=0;i<size;i++)
   {
       filter[i] = m[i];
   }
   delete [] mrotate;

}


void gaussian_cs_2D(
   float* filter,
   double        sigma_x, 
   double        sigma_y,
   double        ori,
   double        scale_factor,
   unsigned long support_x,
   unsigned long support_y)
{
   /* compute standard deviation for center kernel */
   double sigma_x_c = sigma_x / scale_factor;
   double sigma_y_c = sigma_y / scale_factor;
   /* compute center and surround kernels */
   float* m_center, *m_surround;
   m_center = new float[(2*support_x+1)*(2*support_y+1)];
   m_surround = new float[(2*support_x+1)*(2*support_y+1)];


   gaussian_2D(
      m_center, sigma_x_c, sigma_y_c, ori, 0, false, support_x, support_y
   );
   gaussian_2D(
      m_surround, sigma_x, sigma_y, ori, 0, false, support_x, support_y
   );
   
   /* compute center-surround kernel */
   int size = (2*support_x+1)*(2*support_y+1);
   float* m = new float[size];

   // m = m_surround - m_center;
   /* make zero mean and unit L1 norm */
   //m -= mean(m);
   //m /= sum(abs(m));
   
   float sum=0;
       for(int i=0;i<size;i++)
       {
           m[i]=m_surround[i]-m_center[i];
           sum+= m[i];
       }
       float mean=sum/size;
       for(int i=0;i<size;i++)
       {
           m[i] -= mean;
       }
   /* make unit L1 norm */
   float sabs=0;
   for(int i=0;i<size;i++)
   {
       sabs += abs(m[i]);
   }
   for(int i=0;i<size;i++)
   {
       m[i] /= sabs;
   }

   for(int i=0;i<size;i++)
   {
       filter[i] = m[i];
   }
   delete [] m;
   
}


void createTextonFilters(float** filters, int* nFilterCoeff, int** radii, float* scale, int nscales, float elongation=3.0)
{
    unsigned int size=0;
    for(int n=0; n<nscales; n++)
    {
        int filterLength = ceil(3*scale[n])*2+1;
        size += filterLength*filterLength*17;
    }
    //printf("allocating %u floats for filters \n", size);
    *filters = (float*) malloc(size*sizeof(float));
    *radii = (int*)malloc(17*nscales*sizeof(int));

    int* aradii = *radii;
    int i=0;
    unsigned int r =0;
    int support, filterLength;

    for(int n = 0; n<nscales; n++)
    {
        support=ceil(3*scale[n]);
        filterLength = support*2 + 1;
    
        int orient;
//    *filters = (float*)realloc(*filters, 17*filterLength*sizeof(float));

        for(orient=0;orient<8;orient += 1)
        {
            gaussian_2D(*filters+i, scale[n], scale[n]/elongation, orient*M_PI/8, 2, false, support, support);
            //printf("creating filter : %f %f %f %d %d\n", scale[n], scale[n]/elongation, orient*M_PI/8,support, support);

            i += filterLength*filterLength;
            aradii[r] = support;
            r++;
        }
        for(orient=0;orient<8;orient += 1)
        {
            gaussian_2D(*filters+i, scale[n], scale[n]/elongation, orient*M_PI/8, 2, true, support, support);
            //printf("creating filter : %f %f %f %d %d\n", scale[n], scale[n]/elongation, orient*M_PI/8,support, support);
            i+= filterLength*filterLength;
            aradii[r] = support; 
            r++;
        }
    
        gaussian_cs_2D(*filters+i, scale[n],scale[n],0, M_SQRT2l, support, support);
        //printf("creating filter : %f %f %f %d %d\n", scale[n], scale[n], 0.0f, support, support);
        i+= filterLength*filterLength;
        aradii[r] = support;     
        r++;
    }
    *nFilterCoeff = i-filterLength*filterLength;
}
