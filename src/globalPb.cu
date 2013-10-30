// vim: ts=4 syntax=cpp comments=

#include "globalPb.h"
#define XSIZE 256
__global__ void CalcGPb(int p_nPixels, int p_nMatrixPitch, int p_nOrient, 
                        float* devcombinedg, float* devspb, float* devmpb, float* devGpball,
                        float* devResult)
{
  int index = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  if (index < p_nPixels)
    {
      //float weight[] = {0,   0,    0.0028,    0.0041,    0.0042,    0.0047,    0.0033,    0.0033,    0.0035,    0.0025,    0.0025,    0.0137,    0.0139};
      float weight12 = 0.0139;
      float maxValue = -200;
      for (int i = 0; i < p_nOrient; i++)
        {
          int orientedIndex = index + i * p_nMatrixPitch;
          float weightedSum = 
            devcombinedg[orientedIndex] + 
            weight12 * devspb[orientedIndex];
          devGpball[orientedIndex] = weightedSum;
          if (weightedSum > maxValue)
            {
              maxValue = weightedSum;
            }
        }
      /* 		// Normalize Output */
      /* 		if (maxValue*1.2 > 1) */
      /* 		{ */
      /* 			maxValue = 1; */
      /* 		} */
      /* 		else */
      /* 		{ */
      /* 			maxValue *=1.2; */
      /* 		} */
      /* 		if (maxValue < 0) */
      /* 		{ */
      /* 			maxValue = 0; */
      /* 		} */
		
      /* 		maxValue = 1 / (1 + expf(2.6433-10.7998*maxValue)); */
      /* 		maxValue = (maxValue - 0.0667)/0.9333; */
      if (devmpb[index] > 0.05) {
        devResult[index] = maxValue;
      } else {
        devResult[index] = 0.0f;
      }
    }
}

/* __global__ void CalcGPb(int p_nPixels, int p_nMatrixPitch, int p_nOrient, 
                        float* devbg1, float* devbg2, float* devbg3, 
                        float* devcga1, float* devcga2, float* devcga3, 
                        float* devcgb1, float* devcgb2, float* devcgb3,
                        float* devtg1, float* devtg2, float* devtg3, 
                        float* devspb, float* devmpb, float* devGpball,
                        float* devResult)
{
  int index = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  if (index < p_nPixels)
    {
      float weight[] = {0,   0,    0.0028,    0.0041,    0.0042,    0.0047,    0.0033,    0.0033,    0.0035,    0.0025,    0.0025,    0.0137,    0.0139};
      float maxValue = -200;
      for (int i = 0; i < p_nOrient; i++)
        {
          int orientedIndex = index + i * p_nMatrixPitch;
          float weightedSum = 
            weight[0] *  devbg1[orientedIndex] + 
            weight[1] *  devbg2[orientedIndex] + 
            weight[2] *  devbg3[orientedIndex] + 
            weight[3] * devcga1[orientedIndex] + 
            weight[4] * devcga2[orientedIndex] + 
            weight[5] * devcga3[orientedIndex] + 
            weight[6] * devcgb1[orientedIndex] + 
            weight[7] * devcgb2[orientedIndex] + 
            weight[8] * devcgb3[orientedIndex] + 
            weight[9] *  devtg1[orientedIndex] + 
            weight[10] * devtg2[orientedIndex] + 
            weight[11] * devtg3[orientedIndex] + 
            weight[12] * devspb[orientedIndex];
          devGpball[orientedIndex] = weightedSum;
          if (weightedSum > maxValue)
            {
              maxValue = weightedSum;
            }
        }
      if (devmpb[index] > 0.05) {
        devResult[index] = maxValue;
      } else {
        devResult[index] = 0.0f;
      }
    }
}
*/

  void StartCalcGPb(int p_nPixels, int p_nMatrixPitch, int p_nOrient,
                    float* devcombinedg, float* devspb, float* devmpb, float* devGpball,
                    float* devResult)
  {
	
    dim3 blockDim(XSIZE, 1);
    dim3 gridDim((p_nPixels - 1)/XSIZE + 1, 1);
    CalcGPb<<<gridDim, blockDim>>>(p_nPixels, p_nMatrixPitch, p_nOrient, 
                                   devcombinedg, devspb, devmpb, devGpball,
                                   devResult);
  }
																																		
