#pragma once

#define CUDA_SAFE_CALL(ans) { __assert_cuda((ans), __FILE__, __LINE__); }
inline void __assert_cuda(cudaError_t code,
                          char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n",
              cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define CHECK_ERROR(warn) { __check_error((warn), __FILE__, __LINE__); }
inline void __check_error(char* warning, char* file, int line) {
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"At %s, CUDA Error: %s %s %d\n",
                warning, cudaGetErrorString(code), file, line);
        exit(code);
    }
}

struct cuda_timer {
    cudaEvent_t m_start, m_stop;
    inline cuda_timer() {
        m_start = 0;
        m_stop = 0;
    }
    inline ~cuda_timer() {
        if (m_start != 0) {
            cudaEventDestroy(m_start);
        }
        if (m_start != 0) {
            cudaEventDestroy(m_stop);
        }
    }
    inline void start() {
        if (m_start != 0) {
            cudaEventDestroy(m_start);
        }
        if (m_start != 0) {
            cudaEventDestroy(m_stop);
        }
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
        cudaEventRecord(m_start, 0);
    }
    inline float stop() {
        float time;
        cudaEventRecord(m_stop, 0);
        cudaEventSynchronize(m_stop);
        cudaEventElapsedTime(&time, m_start, m_stop);
        return time;
    }
};
