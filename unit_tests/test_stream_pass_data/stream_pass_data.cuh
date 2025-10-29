#ifndef STREAM_PASS_DATA_CUH
#define STREAM_PASS_DATA_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>

// 流式处理模式枚举
enum StreamMode {
    STREAM_MODE,    // 流式处理
    DIRECT_MODE     // 直接处理
};

// 空核函数 - 用于数据传输测试
__global__ void empty_kernel(float* data, int size);

// 流式处理函数
void stream_pass_data_test(
    float** h_data, 
    int n_groups, 
    int n_vectors, 
    int n_dim, 
    StreamMode mode,
    double* duration_ms,
    float* h_output  // 输出数据用于验证
);

// 直接处理函数
void direct_pass_data_test(
    float** h_data, 
    int n_groups, 
    int n_vectors, 
    int n_dim, 
    double* duration_ms,
    float* h_output  // 输出数据用于验证
);

#endif // STREAM_PASS_DATA_CUH
