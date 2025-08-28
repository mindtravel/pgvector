#include <cuda_runtime.h>
#include <stdio.h>

// CUDA核函数：简单的Hello World
__global__ void cuda_hello_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        printf("Hello from CUDA kernel! Thread ID: %d\n", tid);
    }
}

// C接口函数
extern "C" {
    int cuda_hello_world() {
        // 设置CUDA核函数参数
        int block_size = 256;
        int grid_size = 1;
        
        // 调用CUDA核函数
        cuda_hello_kernel<<<grid_size, block_size>>>();
        
        // 同步设备
        cudaDeviceSynchronize();
        
        // 检查错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        printf("CUDA Hello World executed successfully!\n");
        return 0;
    }
    
    // 检查CUDA是否可用
    bool cuda_is_available() {
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess && device_count > 0);
    }
}