#include "stream_pass_data.cuh"
#include "../common/test_utils.cuh"
#include "../../cuda/pch.h"

// 空核函数实现 - 仅用于数据传输测试
__global__ void empty_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 空操作 - 仅用于测试数据传输
        // 后续可以替换为实际的业务逻辑
        data[idx] = data[idx]; // 简单的数据传递
    }
}

// 流式处理实现
void stream_pass_data_test(
    float** h_data, 
    int n_groups, 
    int n_vectors, 
    int n_dim, 
    StreamMode mode,
    double* duration_ms,
    float* h_output
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 计算总数据量
    int total_elements = n_groups * n_vectors * n_dim;
    size_t data_size = total_elements * sizeof(float);
    
    // 分配GPU内存
    float* d_data;
    cudaError_t err = cudaMalloc(&d_data, data_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA内存分配失败: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // 分配CPU输出内存 - 使用传入的指针
    // h_output 已经在调用者中分配
    
    cudaEventRecord(start);
    
    if (mode == STREAM_MODE) {
        // 流式处理模式
        const int num_streams = 4;
        cudaStream_t streams[num_streams];
        
        // 创建CUDA流
        for (int i = 0; i < num_streams; i++) {
            err = cudaStreamCreate(&streams[i]);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA流创建失败: %s\n", cudaGetErrorString(err));
                return;
            }
        }
        
        // 分批处理数据
        int elements_per_stream = total_elements / num_streams;
        int remaining_elements = total_elements % num_streams;
        
        for (int i = 0; i < num_streams; i++) {
            int stream_elements = elements_per_stream;
            if (i == num_streams - 1) {
                stream_elements += remaining_elements; // 最后一个流处理剩余数据
            }
            
            int offset = i * elements_per_stream;
            size_t stream_data_size = stream_elements * sizeof(float);
            
            // 异步上传数据到GPU
            err = cudaMemcpyAsync(
                d_data + offset,
                h_data[0] + offset, // 数据是连续的
                stream_data_size,
                cudaMemcpyHostToDevice,
                streams[i]
            );
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA异步内存拷贝失败: %s\n", cudaGetErrorString(err));
                return;
            }
            
            // 启动核函数
            int block_size = 256;
            int grid_size = (stream_elements + block_size - 1) / block_size;
            empty_kernel<<<grid_size, block_size, 0, streams[i]>>>(
                d_data + offset, 
                stream_elements
            );
            // 检查核函数启动错误
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA核函数启动失败: %s\n", cudaGetErrorString(err));
                return;
            }
            
            // 异步下载数据到CPU
            err = cudaMemcpyAsync(
                h_output + offset,
                d_data + offset,
                stream_data_size,
                cudaMemcpyDeviceToHost,
                streams[i]
            );
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA异步内存拷贝失败: %s\n", cudaGetErrorString(err));
                return;
            }
        }
        
        // 同步所有流
        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
        }
        
        // 销毁流
        for (int i = 0; i < num_streams; i++) {
            cudaStreamDestroy(streams[i]);
        }
        
    } else {
        // 直接处理模式
        // 同步上传数据
        err = cudaMemcpy(d_data, h_data[0], data_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA内存拷贝失败: %s\n", cudaGetErrorString(err));
            return;
        }
        
        // 启动核函数
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        empty_kernel<<<grid_size, block_size>>>(d_data, total_elements);
        // 检查核函数启动错误
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA核函数启动失败: %s\n", cudaGetErrorString(err));
            return;
        }
        
        // 同步下载数据
        err = cudaMemcpy(h_output, d_data, data_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA内存拷贝失败: %s\n", cudaGetErrorString(err));
            return;
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    *duration_ms = (double)milliseconds;
    
    // 清理资源
    cudaFree(d_data);
    // 注意：不释放h_output，由调用者负责释放
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 直接处理实现（简化版本，用于对比）
void direct_pass_data_test(
    float** h_data, 
    int n_groups, 
    int n_vectors, 
    int n_dim, 
    double* duration_ms,
    float* h_output
) {
    stream_pass_data_test(h_data, n_groups, n_vectors, n_dim, DIRECT_MODE, duration_ms, h_output);
}
