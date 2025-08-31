#include "kernels.h"
#include "normalize.h"
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include "cudatimer.h"

/*
* normalization接口
* Args:
*   vector_data_cpu: cpu上的原始向量组的指针
*   n_dim: 向量维数 
*   n_batch: 一组中向量个数 
*/ 
void normalize(float** h_vector_list, int n_batch, int n_dim) {
    assert(h_vector_list != nullptr && n_dim > 0 && n_batch > 0);
    #define ENABLE_CUDA_TIMING 0 /*是否启用CUDATimer计时*/

    float *d_vector_list, *d_vector_l2_norm;
    dim3 gridDim(n_batch);
    dim3 blockDim(n_dim);

    /* GPU内存分配 */
    {
        CUDATimer timer_manage("GPU Memory Allocation", ENABLE_CUDA_TIMING, false);
        cudaMalloc((void**)&d_vector_list, n_batch * n_dim * sizeof(float));
        cudaMalloc((void**)&d_vector_l2_norm, n_batch * sizeof(float));        
    }

    /* 数据传输 CPU->GPU */ 
    { 
        CUDATimer timer_trans1("H2D Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemset(d_vector_l2_norm, 0, n_batch * sizeof(float));
        cudaMemcpy2D(
            d_vector_list,
            n_dim * sizeof(float),
            h_vector_list[0],
            n_dim * sizeof(float),
            n_dim * sizeof(float),
            n_batch,
            cudaMemcpyHostToDevice
        );
    }

    /* 核函数执行 */
    {
        CUDATimer timer_compute("Kernel Execution", ENABLE_CUDA_TIMING);
        l2_norm_kernel<<<gridDim, blockDim, n_batch * sizeof(float)>>>(
            d_vector_list,
            d_vector_l2_norm, 
            n_batch, n_dim
        );

        normalize_kernel<<<gridDim, n_dim>>>(
            d_vector_list, 
            d_vector_l2_norm, 
            n_batch, n_dim
        );
        
        // 注意：这里没有显式同步，同步在CUDATimer析构函数中的cudaEventSynchronize(stop_event_)完成
    }

    /* 数据传输 GPU->CPU */
    { 
        CUDATimer timer_trans2("D2H Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemcpy2D(
            h_vector_list[0],
            n_dim * sizeof(float),
            d_vector_list,
            n_dim * sizeof(float),
            n_dim * sizeof(float),
            n_batch,
            cudaMemcpyDeviceToHost
        );
    }

    /* GPU内存释放 */
    {
        CUDATimer timer_manage2("GPU Memory Free", ENABLE_CUDA_TIMING, false);
        cudaFree(d_vector_list);
        cudaFree(d_vector_l2_norm);   
    }

    #undef ENABLE_CUDA_TIMING
}



/*
* 异步normalization接口
* Args:
*   vector_data_cpu: float*** cpu上的多个原始向量组的，形状为 [n_lists, n_batch, n_dim]
*   n_lists: 向量组个数 
*   n_dim: 向量维数 
*   n_batch: 一组中向量个数
* 
* 使用多个cuda流（一般2-4个效果较好）和异步内存复制进行传输，调用前将原始数据所在的内存设定为页固定
* cpu-gpu传输带宽达到10GB/s
*/ 
void normalize_async(float*** vector_lists_cpu, int n_lists, int n_batch, int n_dim) {
    assert(vector_lists_cpu != nullptr && n_dim > 0 && n_batch > 0);
    #define ENABLE_CUDA_TIMING 1 /*是否启用CUDATimer计时*/
    
    const int NUM_STREAMS = 2; // cuda流的数量
    dim3 gridDim(n_batch);
    dim3 blockDim(n_dim);

    // 创建CUDA流和用于流间同步的事件
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t h2d_events[NUM_STREAMS], compute_events[NUM_STREAMS];

    // 双缓冲设备内存
    float *d_vector_data[NUM_STREAMS], *d_vector_square_sum[NUM_STREAMS];
    
    /* 创建流和事件 */
    {
        CUDATimer timer_alloc("Create cuda streams and cuda events", ENABLE_CUDA_TIMING, false);
        for (int i = 0; i < NUM_STREAMS; i++) {
            // cudaStreamCreate(&streams[i]);
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);/*使用非阻塞流*/
            cudaEventCreate(&h2d_events[i]);
            cudaEventCreate(&compute_events[i]);
        }
    }

    // 2. 异步分配设备内存 (CUDA 11.2+)
    {
        CUDATimer timer_alloc("Async GPU Memory Allocation", ENABLE_CUDA_TIMING, false);
        for (int i = 0; i < NUM_STREAMS; i++) {
            // 使用异步内存分配，指定流
            cudaMallocAsync((void**)&d_vector_data[i], n_batch * n_dim * sizeof(float), streams[i]);
            cudaMallocAsync((void**)&d_vector_square_sum[i], n_batch * sizeof(float), streams[i]);
            // 初始化square_sum为0
            cudaMemsetAsync(d_vector_square_sum[i], 0, n_batch * sizeof(float), streams[i]);
        }
    }

    // 3. 流水线处理
    {
        CUDATimer timer_alloc("Pipeline Process", ENABLE_CUDA_TIMING);
        for (int list_id = 0; list_id < n_lists; list_id++) {
            int stream_id = list_id % NUM_STREAMS; // 循环使用流

            // 等待前一个使用相同流的内核计算完成，避免覆盖正在计算的数据
            if (list_id >= NUM_STREAMS) {
                // 等待该流上上一个计算的完成，确保H2D不会覆盖正在计算的数据
                cudaStreamWaitEvent(streams[stream_id], compute_events[stream_id], 0);
            }

            // a. 异步H2D拷贝
            // cudaMemcpyAsync(
            //     d_vector_data[stream_id], 
            //     vector_lists_cpu[list_id][0], 
            //     n_batch * n_dim * sizeof(float), 
            //     cudaMemcpyHostToDevice,
            //     streams[stream_id]
            // );
            cudaMemcpy2DAsync(
                d_vector_data[stream_id],
                n_dim * sizeof(float),
                vector_lists_cpu[list_id][0], // 假设内存是连续的或已正确处理
                n_dim * sizeof(float),
                n_dim * sizeof(float),
                n_batch,
                cudaMemcpyHostToDevice,
                streams[stream_id]
            );
            // 记录H2D完成事件
            cudaEventRecord(h2d_events[stream_id], streams[stream_id]);

            // b. 计算平方和 (需要等待H2D完成)
            l2_norm_kernel<<<gridDim, blockDim, n_dim * sizeof(float), streams[stream_id]>>>(
                d_vector_data[stream_id],
                d_vector_square_sum[stream_id],
                n_batch, n_dim
            );

            // c. 归一化 (依赖平方和计算完成，但在同一个流中，所以是顺序的)
            normalize_kernel<<<gridDim, blockDim, 0, streams[stream_id]>>>(
                d_vector_data[stream_id],
                d_vector_square_sum[stream_id],
                n_batch, n_dim
            );
            // 记录计算完成事件
            cudaEventRecord(compute_events[stream_id], streams[stream_id]);

            // d. 异步D2H拷贝 (需要等待归一化计算完成)
            // 但D2H在同一个流中，所以会自动等待前面的内核完成
            // cudaMemcpyAsync(
            //     vector_lists_cpu[list_id][0], 
            //     d_vector_data[stream_id], 
            //     n_batch * n_dim * sizeof(float), 
            //     cudaMemcpyDeviceToHost,
            //     streams[stream_id]
            // );
            cudaMemcpy2DAsync(
                vector_lists_cpu[list_id][0],
                n_dim * sizeof(float),
                d_vector_data[stream_id],
                n_dim * sizeof(float),
                n_dim * sizeof(float),
                n_batch,
                cudaMemcpyDeviceToHost,
                streams[stream_id]
            );
        }
    }

    // 5. 同步所有流，确保所有操作完成
    {
        CUDATimer timer_sync("Synchronize Streams", ENABLE_CUDA_TIMING, false);
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamSynchronize(streams[i]);
        }
    }

    // 6. 异步释放设备内存和销毁流、事件
    {
        CUDATimer timer_free("Async Resource Free", ENABLE_CUDA_TIMING, false);
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaFreeAsync(d_vector_data[i], streams[i]);
            cudaFreeAsync(d_vector_square_sum[i], streams[i]);
            cudaEventDestroy(h2d_events[i]);
            cudaEventDestroy(compute_events[i]);
            cudaStreamDestroy(streams[i]);
        }
    }
    
    #undef ENABLE_CUDA_TIMING
}