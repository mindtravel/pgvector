#include "distances.h"
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include "CUDATimer.h"

/*
* CUDA核函数：计算模长
* Args:
*   vector_data: 原始向量组
*   vetcor_suqared_sum: 规范化后向量组的l2 norm
*   n_dim: 向量维数 
*   n_batch: 一组中向量个数 
*  
* 采用原地修改向量组的方式，在向量的分量上并行计算元素的平方，再使用树形规约
* 速度更快但是无法保留原始向量信息
*/
__global__ void compute_square_sum(float *vector_data, float *vector_square_sum, int n_batch, int n_dim) {
    extern __shared__ float shared_mem[];


    /*
        先计算每个元素的平方，因为计算平方不用知道哪个元素属于哪个向量，所以直接计算就完事
    */
    int tid = threadIdx.x; /*当前线程块内索引*/
    int bid = blockIdx.x; /*当前线程块内索引*/
    int idx = blockDim.x * blockIdx.x + threadIdx.x; /*当前线程块对应全局索引*/

    // shared_mem[tid] = 0;
    // __syncthreads();

    // float sum = 0.0f;
    // if (idx < n_dim * n_batch) 
    //     vetcor_suqare_sum[bid] += vector_data[bid][tid] * vector_data[bid][tid];  

    float square = 0.0f;
    if (tid < n_dim) {
        square = vector_data[idx] * vector_data[idx];
    }

    shared_mem[tid] = square;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    // shared_mem[tid] = sum; /*分段计算平方*/
    // __syncthreads(); /*一个​​块内屏障​​，确保线程块中​​所有线程​​都已将各自的平方结果写入共享内存*/
    
    if (tid == 0) {
        vector_square_sum[bid] = shared_mem[0];
    }
    __syncthreads(); 

}
    

/*
* CUDA核函数：按模长归一化
* Args:
*   vector_data: 原始向量组
*   vetcor_suqared_sum: 规范化后向量组的l2 norm
*   n_dim: 向量维数 
*   n_batch: 一组中向量个数 
*/
__global__ void vector_normalize(float *vector_data, float* vector_norms, int n_batch, int n_dim) {
    int bid = blockIdx.x;
    // int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // int pos_idx = idx % n_dim;
    // int vector_idx = idx / n_dim;

    /*全零向量不做normalization*/
    if(vector_norms[bid] != 0)
        vector_data[idx] /= sqrt(vector_norms[bid]);

}

/*
* 构造和析构
*/
VectorNormalizer::VectorNormalizer() : norm_(0.0f) {}
VectorNormalizer::~VectorNormalizer() {}

/*
* normalization接口
* Args:
*   vector_data_cpu: cpu上的原始向量组的指针
*   n_dim: 向量维数 
*   n_batch: 一组中向量个数 
*/ 
void VectorNormalizer::normalize(float** vector_list_cpu, int n_batch, int n_dim) {
    assert(vector_list_cpu != nullptr && n_dim > 0 && n_batch > 0);
    #define ENABLE_CUDA_TIMING 0 /*是否启用CUDATimer计时*/

    float *vector_data, *vector_square_sum;
    dim3 gridDim(n_batch);
    dim3 blockDim(n_dim);

    /* GPU内存分配 */
    {
        CUDATimer timer_manage("GPU Memory Allocation", ENABLE_CUDA_TIMING, false);
        cudaMalloc((void**)&vector_data, n_batch * n_dim * sizeof(float));
        cudaMalloc((void**)&vector_square_sum, n_batch * sizeof(float));        
    }

    /* 数据传输 CPU->GPU */ 
    { 
        CUDATimer timer_trans1("H2D Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemset(vector_square_sum, 0, n_batch * sizeof(float));
        cudaMemcpy2D(
            vector_data,
            n_dim * sizeof(float),
            vector_list_cpu[0],
            n_dim * sizeof(float),
            n_dim * sizeof(float),
            n_batch,
            cudaMemcpyHostToDevice
        );
    }

    /* 核函数执行 */
    {
        CUDATimer timer_compute("Kernel Execution", ENABLE_CUDA_TIMING);
        compute_square_sum<<<gridDim, blockDim, n_batch * sizeof(float)>>>(
            vector_data,
            vector_square_sum, 
            n_batch, n_dim
        );

        vector_normalize<<<gridDim, n_dim>>>(
            vector_data, 
            vector_square_sum, 
            n_batch, n_dim
        );
        
        // 注意：这里没有显式同步，同步在CUDATimer析构函数中的cudaEventSynchronize(stop_event_)完成
    }

    /* 数据传输 GPU->CPU */
    { 
        CUDATimer timer_trans2("D2H Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemcpy2D(
            vector_list_cpu[0],
            n_dim * sizeof(float),
            vector_data,
            n_dim * sizeof(float),
            n_dim * sizeof(float),
            n_batch,
            cudaMemcpyDeviceToHost
        );
    }

    /* GPU内存释放 */
    {
        CUDATimer timer_manage2("GPU Memory Free", ENABLE_CUDA_TIMING, false);
        cudaFree(vector_data);
        cudaFree(vector_square_sum);   
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
void VectorNormalizer::normalize_async(float*** vector_lists_cpu, int n_lists, int n_batch, int n_dim) {
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
            cudaStreamCreate(&streams[i]);
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
            compute_square_sum<<<gridDim, blockDim, n_dim * sizeof(float), streams[stream_id]>>>(
                d_vector_data[stream_id],
                d_vector_square_sum[stream_id],
                n_batch, n_dim
            );

            // c. 归一化 (依赖平方和计算完成，但在同一个流中，所以是顺序的)
            vector_normalize<<<gridDim, blockDim, 0, streams[stream_id]>>>(
                d_vector_data[stream_id],
                d_vector_square_sum[stream_id],
                n_batch, n_dim
            );
            // 记录计算完成事件
            cudaEventRecord(compute_events[stream_id], streams[stream_id]);

            // d. 异步D2H拷贝 (需要等待归一化计算完成)
            // 但D2H在同一个流中，所以会自动等待前面的内核完成
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

// 获取最近一次归一化的模长
float VectorNormalizer::last_norm() const {
    return norm_;
}

CosineDistanceOp::CosineDistanceOp(int n)
    : n_(n), blockSize_(256), gridSize_((n + blockSize_ - 1) / blockSize_) {
    cudaMalloc((void**)&d_a_, n_ * sizeof(float));
    cudaMalloc((void**)&d_b_, n_ * sizeof(float));
    cudaMalloc((void**)&d_dot_, n_ * sizeof(float));
    cudaMalloc((void**)&d_sq_a_, n_ * sizeof(float));
    cudaMalloc((void**)&d_sq_b_, n_ * sizeof(float));
    cudaMalloc((void**)&d_dot_sum_, gridSize_ * sizeof(float));
    cudaMalloc((void**)&d_sq_a_sum_, gridSize_ * sizeof(float));
    cudaMalloc((void**)&d_sq_b_sum_, gridSize_ * sizeof(float));
    h_dot_sum_ = new float[gridSize_];
    h_sq_a_sum_ = new float[gridSize_];
    h_sq_b_sum_ = new float[gridSize_];
}

CosineDistanceOp::~CosineDistanceOp() {
    cudaFree(d_a_);
    cudaFree(d_b_);
    cudaFree(d_dot_);
    cudaFree(d_sq_a_);
    cudaFree(d_sq_b_);
    cudaFree(d_dot_sum_);
    cudaFree(d_sq_a_sum_);
    cudaFree(d_sq_b_sum_);
    delete[] h_dot_sum_;
    delete[] h_sq_a_sum_;
    delete[] h_sq_b_sum_;
}

float CosineDistanceOp::compute(const float* a, const float* b, const int n) {
    // 复制数据到设备
    cudaMemcpy(d_a_, a, n_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_, b, n_ * sizeof(float), cudaMemcpyHostToDevice);

    // 计算点积
    dotProductKernel<<<gridSize_, blockSize_>>>(d_a_, d_b_, d_dot_, n_);
    
    // 计算向量a的平方
    squareKernel<<<gridSize_, blockSize_>>>(d_a_, d_sq_a_, n_);
    
    // 计算向量b的平方
    squareKernel<<<gridSize_, blockSize_>>>(d_b_, d_sq_b_, n_);

    // 规约求和
    reduceSumKernel<<<gridSize_, blockSize_, blockSize_ * sizeof(float)>>>(d_dot_, d_dot_sum_, n_);
    reduceSumKernel<<<gridSize_, blockSize_, blockSize_ * sizeof(float)>>>(d_sq_a_, d_sq_a_sum_, n_);
    reduceSumKernel<<<gridSize_, blockSize_, blockSize_ * sizeof(float)>>>(d_sq_b_, d_sq_b_sum_, n_);

    // 复制结果到主机
    cudaMemcpy(h_dot_sum_, d_dot_sum_, gridSize_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sq_a_sum_, d_sq_a_sum_, gridSize_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sq_b_sum_, d_sq_b_sum_, gridSize_ * sizeof(float), cudaMemcpyDeviceToHost);

    // 计算最终结果
    float dot_sum = 0.0f, sq_a_sum = 0.0f, sq_b_sum = 0.0f;
    for (int i = 0; i < gridSize_; ++i) {
        dot_sum += h_dot_sum_[i];
        sq_a_sum += h_sq_a_sum_[i];
        sq_b_sum += h_sq_b_sum_[i];
    }

    float norm_a = sqrtf(sq_a_sum);
    float norm_b = sqrtf(sq_b_sum);
    
    // 检查零向量情况
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 1.0f; // 如果任一向量为零向量，距离为1
    }
    
    float cosine_similarity = dot_sum / (norm_a * norm_b);
    
    // 返回余弦距离（1 - 余弦相似度）
    return 1.0f - cosine_similarity;
}

// CUDA核函数：计算两个向量的点积
__global__ void dotProductKernel(const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

// CUDA核函数：计算向量的平方
__global__ void squareKernel(const float* vec, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = vec[idx] * vec[idx];
    }
}

// CUDA规约核函数：块内求和
__global__ void reduceSumKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

float L2DistanceOp::operator()(const float* d_A, const float* d_B, int n) {
    float* d_result;
    cudaMalloc(&d_result, n * sizeof(float));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    l2_distance_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_result, n);

    thrust::device_ptr<float> thrust_ptr(d_result);
    float sum = thrust::reduce(thrust_ptr, thrust_ptr + n);

    float l2_distance = sqrtf(sum);

    cudaFree(d_result);
    return l2_distance;
}

// CUDA Kernel 计算平方差
__global__ void l2_distance_kernel(const float* A, const float* B, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = A[idx] - B[idx];
        result[idx] = diff * diff;
    }
}

