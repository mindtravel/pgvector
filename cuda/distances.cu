#include "distances.h"
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

// CUDA核函数：计算模长
__global__ void compute_norm(float *data, float *norm, int n) {
    __shared__ float shared_mem[256];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.0f;
    if (idx < n) sum = data[idx] * data[idx];
    shared_mem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_mem[tid] += shared_mem[tid + s];
        __syncthreads();
    }
    if (tid == 0) norm[blockIdx.x] = shared_mem[0];
}

// CUDA核函数：归一化
__global__ void vector_normalize(float *data, float norm, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) data[idx] /= norm;
}

// 构造与析构
VectorNormalizer::VectorNormalizer() : norm_(0.0f) {}
VectorNormalizer::~VectorNormalizer() {}

// 归一化接口
void VectorNormalizer::normalize(float* data, int n) {
    assert(data != nullptr && n > 0);

    float *d_data, *d_norm;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaMalloc((void**)&d_data, n * sizeof(float));
    cudaMalloc((void**)&d_norm, blocks * sizeof(float));
    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);

    compute_norm<<<blocks, threads>>>(d_data, d_norm, n);

    float* partial_norms = new float[blocks];
    cudaMemcpy(partial_norms, d_norm, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float total_norm = 0.0f;
    for (int i = 0; i < blocks; ++i) total_norm += partial_norms[i];
    norm_ = sqrt(total_norm);
    delete[] partial_norms;

    vector_normalize<<<blocks, threads>>>(d_data, norm_, n);

    cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_norm);
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

float CosineDistanceOp::compute(const float* a, const float* b) {
    cudaMemcpy(d_a_, a, n_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_, b, n_ * sizeof(float), cudaMemcpyHostToDevice);

    dotProductKernel<<<gridSize_, blockSize_>>>(d_a_, d_b_, d_dot_, n_);
    squareKernel<<<gridSize_, blockSize_>>>(d_a_, d_sq_a_, n_);
    squareKernel<<<gridSize_, blockSize_>>>(d_b_, d_sq_b_, n_);

    size_t sharedMemSize = blockSize_ * sizeof(float);
    reduceSumKernel<<<gridSize_, blockSize_, sharedMemSize>>>(d_dot_, d_dot_sum_, n_);
    reduceSumKernel<<<gridSize_, blockSize_, sharedMemSize>>>(d_sq_a_, d_sq_a_sum_, n_);
    reduceSumKernel<<<gridSize_, blockSize_, sharedMemSize>>>(d_sq_b_, d_sq_b_sum_, n_);

    cudaMemcpy(h_dot_sum_, d_dot_sum_, gridSize_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sq_a_sum_, d_sq_a_sum_, gridSize_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sq_b_sum_, d_sq_b_sum_, gridSize_ * sizeof(float), cudaMemcpyDeviceToHost);

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < gridSize_; ++i) {
        dot += h_dot_sum_[i];
        norm_a += h_sq_a_sum_[i];
        norm_b += h_sq_b_sum_[i];
    }

    float similarity = dot / (sqrtf(norm_a) * sqrtf(norm_b));
    float distance = 1.0f - similarity;
    return distance;
}

// CUDA核函数：计算两个向量的点积
__global__ void CosineDistanceOp::dotProductKernel(const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

// CUDA核函数：计算向量的平方
__global__ void CosineDistanceOp::squareKernel(const float* vec, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = vec[idx] * vec[idx];
    }
}

// CUDA规约核函数：块内求和
__global__ void CosineDistanceOp::reduceSumKernel(const float* input, float* output, int n) {
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
__global__ void L2DistanceOp::l2_distance_kernel(const float* A, const float* B, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = A[idx] - B[idx];
        result[idx] = diff * diff;
    }
}

