#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

// Normalization kernel
__global__ void normalize_kernel(float* data, int n, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float norm = 0.0f;
        for (int j = 0; j < d; ++j)
            norm += data[idx * d + j] * data[idx * d + j];
        norm = sqrtf(norm) + 1e-8f;
        for (int j = 0; j < d; ++j)
            data[idx * d + j] /= norm;
    }
}

// Cosine distance kernel
__global__ void cosine_distance_kernel(const float* query, const float* data, float* dist, int N1, int N2, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N1 && j < N2) {
        float dot = 0.0f;
        for (int k = 0; k < D; ++k)
            dot += query[i * D + k] * data[j * D + k];
        dist[i * N2 + j] = 1.0f - dot;
    }
}

// L2 distance kernel
__global__ void l2_distance_kernel(const float* query, const float* data, float* dist, int N1, int N2, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N1 && j < N2) {
        float sum = 0.0f;
        for (int k = 0; k < D; ++k) {
            float diff = query[i * D + k] - data[j * D + k];
            sum += diff * diff;
        }
        dist[i * N2 + j] = sum;
    }
}

// Top-k kernel
__global__ void topk_kernel(const float* dist, int* result, int N1, int N2, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N1) {
        for (int t = 0; t < k; ++t) {
            float min_val = FLT_MAX;
            int min_idx = -1;
            for (int j = 0; j < N2; ++j) {
                int already = 0;
                for (int s = 0; s < t; ++s)
                    if (result[i * k + s] == j) already = 1;
                if (!already && dist[i * N2 + j] < min_val) {
                    min_val = dist[i * N2 + j];
                    min_idx = j;
                }
            }
            result[i * k + t] = min_idx;
        }
    }
}

// Cosine distance search
extern "C"
void search_batch_cosine_distance_cuda(
    float* h_query, float* h_data,
    int N1, int N2, int D, int k,
    int* h_result
) {
    float *d_query, *d_data, *d_dist;
    int *d_result;
    cudaMalloc(&d_query, N1 * D * sizeof(float));
    cudaMalloc(&d_data, N2 * D * sizeof(float));
    cudaMalloc(&d_dist, N1 * N2 * sizeof(float));
    cudaMalloc(&d_result, N1 * k * sizeof(int));

    cudaMemcpy(d_query, h_query, N1 * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data, N2 * D * sizeof(float), cudaMemcpyHostToDevice);

    normalize_kernel<<<(N1+255)/256, 256>>>(d_query, N1, D);
    normalize_kernel<<<(N2+255)/256, 256>>>(d_data, N2, D);

    dim3 block(16, 16);
    dim3 grid((N1+15)/16, (N2+15)/16);
    cosine_distance_kernel<<<grid, block>>>(d_query, d_data, d_dist, N1, N2, D);

    topk_kernel<<<(N1+255)/256, 256>>>(d_dist, d_result, N1, N2, k);

    cudaMemcpy(h_result, d_result, N1 * k * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_query);
    cudaFree(d_data);
    cudaFree(d_dist);
    cudaFree(d_result);
}

// L2 distance search
extern "C"
void search_batch_l2_distance_cuda(
    float* h_query, float* h_data,
    int N1, int N2, int D, int k,
    int* h_result
) {
    float *d_query, *d_data, *d_dist;
    int *d_result;
    cudaMalloc(&d_query, N1 * D * sizeof(float));
    cudaMalloc(&d_data, N2 * D * sizeof(float));
    cudaMalloc(&d_dist, N1 * N2 * sizeof(float));
    cudaMalloc(&d_result, N1 * k * sizeof(int));

    cudaMemcpy(d_query, h_query, N1 * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data, N2 * D * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N1+15)/16, (N2+15)/16);
    l2_distance_kernel<<<grid, block>>>(d_query, d_data, d_dist, N1, N2, D);

    topk_kernel<<<(N1+255)/256, 256>>>(d_dist, d_result, N1, N2, k);

    cudaMemcpy(h_result, d_result, N1 * k * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_query);
    cudaFree(d_data);
    cudaFree(d_dist);
    cudaFree(d_result);
}