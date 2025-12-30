#include "kmeans.cuh"
#include "../utils.cuh"
#include "../l2norm/l2norm.cuh"
#include "../cudatimer.h"
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <../unit_tests/common/test_utils.cuh>

#define ENABLE_CUDA_TIMING 1 /*是否启用CUDATimer计时*/

static inline void cublas_check(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublas error %d at %s\n", (int)st, msg);
        std::abort();
    }
}

// ============================================================
// GPU KMeans Minibatch Implementation (GEMM-optimized)
// ============================================================

void gpu_kmeans_minibatch(
    const KMeansCase& cfg,
    const float* h_data,             // [n, dim] row-major，主机端数据
    int* d_assign,                   // [n] (可选，minibatch可能不更新所有assign)
    float* d_centroids,              // [k, dim] (in/out)，常驻显存
    float* h_objective                // sum dist2 (可选)
) {
    CUDATimer timer_total("gpu_kmeans_minibatch: Total Time", ENABLE_CUDA_TIMING);
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;

    // ====== Minibatch 参数 ======
    const int MINIBATCH_SIZE = 1 << 15;  // 每次迭代使用的minibatch大小（32K）
    // 注意：学习率是自适应的，基于每个centroid被分配到的点的累计数量
    const int Ktile = 4096;              // 每次GEMM处理的centroid数量

    // ====== buffer ======
    float* d_cnorm2 = nullptr;     // [k]

    // Minibatch缓冲区：xnorm2[M], best_dist2[M], best_idx[M], dot[M*Ktile]
    const int M = MINIBATCH_SIZE;
    float* d_xnorm2 = nullptr;
    float* d_best_dist2 = nullptr;
    int*   d_best_idx = nullptr;
    float* d_dot = nullptr;        // [M, Ktile] 但实际存的是col-major [curK, curM]
    
    // 双缓冲：两个minibatch数据缓冲区交替使用
    float* d_minibatch_data[2] = {nullptr, nullptr};  // [M, dim]
    cudaStream_t stream[2];                           // 两个CUDA流用于异步传输
    // 修复：使用两个事件数组，明确语义
    cudaEvent_t evt_h2d_done[2];   // H2D传输完成事件
    cudaEvent_t evt_compute_done[2];  // 计算完成事件

    // d_centroids 是输入参数，不需要在这里分配
    cudaMalloc(&d_cnorm2,    sizeof(float) * (size_t)k);

    cudaMalloc(&d_xnorm2,    sizeof(float) * (size_t)M);
    cudaMalloc(&d_best_dist2,sizeof(float) * (size_t)M);
    cudaMalloc(&d_best_idx,  sizeof(int)   * (size_t)M);
    cudaMalloc(&d_dot,       sizeof(float) * (size_t)M * (size_t)Ktile);
    
    // 分配双缓冲minibatch数据缓冲区
    cudaMalloc(&d_minibatch_data[0], sizeof(float) * (size_t)M * dim);
    cudaMalloc(&d_minibatch_data[1], sizeof(float) * (size_t)M * dim);
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    // 创建两个事件数组
    cudaEventCreate(&evt_h2d_done[0]);
    cudaEventCreate(&evt_h2d_done[1]);
    cudaEventCreate(&evt_compute_done[0]);
    cudaEventCreate(&evt_compute_done[1]);
    
    // Minibatch累加器：每个centroid的累加和计数（用于更新）
    float* d_minibatch_accum = nullptr;  // [k, dim]
    int*   d_minibatch_counts = nullptr; // [k]
    cudaMalloc(&d_minibatch_accum, sizeof(float) * (size_t)k * dim);
    cudaMalloc(&d_minibatch_counts, sizeof(int) * (size_t)k);
    
    // GPU上维护累计分配次数（用于自适应学习率）
    int* d_total_counts = nullptr;  // [k]
    cudaMalloc(&d_total_counts, sizeof(int) * (size_t)k);
    cudaMemset(d_total_counts, 0, sizeof(int) * (size_t)k);
    
    // GPU 上的 objective 累加器（单个 float）
    float* d_objective_sum = nullptr;
    if (h_objective) {
        cudaMalloc(&d_objective_sum, sizeof(float));
    }
    
    // 随机数生成器（用于选择minibatch）
    std::mt19937 rng(cfg.seed);


    // ====== cuBLAS handle ======
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    cublas_check(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "setMathMode");

    // ====== 常量 ======
    const float alpha = 1.f;
    const float beta  = 0.f;

    // ====== Minibatch迭代（使用双缓冲） ======
    int cur_buf = 0;
    
    // 如果n < M，使用所有数据作为minibatch
    const int actual_minibatch_size = (n < M) ? n : M;
    
    // 预加载第一个minibatch
    int first_minibatch_start = 0;
    if (n > M) {
        std::uniform_int_distribution<int> dist_idx(0, n - M);
        first_minibatch_start = dist_idx(rng);
        if (first_minibatch_start + M > n) {
            first_minibatch_start = n - M;
        }
    }
    // 当 n <= M 时，使用所有数据，不需要随机选择
    cudaMemcpyAsync(d_minibatch_data[cur_buf], h_data + (size_t)first_minibatch_start * dim,
                   sizeof(float) * (size_t)actual_minibatch_size * dim,
                   cudaMemcpyHostToDevice, stream[cur_buf]);
    cudaEventRecord(evt_h2d_done[cur_buf], stream[cur_buf]);
    
    for (int it = 0; it < cfg.minibatch_iters; ++it) {
        CUDATimer timer_iter("gpu_kmeans_minibatch: Iteration " + std::to_string(it), ENABLE_CUDA_TIMING);
        
        // 等待当前minibatch数据传输完成
        cudaStreamWaitEvent(stream[cur_buf], evt_h2d_done[cur_buf], 0);
        
        // 异步预加载下一个minibatch（如果还有）
        int next_buf = 1 - cur_buf;
        if (it + 1 < cfg.minibatch_iters && n > M) {
            std::uniform_int_distribution<int> dist_idx(0, n - M);
            int next_minibatch_start = dist_idx(rng);
            if (next_minibatch_start + M > n) {
                next_minibatch_start = n - M;
            }
            // 修复：等待next_buf的计算完成，避免覆盖正在使用的缓冲区
            cudaStreamWaitEvent(stream[next_buf], evt_compute_done[next_buf], 0);
            cudaMemcpyAsync(d_minibatch_data[next_buf], h_data + (size_t)next_minibatch_start * dim,
                           sizeof(float) * (size_t)actual_minibatch_size * dim,
                           cudaMemcpyHostToDevice, stream[next_buf]);
            cudaEventRecord(evt_h2d_done[next_buf], stream[next_buf]);
        }
        
        float* d_data_cur = d_minibatch_data[cur_buf];
        
        {
            CUDATimer timer_init("  Iter " + std::to_string(it) + ": Initialize (compute centroid norms, clear accum/counts)", ENABLE_CUDA_TIMING);
            // 修复：在stream[cur_buf]上计算centroid范数，确保依赖正确
            compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO, stream[cur_buf]);
            
            // 清零minibatch累加器（使用当前流）
            cudaMemsetAsync(d_minibatch_accum, 0, sizeof(float) * (size_t)k * dim, stream[cur_buf]);
            cudaMemsetAsync(d_minibatch_counts, 0, sizeof(int) * (size_t)k, stream[cur_buf]);
            
            // 清零objective累加器（如果启用）
            if (h_objective && d_objective_sum) {
                cudaMemsetAsync(d_objective_sum, 0, sizeof(float), stream[cur_buf]);
            }
        }
        
        {
            CUDATimer timer_minibatch("  Iter " + std::to_string(it) + ": Process Minibatch", ENABLE_CUDA_TIMING);
            // 计算minibatch的xnorm2（在当前流上）
            compute_l2_squared_gpu(d_data_cur, d_xnorm2, actual_minibatch_size, dim, L2NORM_AUTO, stream[cur_buf]);
            
            // 初始化best_dist2和best_idx（在当前流上）
            {
                int threads = 256;
                int blocks = (actual_minibatch_size + threads - 1) / threads;
                kernel_init_best<<<blocks, threads, 0, stream[cur_buf]>>>(d_best_dist2, d_best_idx, actual_minibatch_size);
            }
            
            // ====== centroid分块处理 ======
            for (int cbase = 0; cbase < k; cbase += Ktile) {
                int curK = std::min(Ktile, k - cbase);
                
                const float* A = d_centroids + (size_t)cbase * dim;
                const float* Bm = d_data_cur;
                float* Cc = d_dot;
                
                // 设置cuBLAS流
                cublas_check(cublasSetStream(handle, stream[cur_buf]), "cublasSetStream");
                
                // GEMM: dotT[curK, actual_M] = centroids[curK, dim]^T * minibatch[actual_M, dim]
                cublas_check(
                    cublasSgemm(
                        handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        curK, actual_minibatch_size, dim,
                        &alpha,
                        A, dim,
                        Bm, dim,
                        &beta,
                        Cc, curK
                    ),
                    "cublasSgemm"
                );
                
                // 更新best（在当前流上）
                {
                    int threads = 256;
                    int blocks = (actual_minibatch_size + threads - 1) / threads;
                    kernel_update_best_from_dotT<<<blocks, threads, 0, stream[cur_buf]>>>(
                        d_dot, d_xnorm2, d_cnorm2,
                        actual_minibatch_size, curK, cbase,
                        d_best_idx, d_best_dist2
                    );
                }
            }
            
            // 累加minibatch的assign结果（在当前流上）
            // 优化：使用更大的 block size 以提高占用率
            {
                int threads = 512;  // 增加到 512 以提高占用率
                int blocks = (actual_minibatch_size + threads - 1) / threads;
                blocks = std::min(blocks, 65535);
                kernel_accum_from_assign<<<blocks, threads, 0, stream[cur_buf]>>>(
                    d_data_cur, actual_minibatch_size, dim,
                    d_best_idx, d_minibatch_accum, d_minibatch_counts
                );
            }
        }
        
        {
            CUDATimer timer_update("  Iter " + std::to_string(it) + ": Update Centroids (Minibatch)", ENABLE_CUDA_TIMING);
            // 修复：在GPU上更新centroids，避免host循环和多次D2H/H2D拷贝
            // Minibatch更新聚类中心：使用自适应学习率
            // 标准minibatch k-means更新公式：
            // centroid_new = centroid_old + lr * (centroid_batch - centroid_old)
            // 其中 lr = 1 / (total_count + 1)，total_count是累计分配次数
            // 等价于：centroid_new = (1 - lr) * centroid_old + lr * centroid_batch
            {
                // 使用GPU kernel更新centroids，完全避免host循环
                kernel_update_centroids_minibatch<<<k, 256, 0, stream[cur_buf]>>>(
                    d_centroids, d_minibatch_accum, d_minibatch_counts, d_total_counts, k, dim);
            }
        }
        
        // 记录当前minibatch计算完成事件（用于下一个minibatch的依赖）
        cudaEventRecord(evt_compute_done[cur_buf], stream[cur_buf]);
        
        // 切换到下一个缓冲区（用于下一次迭代）
        if (it + 1 < cfg.minibatch_iters) {
            cur_buf = next_buf;
        }
    }
    
    // // 迭代结束后，计算所有数据点的objective（用于与CPU结果对比）
    // if (h_objective && d_objective_sum) {
    //     CUDATimer timer_objective("gpu_kmeans_minibatch: Compute Objective (All Data)", ENABLE_CUDA_TIMING);
    //     cudaMemset(d_objective_sum, 0, sizeof(float));
        
    //     // 计算centroid范数
    //     compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO, stream[0]);
        
    //     // 分块处理所有数据点计算objective
    //     const int B = 1 << 15;  // 32K per batch
    //     for (int base = 0; base < n; base += B) {
    //         int curB = std::min(B, n - base);
            
    //         // 传输当前batch数据
    //         cudaMemcpyAsync(d_minibatch_data[0], h_data + (size_t)base * dim,
    //                        sizeof(float) * (size_t)curB * dim,
    //                        cudaMemcpyHostToDevice, stream[0]);
            
    //         // 计算xnorm2
    //         compute_l2_squared_gpu(d_minibatch_data[0], d_xnorm2, curB, dim, L2NORM_AUTO, stream[0]);
            
    //         // 初始化best
    //         {
    //             int threads = 256;
    //             int blocks = (curB + threads - 1) / threads;
    //             kernel_init_best<<<blocks, threads, 0, stream[0]>>>(d_best_dist2, d_best_idx, curB);
    //         }
            
    //         // 计算到所有centroid的距离
    //         for (int cbase = 0; cbase < k; cbase += Ktile) {
    //             int curK = std::min(Ktile, k - cbase);
                
    //             const float* A = d_centroids + (size_t)cbase * dim;
    //             const float* Bm = d_minibatch_data[0];
    //             float* Cc = d_dot;
                
    //             cublas_check(cublasSetStream(handle, stream[0]), "cublasSetStream");
                
    //             cublas_check(
    //                 cublasSgemm(
    //                     handle,
    //                     CUBLAS_OP_T, CUBLAS_OP_N,
    //                     curK, curB, dim,
    //                     &alpha,
    //                     A, dim,
    //                     Bm, dim,
    //                     &beta,
    //                     Cc, curK
    //                 ),
    //                 "cublasSgemm"
    //             );
                
    //             {
    //                 int threads = 256;
    //                 int blocks = (curB + threads - 1) / threads;
    //                 kernel_update_best_from_dotT<<<blocks, threads, 0, stream[0]>>>(
    //                     d_dot, d_xnorm2, d_cnorm2,
    //                     curB, curK, cbase,
    //                     d_best_idx, d_best_dist2
    //                 );
    //             }
    //         }
            
    //         // 累加objective
    //         {
    //             int reduce_threads = 256;
    //             int reduce_blocks = (curB + reduce_threads - 1) / reduce_threads;
    //             int reduce_shmem = reduce_threads * sizeof(float);
    //             kernel_reduce_sum<<<reduce_blocks, reduce_threads, reduce_shmem, stream[0]>>>(
    //                 d_best_dist2, d_objective_sum, curB);
    //         }
    //     }
        
    //     // 等待所有计算完成
    //     cudaStreamSynchronize(stream[0]);
        
    //     // 读取objective
    //     float obj_val = 0.0f;
    //     cudaMemcpy(&obj_val, d_objective_sum, sizeof(float), cudaMemcpyDeviceToHost);
    //     *h_objective = obj_val;
    //     COUT_ENDL("minibatch final obj=", obj_val);
    // }

    // ====== 迭代结束后，计算所有点的最终分配 ======
    // Minibatch算法只处理了部分点，需要在最后对所有点进行分配计算
    if (d_assign) {
        CUDATimer timer_final_assign("gpu_kmeans_minibatch: Final Assignment (All Points)", ENABLE_CUDA_TIMING);
        
        // 计算centroid范数（用于距离计算）
        compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO, stream[0]);
        cudaStreamSynchronize(stream[0]);
        
        // 分块处理所有数据点，计算最终分配
        const int B = MINIBATCH_SIZE;  // 使用相同的batch大小
        for (int base = 0; base < n; base += B) {
            int curB = std::min(B, n - base);
            
            // 传输当前batch数据
            cudaMemcpyAsync(d_minibatch_data[0], h_data + (size_t)base * dim,
                           sizeof(float) * (size_t)curB * dim,
                           cudaMemcpyHostToDevice, stream[0]);
            
            // 计算xnorm2
            compute_l2_squared_gpu(d_minibatch_data[0], d_xnorm2, curB, dim, L2NORM_AUTO, stream[0]);
            
            // 初始化best
            {
                int threads = 256;
                int blocks = (curB + threads - 1) / threads;
                kernel_init_best<<<blocks, threads, 0, stream[0]>>>(d_best_dist2, d_best_idx, curB);
            }
            
            // 计算到所有centroid的距离（分块处理centroids）
            for (int cbase = 0; cbase < k; cbase += Ktile) {
                int curK = std::min(Ktile, k - cbase);
                
                const float* A = d_centroids + (size_t)cbase * dim;
                const float* Bm = d_minibatch_data[0];
                float* Cc = d_dot;
                
                cublas_check(cublasSetStream(handle, stream[0]), "cublasSetStream");
                
                cublas_check(
                    cublasSgemm(
                        handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        curK, curB, dim,
                        &alpha,
                        A, dim,
                        Bm, dim,
                        &beta,
                        Cc, curK
                    ),
                    "cublasSgemm"
                );
                
                // 更新best
                {
                    int threads = 256;
                    int blocks = (curB + threads - 1) / threads;
                    kernel_update_best_from_dotT<<<blocks, threads, 0, stream[0]>>>(
                        d_dot, d_xnorm2, d_cnorm2,
                        curB, curK, cbase,
                        d_best_idx, d_best_dist2
                    );
                }
            }
            
            // 写回assign（所有点的最终分配）
            cudaMemcpyAsync(d_assign + base, d_best_idx, sizeof(int) * (size_t)curB, 
                           cudaMemcpyDeviceToDevice, stream[0]);
        }
        
        // 等待所有分配计算完成
        cudaStreamSynchronize(stream[0]);
    }

    cublasDestroy(handle);

    // 释放双缓冲
    cudaFree(d_minibatch_data[0]);
    cudaFree(d_minibatch_data[1]);
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaEventDestroy(evt_h2d_done[0]);
    cudaEventDestroy(evt_h2d_done[1]);
    cudaEventDestroy(evt_compute_done[0]);
    cudaEventDestroy(evt_compute_done[1]);
    
    if (d_objective_sum) {
        cudaFree(d_objective_sum);
    }

    // d_centroids 是输入参数，不需要在这里释放
    cudaFree(d_cnorm2);
    cudaFree(d_minibatch_accum);
    cudaFree(d_minibatch_counts);
    cudaFree(d_total_counts);
    cudaFree(d_xnorm2);
    cudaFree(d_best_dist2);
    cudaFree(d_best_idx);
    cudaFree(d_dot);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_kmeans_minibatch: %s\n", cudaGetErrorString(err));
    }
}