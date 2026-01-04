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

#define ENABLE_CUDA_TIMING 0 /*是否启用CUDATimer计时*/

static inline void cublas_check(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublas error %d at %s\n", (int)st, msg);
        std::abort();
    }
}

// ============================================================
// GPU KMeans Runner Implementation (GEMM-optimized)
// ============================================================

void gpu_kmeans_lloyd(
    const KMeansCase& cfg,
    const float* h_data,             // [n, dim] row-major，主机端数据
    int* d_assign,                   // [n]
    float* d_centroids,              // [k, dim] (in/out)，常驻显存
    float* h_objective                // sum dist2
) {
    CUDATimer timer_total("gpu_kmeans_lloyd: Total Time", ENABLE_CUDA_TIMING);
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;

    // ====== 可调参数：batch & ktile ======
    // 减小 B 和 Ktile 以控制 d_dot 显存占用
    // 目标：d_dot = B * Ktile <= 1GB (约 256M floats)
    const int Ktile = 4096;  // 每次GEMM处理的centroid数量（从8192减小到4096）
    const int B = 1 << 15;   // 每次处理的点数量（32K，从256K减小到32K）
    // d_dot = 32K * 4K = 128M floats ≈ 512MB（可接受）

    // ====== buffer ======
    float* d_accum = nullptr;      // [k, dim]
    int*   d_counts = nullptr;     // [k]
    float* d_cnorm2 = nullptr;     // [k]

    // 双缓冲：两个GPU缓冲区交替使用
    float* d_data_buf[2] = {nullptr, nullptr};  // 双缓冲数据缓冲区
    cudaStream_t stream[2];                      // 两个CUDA流用于异步传输
    cudaEvent_t event[2];                        // 事件用于依赖管理，替代同步
    
    // StreamEnv：为每个流分配独立的中间 buffer，避免竞态条件
    StreamEnv stream_env[2];

    // d_centroids 是输入参数，不需要在这里分配
    cudaMalloc(&d_accum,     sizeof(float) * (size_t)k * dim);
    cudaMalloc(&d_counts,    sizeof(int)   * (size_t)k);
    cudaMalloc(&d_cnorm2,    sizeof(float) * (size_t)k);

    // 为两个流分别分配独立的中间 buffer
    stream_env[0].allocate(B, Ktile);
    stream_env[1].allocate(B, Ktile);

    // 分配双缓冲
    cudaMalloc(&d_data_buf[0], sizeof(float) * (size_t)B * dim);
    cudaMalloc(&d_data_buf[1], sizeof(float) * (size_t)B * dim);
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    cudaEventCreate(&event[0]);
    cudaEventCreate(&event[1]);
    
    // GPU 上的 objective 累加器（单个 float）
    float* d_objective_sum = nullptr;
    if (h_objective) {
        cudaMalloc(&d_objective_sum, sizeof(float));
    }


    // ====== cuBLAS handle ======
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    cublas_check(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "setMathMode");

    // ====== 常量 ======
    const float alpha = 1.f;
    const float beta  = 0.f;

    // 当前使用的缓冲区索引（0或1）
    int cur_buf = 0;

    // ====== iteration ======
    for (int it = 0; it < cfg.iters; ++it) {
        CUDATimer timer_iter("gpu_kmeans_lloyd: Iteration " + std::to_string(it), ENABLE_CUDA_TIMING);
        // clear accum & counts
        // 这些操作需要在所有batch开始前完成
        {
            CUDATimer timer_init("  Iter " + std::to_string(it) + ": Initialize (clear accum/counts, compute centroid norms)", ENABLE_CUDA_TIMING);
            cudaMemsetAsync(d_accum, 0, sizeof(float) * (size_t)k * dim, stream[0]);
            cudaMemsetAsync(d_counts, 0, sizeof(int) * (size_t)k, stream[0]);
            // 绑定到 stream[0]，确保在所有batch计算前完成
            compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO, stream[0]);
        }

        // ====== 分块处理数据点（使用双缓冲） ======
        cur_buf = 0;
        
        // 清零 objective 累加器（如果启用，使用异步版本）
        if (h_objective && d_objective_sum) {
            cudaMemsetAsync(d_objective_sum, 0, sizeof(float), stream[0]);
        }
        
        // 记录 stream[0] 上的初始化操作完成事件（accum清零、centroid范数计算）
        // 确保后续batch计算时 d_cnorm2 和 d_accum 已准备好
        cudaEvent_t init_event;
        cudaEventCreate(&init_event);
        cudaEventRecord(init_event, stream[0]);

        // 预加载第一个batch
        int first_base = 0;
        int first_curB = std::min(B, n - first_base);
        // 等待初始化操作完成，确保 d_cnorm2 和 d_accum 已准备好
        cudaStreamWaitEvent(stream[cur_buf], init_event, 0);
        cudaMemcpyAsync(d_data_buf[cur_buf], h_data + (size_t)first_base * dim,
                       sizeof(float) * (size_t)first_curB * dim,
                       cudaMemcpyHostToDevice, stream[cur_buf]);
        // 记录事件，用于后续依赖
        cudaEventRecord(event[cur_buf], stream[cur_buf]);

        {
            CUDATimer timer_batch("  Iter " + std::to_string(it) + ": Process All Batches", ENABLE_CUDA_TIMING);
        for (int base = 0; base < n; base += B) {
            int curB = std::min(B, n - base);
            float* d_data_cur = d_data_buf[cur_buf];
            
            // 等待当前batch的数据传输完成（使用事件依赖）
            // 第一次迭代：等待预加载完成；后续迭代：等待上一个batch的计算完成
            cudaStreamWaitEvent(stream[cur_buf], event[cur_buf], 0);

            // 异步预加载下一个batch（如果还有）
            // 注意：这里先启动预加载，但会在计算完成后才真正传输（通过事件依赖）
            int next_base = base + B;
            int next_buf = 1 - cur_buf;

            // 使用当前流对应的 StreamEnv buffer，避免竞态条件
            StreamEnv& cur_env = stream_env[cur_buf];
            
            // xnorm2 for this batch（在当前流的流上执行）
            compute_l2_squared_gpu(d_data_cur, cur_env.d_xnorm2, curB, dim, L2NORM_AUTO, stream[cur_buf]);

            // init best_dist2 = +inf, best_idx = 0（在当前流的流上执行）
            {
                int threads = 256;
                int blocks = (curB + threads - 1) / threads;
                kernel_init_best<<<blocks, threads, 0, stream[cur_buf]>>>(cur_env.d_best_dist2, cur_env.d_best_idx, curB);
            }

            // ====== centroid 分块 Ktile ======
            for (int cbase = 0; cbase < k; cbase += Ktile) {
                int curK = std::min(Ktile, k - cbase);

                const float* A = d_centroids + (size_t)cbase * dim;
                const float* Bm = d_data_cur;
                float* Cc = cur_env.d_dot;

                // 设置cuBLAS流
                cublas_check(cublasSetStream(handle, stream[cur_buf]), "cublasSetStream");

                // Treat row-major C[curK,dim] as col-major Ccm[dim,curK]
                // Treat row-major X[curB,dim] as col-major Xcm[dim,curB]
                // Compute dotT[curK,curB] (col-major) = Ccm^T[curK,dim] * Xcm[dim,curB]
                cublas_check(
                    cublasSgemm(
                        handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,     // A^T * B
                        curK, curB, dim,              // m=curK, n=curB, k=dim
                        &alpha,
                        A, dim,                       // A is Ccm with shape [dim,curK], lda=dim
                        Bm, dim,                      // B is Xcm with shape [dim,curB], ldb=dim
                        &beta,
                        Cc, curK                     // C is [curK,curB] col-major, ldc=curK
                    ),
                    "cublasSgemm(Ccm^T * Xcm)"
                );


                // 更新best（在当前流的流上执行）
                {
                    int threads = 256;
                    int blocks = (curB + threads - 1) / threads;
                    kernel_update_best_from_dotT<<<blocks, threads, 0, stream[cur_buf]>>>(
                        cur_env.d_dot, cur_env.d_xnorm2, d_cnorm2,
                        curB, curK, cbase,
                        cur_env.d_best_idx, cur_env.d_best_dist2
                    );
                }
            }

            // 写回assign（全局，使用异步拷贝）
            cudaMemcpyAsync(d_assign + base, cur_env.d_best_idx, sizeof(int) * (size_t)curB, 
                           cudaMemcpyDeviceToDevice, stream[cur_buf]);

            // accum + counts（在当前流的流上执行）
            // 优化：使用更大的 block size 以提高占用率，减少 kernel 启动开销
            {
                int threads = 512;  // 增加到 512 以提高占用率
                int blocks = (curB + threads - 1) / threads;
                // 限制最大 block 数，避免过度启动
                blocks = std::min(blocks, 65535);
                kernel_accum_from_assign<<<blocks, threads, 0, stream[cur_buf]>>>(
                    d_data_cur, curB, dim,
                    cur_env.d_best_idx, d_accum, d_counts
                );
            }

            // objective：在 GPU 上做 reduce，避免回传到 host
            if (h_objective && d_objective_sum) {
                int reduce_threads = 256;
                int reduce_blocks = (curB + reduce_threads - 1) / reduce_threads;
                int reduce_shmem = reduce_threads * sizeof(float);
                kernel_reduce_sum<<<reduce_blocks, reduce_threads, reduce_shmem, stream[cur_buf]>>>(
                    cur_env.d_best_dist2, d_objective_sum, curB);
            }

            // 记录当前batch计算完成事件（用于下一个batch的依赖）
            cudaEventRecord(event[cur_buf], stream[cur_buf]);
            
            // 现在可以安全地预加载下一个batch（等待当前batch计算完成）
            if (next_base < n) {
                int next_curB = std::min(B, n - next_base);
                // 等待当前batch的计算完成，避免覆盖正在使用的缓冲区
                cudaStreamWaitEvent(stream[next_buf], event[cur_buf], 0);
                cudaMemcpyAsync(d_data_buf[next_buf], h_data + (size_t)next_base * dim,
                               sizeof(float) * (size_t)next_curB * dim,
                               cudaMemcpyHostToDevice, stream[next_buf]);
                // 记录传输完成事件（用于下一个迭代的依赖）
                cudaEventRecord(event[next_buf], stream[next_buf]);
            }
            
            // 切换到下一个缓冲区
            cur_buf = next_buf;
        }
        }

        // 等待所有batch完成（只同步必要的流）
        // 等待两个流都完成，确保所有batch的计算和accum都完成
        cudaStreamSynchronize(stream[0]);
        cudaStreamSynchronize(stream[1]);
        
        // update centroids: one block per centroid（使用 stream[0]）
        {
            CUDATimer timer_update("  Iter " + std::to_string(it) + ": Update Centroids", ENABLE_CUDA_TIMING);
        kernel_update_centroids<<<k, 256, 0, stream[0]>>>(d_centroids, d_accum, d_counts, k, dim);
        
        // 等待 centroid 更新完成（只同步 stream[0]）
        cudaStreamSynchronize(stream[0]);
        }

        // 从 GPU 读取 objective 值（只在迭代结束时读取一次）
        if (h_objective && d_objective_sum) {
            float obj_val = 0.0f;
            cudaMemcpy(&obj_val, d_objective_sum, sizeof(float), cudaMemcpyDeviceToHost);
            *h_objective = obj_val;
            printf("iter=%d, obj=%f\n", it, obj_val);
        }
        
        // 清理临时事件
        cudaEventDestroy(init_event);
    }

    // ====== Final Assignment Pass ======
    // 修复问题2：最后一次迭代的逻辑滞后
    // 循环结束后，d_centroids 已经更新，但 d_assign 还是基于旧位置计算的
    // 需要再次执行 assignment，确保 d_assign 严格对应最终的 d_centroids
    {
        CUDATimer timer_final("gpu_kmeans_lloyd: Final Assignment Pass", ENABLE_CUDA_TIMING);
        // 使用 stream[0] 和 stream_env[0] 执行最终的 assignment
        // 复用 d_data_buf[0] 作为数据缓冲区，避免循环内分配/释放显存
        perform_assignment_only(
            cfg, h_data, d_assign, d_centroids, d_cnorm2,
            stream_env[0], d_data_buf[0], stream[0], handle, B, Ktile, n, dim, k
        );
    }

    cublasDestroy(handle);

    // 释放双缓冲
    cudaFree(d_data_buf[0]);
    cudaFree(d_data_buf[1]);
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaEventDestroy(event[0]);
    cudaEventDestroy(event[1]);
    
    // 释放 StreamEnv buffer
    stream_env[0].free();
    stream_env[1].free();
    
    if (d_objective_sum) {
        cudaFree(d_objective_sum);
    }

    // d_centroids 是输入参数，不需要在这里释放
    cudaFree(d_accum);
    cudaFree(d_counts);
    cudaFree(d_cnorm2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_kmeans_lloyd: %s\n", cudaGetErrorString(err));
    }
}


// ============================================================
// CPU Initialization Functions Implementation
// ============================================================

__host__ void init_centroids_by_sampling(
    const KMeansCase& cfg,
    const float* data,        // [n, dim]
    float* out_centroids      // [k, dim]
) {
    std::mt19937 rng(cfg.seed);
    
    // 生成索引序列 [0, 1, 2, ..., n-1]
    std::vector<int> indices(cfg.n);
    for (int i = 0; i < cfg.n; ++i) {
        indices[i] = i;
    }
    
    // Fisher-Yates 洗牌算法：随机打乱前 k 个位置
    // 这样前 k 个索引就是随机选择的 k 个不同的点
    for (int i = 0; i < cfg.k && i < cfg.n; ++i) {
        std::uniform_int_distribution<int> dist(i, cfg.n - 1);
        int j = dist(rng);
        std::swap(indices[i], indices[j]);
    }
    
    // 复制前 k 个点作为聚类中心
    for (int c = 0; c < cfg.k; ++c) {
        int idx = indices[c];
        std::memcpy(out_centroids + (size_t)c * cfg.dim,
                    data + (size_t)idx * cfg.dim,
                    sizeof(float) * cfg.dim);
    }
}

