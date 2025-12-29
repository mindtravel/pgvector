#include "kmeans.cuh"
#include "../utils.cuh"
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// ============================================================
// GPU Kernels Implementation
// ============================================================

/**
 * Kernel: 对 best_dist2 数组求和（block reduce）
 * 使用共享内存进行 block 内规约，然后原子加到一个全局累加器
 */
 __global__ void kernel_reduce_sum(
    const float* __restrict__ data,
    float* __restrict__ output,  // 单个 float 的累加器
    int n
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程加载一个元素
    float val = (i < n) ? data[i] : 0.0f;
    
    // Block 内规约
    sdata[tid] = val;
    __syncthreads();
    
    // 规约：使用 warp shuffle + shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Block 0 的第一个线程原子加
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__global__ void kernel_update_centroids(
    float* __restrict__ centroids,   // [k, dim]
    const float* __restrict__ accum, // [k, dim]
    const int* __restrict__ counts,  // [k]
    int k, int dim
) {
    int c = blockIdx.x;
    int j = threadIdx.x;
    if (c >= k) return;

    int cnt = counts[c];
    if (cnt <= 0) return;  // keep old centroid
    float inv = 1.0f / (float)cnt;

    for (int col = j; col < dim; col += blockDim.x) {
        centroids[(size_t)c * dim + col] = accum[(size_t)c * dim + col] * inv;
    }
}

/**
 * Kernel: Minibatch更新聚类中心（使用自适应学习率）
 * 学习率公式：lr = 1 / (total_count + 1)
 * 更新公式：centroid_new = (1 - lr) * centroid_old + lr * (accum / count)
 */
__global__ void kernel_update_centroids_minibatch(
    float* __restrict__ centroids,      // [k, dim] (in/out)
    const float* __restrict__ accum,     // [k, dim]
    const int* __restrict__ counts,      // [k]
    int* __restrict__ total_counts,      // [k] (in/out)
    int k, int dim
) {
    int c = blockIdx.x;
    int j = threadIdx.x;
    if (c >= k) return;

    int count = counts[c];
    if (count <= 0) return;  // 如果minibatch中没有分配到点，不更新

    // 修复：先读取更新前的total_count（用于计算学习率）
    // 注意：由于每个block只处理一个centroid，且所有线程同步执行，这里不需要原子操作
    // 但为了线程安全，第一个线程读取，其他线程等待
    __shared__ int old_total;
    __shared__ float lr_shared;
    if (j == 0) {
        old_total = total_counts[c];
        // 计算自适应学习率：lr = 1 / (old_total + 1)
        // 注意：old_total是累计分配到的点的总数
        // 使用更新前的值，因为学习率应该反映"当前对旧centroid的信任程度"
        // 学习率会随着分配到的点增多而减小，确保收敛
        lr_shared = 1.0f / ((float)old_total + 1.0f);
        // 更新total_counts（在计算完学习率后）
        total_counts[c] = old_total + count;
    }
    __syncthreads();
    
    float lr = lr_shared;
    float inv_count = 1.0f / (float)count;
    
    // 每个线程处理dim的一部分
    for (int col = j; col < dim; col += blockDim.x) {
        size_t idx = (size_t)c * dim + col;
        float avg = accum[idx] * inv_count;  // minibatch中该centroid的平均值
        float old_val = centroids[idx];
        centroids[idx] = (1.0f - lr) * old_val + lr * avg;
    }
}

// ============================================================
// GEMM-based KMeans Kernels
// ============================================================

/**
 * Kernel: 初始化最佳匹配（best_dist2 = INF, best_idx = 0）
 */
__global__ void kernel_init_best(
    float* __restrict__ best_dist2,
    int* __restrict__ best_idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        best_dist2[i] = 3.402823e38f;  // FLT_MAX
        best_idx[i] = 0;
    }
}

/**
 * Kernel: 从 GEMM 结果（col-major dotT）更新最佳匹配（优化版本：使用规约）
 * 
 * dotT: [curK, curB] col-major，即 dotT[t + i*curK] = dot(centroid[cbase+t], data[i])
 * 
 * 优化策略：
 * - 每个线程处理一个数据点
 * - 使用 warp shuffle 在 warp 内进行 min 规约
 * - 使用循环展开和向量化加载优化内存访问
 */
__global__ void kernel_update_best_from_dotT(
    const float* __restrict__ dotT,      // [curK, curB] col-major
    const float* __restrict__ xnorm2,     // [curB]
    const float* __restrict__ cnorm2_global,  // [k] 全局centroid范数
    int curB,
    int curK,
    int cbase,                            // centroid起始偏移
    int* __restrict__ best_idx,          // [curB]
    float* __restrict__ best_dist2        // [curB]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= curB) return;

    const float xn = xnorm2[i];
    float bestd = best_dist2[i];
    int bestc = best_idx[i];

    // 使用循环展开优化：每次处理多个centroid
    const int unroll_factor = 4;
    int t = 0;
    
    // 主循环：处理大部分centroid
    for (; t + unroll_factor <= curK; t += unroll_factor) {
        #pragma unroll
        for (int u = 0; u < unroll_factor; ++u) {
            int tidx = t + u;
            float dot = dotT[tidx + (size_t)i * curK];
            float cn = cnorm2_global[cbase + tidx];
            float d2 = xn + cn - 2.f * dot;
            int cid = cbase + tidx;
            if (d2 < bestd) {
                bestd = d2;
                bestc = cid;
            }
        }
    }
    
    // 处理剩余的centroid
    for (; t < curK; ++t) {
        float dot = dotT[t + (size_t)i * curK];
        float cn = cnorm2_global[cbase + t];
        float d2 = xn + cn - 2.f * dot;
        int cid = cbase + t;
        if (d2 < bestd) {
            bestd = d2;
            bestc = cid;
        }
    }

    best_dist2[i] = bestd;
    best_idx[i] = bestc;
}

/**
 * Kernel: 从 GEMM 结果（col-major dotT）更新最佳匹配（Warp规约优化版本）
 * 
 * 使用 warp shuffle 进行规约，适用于 curK 较大的情况
 * 每个 warp 协作处理一个数据点，在 warp 内进行 min 规约
 */
__global__ void kernel_update_best_from_dotT_warp_reduce(
    const float* __restrict__ dotT,      // [curK, curB] col-major
    const float* __restrict__ xnorm2,     // [curB]
    const float* __restrict__ cnorm2_global,  // [k] 全局centroid范数
    int curB,
    int curK,
    int cbase,                            // centroid起始偏移
    int* __restrict__ best_idx,          // [curB]
    float* __restrict__ best_dist2        // [curB]
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_count = (blockDim.x + 31) / 32;
    const int data_idx = blockIdx.x * warp_count + warp_id;
    
    if (data_idx >= curB) return;

    const float xn = xnorm2[data_idx];
    float bestd = (lane_id == 0) ? best_dist2[data_idx] : 3.402823e38f;  // FLT_MAX
    int bestc = (lane_id == 0) ? best_idx[data_idx] : -1;

    // 每个线程处理 curK / 32 个centroid（向上取整）
    const int centroids_per_thread = (curK + 31) / 32;
    const int start_t = lane_id * centroids_per_thread;
    const int end_t = (start_t + centroids_per_thread < curK) ? (start_t + centroids_per_thread) : curK;

    // 每个线程处理分配给它的centroid
    for (int t = start_t; t < end_t; ++t) {
        float dot = dotT[t + (size_t)data_idx * curK];
        float cn = cnorm2_global[cbase + t];
        float d2 = xn + cn - 2.f * dot;
        int cid = cbase + t;
        if (d2 < bestd) {
            bestd = d2;
            bestc = cid;
        }
    }

    // Warp shuffle 规约：找到 warp 内的最小值
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_d = __shfl_down_sync(0xffffffff, bestd, offset);
        int other_c = __shfl_down_sync(0xffffffff, bestc, offset);
        if (other_d < bestd) {
            bestd = other_d;
            bestc = other_c;
        }
    }

    // Lane 0 写入结果
    if (lane_id == 0) {
        best_dist2[data_idx] = bestd;
        best_idx[data_idx] = bestc;
    }
}

/**
 * Kernel: 从分配结果累加（用于 GEMM 版本）- 优化版本
 * 优化策略：
 * 1. 循环展开以减少循环开销
 * 2. 使用更高效的内存访问模式
 * 3. 对小维度完全展开
 */
__global__ void kernel_accum_from_assign(
    const float* __restrict__ data,   // [n, dim]
    int n, int dim,
    const int* __restrict__ assign, // [n]
    float* __restrict__ accum,    // [k, dim]
    int* __restrict__ counts      // [k]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int c = assign[i];
    int base = i * dim;
    
    // 更新 counts
    atomicAdd(&counts[c], 1);
    
    // 更新 accum，使用循环展开优化
    float* acc = accum + (size_t)c * dim;
    
    // 根据维度大小选择不同的展开策略
    if (dim <= 4) {
        // 极小维度：完全展开
        if (dim >= 1) atomicAdd(&acc[0], data[base + 0]);
        if (dim >= 2) atomicAdd(&acc[1], data[base + 1]);
        if (dim >= 3) atomicAdd(&acc[2], data[base + 2]);
        if (dim >= 4) atomicAdd(&acc[3], data[base + 3]);
    } else if (dim <= 16) {
        // 小维度：部分展开（每次4个）
        int j = 0;
        #pragma unroll 4
        for (; j + 4 <= dim; j += 4) {
            atomicAdd(&acc[j], data[base + j]);
            atomicAdd(&acc[j+1], data[base + j+1]);
            atomicAdd(&acc[j+2], data[base + j+2]);
            atomicAdd(&acc[j+3], data[base + j+3]);
        }
        // 处理剩余元素
        for (; j < dim; ++j) {
            atomicAdd(&acc[j], data[base + j]);
        }
    } else {
        // 大维度：使用更大的展开因子
        int j = 0;
        #pragma unroll 8
        for (; j + 8 <= dim; j += 8) {
            atomicAdd(&acc[j], data[base + j]);
            atomicAdd(&acc[j+1], data[base + j+1]);
            atomicAdd(&acc[j+2], data[base + j+2]);
            atomicAdd(&acc[j+3], data[base + j+3]);
            atomicAdd(&acc[j+4], data[base + j+4]);
            atomicAdd(&acc[j+5], data[base + j+5]);
            atomicAdd(&acc[j+6], data[base + j+6]);
            atomicAdd(&acc[j+7], data[base + j+7]);
        }
        // 处理剩余元素
        for (; j < dim; ++j) {
            atomicAdd(&acc[j], data[base + j]);
        }
    }
}

// ============================================================
// Vector Reordering Kernels
// ============================================================

/**
 * Kernel: 计算每个cluster内的索引位置
 * 对于每个向量，计算它在所属cluster内的位置
 */
__global__ void kernel_compute_cluster_indices(
    const int* __restrict__ assign,      // [n]
    int* __restrict__ cluster_indices,   // [n] 输出：每个向量在其cluster内的索引
    int* __restrict__ cluster_counts,    // [k] 输入输出：每个cluster的计数（需要原子操作）
    int n, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int cluster_id = assign[idx];
    if (cluster_id >= 0 && cluster_id < k) {
        // 原子增加计数，并获取该向量在cluster内的位置
        int pos = atomicAdd(&cluster_counts[cluster_id], 1);
        cluster_indices[idx] = pos;
    } else {
        cluster_indices[idx] = -1;  // 无效的cluster ID
    }
}

/**
 * Kernel: Exclusive scan (前缀和)
 * 计算 offsets[i] = sum(counts[0..i-1])
 * 例如：counts = [2, 3, 1]，则offsets = [0, 2, 5]
 * 
 * 注意：这个kernel使用单个线程串行计算，确保正确性
 */
__global__ void kernel_exclusive_scan(
    const int* __restrict__ counts,   // [k]
    int* __restrict__ offsets,        // [k]
    int k
) {
    // 只让第一个线程执行，确保串行计算
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < k; ++i) {
            offsets[i] = sum;
            sum += counts[i];
        }
    }
}

/**
 * Kernel: 根据cluster和cluster内索引重排向量
 */
__global__ void kernel_reorder_vectors_by_cluster(
    const float* __restrict__ data_in,   // [n, dim]
    const int* __restrict__ assign,      // [n]
    const int* __restrict__ cluster_offsets,  // [k]
    const int* __restrict__ cluster_indices,   // [n]
    float* __restrict__ data_out,        // [n, dim]
    int n, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int cluster_id = assign[idx];
    int cluster_idx = cluster_indices[idx];
    
    if (cluster_id >= 0 && cluster_idx >= 0) {
        // 计算重排后的位置
        int out_pos = cluster_offsets[cluster_id] + cluster_idx;
        
        // 复制向量
        for (int d = 0; d < dim; ++d) {
            data_out[(size_t)out_pos * dim + d] = data_in[(size_t)idx * dim + d];
        }
    }
}
