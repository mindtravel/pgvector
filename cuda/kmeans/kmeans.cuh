#ifndef PGVECTOR_CUDA_KMEANS_CUH
#define PGVECTOR_CUDA_KMEANS_CUH

#include "../pch.h"

// ============================================================
// Config
// ============================================================
enum DataType {
    USE_FP32 = 0,
    USE_FP16 = 1,
};

struct KMeansCase {
    int n;          // number of vectors
    int dim;        // vector dimension
    int k;          // number of clusters
    int iters;      // Lloyd iterations
    int seed;       // random seed
    DistanceType dist;  // 使用 pch.h 中定义的 DistanceType
    DataType dtype;
};

// ============================================================
// GPU Kernels Declaration
// ============================================================

/**
 * Kernel: 更新聚类中心
 * 
 * @param centroids 输出：更新后的聚类中心 [k, dim]
 * @param accum 输入：累加器 [k, dim]
 * @param counts 输入：每个聚类的向量数量 [k]
 * @param k 聚类数量
 * @param dim 向量维度
 */
__global__ void kernel_update_centroids(
    float* __restrict__ centroids,   // [k, dim]
    const float* __restrict__ accum, // [k, dim]
    const int* __restrict__ counts,  // [k]
    int k, int dim
);

// ============================================================
// GPU KMeans Runner
// ============================================================

/**
 * GPU KMeans Lloyd 算法实现（GEMM 优化版本）
 * 
 * 使用 GEMM 计算距离矩阵，适用于大规模 K 的情况
 * 复用 l2norm 模块计算范数，使用 cublas GEMM 计算内积
 * 
 * @param cfg KMeans 配置
 * @param d_data 设备端向量数据 [n, dim] row-major (float)
 * @param d_assign 输出：设备端分配结果 [n]
 * @param d_centroids 输入输出：设备端聚类中心 [k, dim] float（输入为初始值，输出为最终值）
 * @param h_objective 输出：目标函数值（所有点到最近聚类中心的距离平方和）
 */
void gpu_kmeans_lloyd(
    const KMeansCase& cfg,
    const float* d_data,            // [n, dim] row-major
    int* d_assign,                 // [n]
    float* d_centroids,             // [k, dim] float (in/out)
    float* h_objective               // scalar (sum dist2)
);

// ============================================================
// CPU Initialization Functions
// ============================================================

/**
 * 确定性初始化聚类中心：从数据中随机采样 k 个不同的点
 * 
 * 使用 Fisher-Yates 洗牌算法确保无放回抽样，k 个聚类中心都是不同的点
 * 使用固定的随机种子确保 CPU 和 GPU 测试使用相同的初始聚类中心
 * 
 * @param cfg KMeans 配置（包含 seed）
 * @param data 输入数据 [n, dim] row-major
 * @param out_centroids 输出：初始聚类中心 [k, dim] row-major
 */
__host__ void init_centroids_by_sampling(
    const KMeansCase& cfg,
    const float* data,        // [n, dim]
    float* out_centroids      // [k, dim]
);

#endif // PGVECTOR_CUDA_KMEANS_CUH

