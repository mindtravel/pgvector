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
    int minibatch_iters;      // Minibatch iters
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

/**
 * Kernel: Minibatch更新聚类中心（使用自适应学习率）
 * 
 * @param centroids 输入输出：聚类中心 [k, dim]
 * @param accum 输入：minibatch累加器 [k, dim]
 * @param counts 输入：minibatch中每个聚类的向量数量 [k]
 * @param total_counts 输入输出：累计分配次数 [k]（用于计算自适应学习率）
 * @param k 聚类数量
 * @param dim 向量维度
 */
__global__ void kernel_update_centroids_minibatch(
    float* __restrict__ centroids,      // [k, dim] (in/out)
    const float* __restrict__ accum,    // [k, dim]
    const int* __restrict__ counts,     // [k]
    int* __restrict__ total_counts,      // [k] (in/out)
    int k, int dim
);

/**
 * Kernel: 初始化最佳匹配（best_dist2 = INF, best_idx = 0）
 */
__global__ void kernel_init_best(
    float* __restrict__ best_dist2,
    int* __restrict__ best_idx,
    int n
);

/**
 * Kernel: 从 GEMM 结果（col-major dotT）更新最佳匹配
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
);

/**
 * Kernel: 从分配结果累加（用于 GEMM 版本）
 */
__global__ void kernel_accum_from_assign(
    const float* __restrict__ data,   // [n, dim]
    int n, int dim,
    const int* __restrict__ assign, // [n]
    float* __restrict__ accum,    // [k, dim]
    int* __restrict__ counts      // [k]
);

/**
 * Kernel: 对数组求和（block reduce）
 */
__global__ void kernel_reduce_sum(
    const float* __restrict__ data,
    float* __restrict__ output,  // 单个 float 的累加器
    int n
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
 * @param h_data 主机端向量数据 [n, dim] row-major (float)，支持大于显存的数据集
 * @param d_assign 输出：设备端分配结果 [n]
 * @param d_centroids 输入输出：设备端聚类中心 [k, dim] float（输入为初始值，输出为最终值），常驻显存
 * @param h_objective 输出：目标函数值（所有点到最近聚类中心的距离平方和）
 */
void gpu_kmeans_lloyd(
    const KMeansCase& cfg,
    const float* h_data,            // [n, dim] row-major，主机端数据
    int* d_assign,                 // [n]
    float* d_centroids,             // [k, dim] float (in/out)，常驻显存
    float* h_objective               // scalar (sum dist2)
);

/**
 * GPU KMeans Minibatch 算法实现（GEMM 优化版本）
 * 
 * 每次迭代只使用一个minibatch来更新聚类中心，使用学习率进行增量更新
 * 适用于大规模数据集和在线学习场景
 * 
 * @param cfg KMeans 配置
 * @param h_data 主机端向量数据 [n, dim] row-major (float)
 * @param d_assign 输出：设备端分配结果 [n] (可选，minibatch可能不更新所有assign)
 * @param d_centroids 输入输出：设备端聚类中心 [k, dim] float（输入为初始值，输出为最终值），常驻显存
 * @param h_objective 输出：目标函数值（可选）
 */
void gpu_kmeans_minibatch(
    const KMeansCase& cfg,
    const float* h_data,            // [n, dim] row-major，主机端数据
    int* d_assign,                 // [n] (可选)
    float* d_centroids,             // [k, dim] float (in/out)，常驻显存
    float* h_objective               // scalar (sum dist2) (可选)
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

// ============================================================
// Vector Reordering After Clustering
// ============================================================

/**
 * Cluster信息结构：存储每个cluster的起始位置和大小
 */
struct ClusterInfo {
    long long* offsets;    // [k] 每个cluster的起始位置（在重排后的数据中，使用int64支持大数据集）
    int* counts;           // [k] 每个cluster的向量数量
    int k;                 // cluster数量
};

/**
 * GPU Kernel: 计算每个cluster内的索引位置
 */
__global__ void kernel_compute_cluster_indices(
    const int* __restrict__ assign,      // [n]
    int* __restrict__ cluster_indices,   // [n] 输出：每个向量在其cluster内的索引
    int* __restrict__ cluster_counts,    // [k] 输入输出：每个cluster的计数（需要原子操作）
    int n, int k
);

/**
 * GPU Kernel: Exclusive scan (前缀和)
 * 计算 offsets[i] = sum(counts[0..i-1])
 * 
 * @param counts 输入：每个cluster的计数 [k]
 * @param offsets 输出：每个cluster的起始位置 [k]
 * @param k cluster数量
 */
__global__ void kernel_exclusive_scan(
    const int* __restrict__ counts,   // [k]
    int* __restrict__ offsets,        // [k]
    int k
);

/**
 * GPU Kernel: 根据assign结果重排向量
 * 
 * @param data_in 输入：原始向量数据 [n, dim] row-major
 * @param assign 输入：分配结果 [n]，assign[i] 表示第i个向量属于哪个cluster
 * @param cluster_offsets 输入：每个cluster的起始位置 [k]
 * @param cluster_indices 输入：每个cluster内的索引 [n]
 * @param data_out 输出：重排后的向量数据 [n, dim] row-major
 * @param n 向量数量
 * @param dim 向量维度
 */
__global__ void kernel_reorder_vectors_by_cluster(
    const float* __restrict__ data_in,   // [n, dim]
    const int* __restrict__ assign,      // [n]
    const int* __restrict__ cluster_offsets,  // [k]
    const int* __restrict__ cluster_indices,   // [n]
    float* __restrict__ data_out,        // [n, dim]
    int n, int dim
);

/**
 * GPU版本：根据聚类结果重排向量（流式处理，支持大数据集）
 * 
 * 将属于同一cluster的向量连续存储在一起，提高后续访问的局部性
 * 使用流式处理，支持大于GPU内存的数据集
 * 
 * @param cfg KMeans 配置
 * @param h_data_in 输入：主机端原始向量数据 [n, dim] row-major（可以是pageable或pinned memory）
 * @param h_assign 输入：主机端分配结果 [n]
 * @param h_data_out 输出：主机端重排后的向量数据 [n, dim] row-major
 * @param h_cluster_info 输出：主机端cluster信息（offsets和counts），可以为nullptr
 * @param device_id GPU设备ID
 * @param B batch大小（例如 1<<20 = 1M）
 * @param stream CUDA流，可以为0使用默认流
 */
void gpu_reorder_vectors_by_cluster(
    const KMeansCase& cfg,
    const float* h_data_in,    // [n,dim] CPU
    const int*   h_assign,     // [n] CPU
    float*       h_data_out,   // [n,dim] CPU
    ClusterInfo* h_cluster_info, // optional host output
    int device_id = 0,
    int B = (1 << 20),         // batch size (e.g. 1<<20)
    cudaStream_t stream = 0
);

/**
 * GPU版本：构建permutation数组（流式处理，支持大数据集）
 * 
 * 构建permutation数组perm[0..n-1]，使得属于cluster 0的点在前，然后是cluster 1，以此类推
 * 每个cluster内的顺序是稳定的
 * 
 * 注意：此函数只处理assign，不处理向量数据，避免大内存分配
 * 调用者可以通过 out[p] = in[perm[p]] 来生成重排后的向量
 * 
 * @param cfg KMeans 配置
 * @param h_assign 输入：主机端分配结果 [n]
 * @param h_perm_out 输出：主机端permutation数组 [n]，可以为nullptr（如果只需要cluster_info）
 * @param h_cluster_info 输出：主机端cluster信息（offsets和counts），可以为nullptr
 * @param device_id GPU设备ID
 * @param B batch大小（例如 1<<20 = 1M）
 * @param stream CUDA流，可以为0使用默认流
 */
void gpu_build_permutation_by_cluster(
    const KMeansCase& cfg,
    const int* h_assign,             // [n] host
    int* h_perm_out,                 // [n] host (optional, can be nullptr)
    ClusterInfo* h_cluster_info,      // optional host output
    int device_id = 0,
    int B = (1 << 20),               // e.g. 1<<20
    cudaStream_t stream = 0
);

/**
 * CPU版本：根据permutation数组重排向量（多线程实现）
 * 
 * 使用多线程并行处理，提高性能
 * 使用pageable memory，避免pinned memory限制
 * 
 * @param h_data_in 输入：原始向量数据 [n, dim] row-major
 * @param h_perm 输入：permutation数组 [n]，perm[p] 表示重排后位置p对应的原始索引
 * @param h_data_out 输出：重排后的向量数据 [n, dim] row-major
 * @param n 向量数量
 * @param dim 向量维度
 */
void cpu_reorder_vectors_by_permutation(
    const float* h_data_in,    // [n, dim] CPU
    const int* h_perm,         // [n] CPU permutation array
    float* h_data_out,         // [n, dim] CPU output
    int n, int dim
);

/**
 * IVF K-means：完整的K-means聚类到内存物理重排流程
 * 
 * 该函数整合了以下步骤：
 * 1. K-means聚类（Lloyd或Minibatch算法）
 * 2. 构建permutation数组（按cluster重排）
 * 3. 重排向量数据（物理内存重排）
 * 
 * @param cfg KMeans配置
 * @param h_data_in 输入：原始向量数据 [n, dim] row-major (pageable memory)
 * @param h_data_out 输出：重排后的向量数据 [n, dim] row-major (pageable memory)，必须预先分配
 * @param d_centroids 输入输出：设备端聚类中心 [k, dim] (输入为初始值，输出为最终值)
 * @param h_cluster_info 输出：cluster信息（offsets和counts），可以为nullptr
 * @param use_minibatch 是否使用Minibatch算法（true=Minibatch, false=Lloyd）
 * @param device_id GPU设备ID
 * @param batch_size permutation构建的batch大小（例如 1<<20）
 * @param h_objective 输出：目标函数值，可以为nullptr
 * @return 成功返回true，失败返回false
 */
bool ivf_kmeans(
    const KMeansCase& cfg,
    const float* h_data_in,        // [n, dim] CPU input
    float* h_data_out,             // [n, dim] CPU output (must be pre-allocated)
    float* d_centroids,            // [k, dim] GPU (in/out)
    ClusterInfo* h_cluster_info,   // optional output
    bool use_minibatch = false,    // true for minibatch, false for Lloyd
    int device_id = 0,
    int batch_size = (1 << 20),   // batch size for permutation building
    float* h_objective = nullptr  // optional output
);

/**
 * 释放ClusterInfo结构的内存
 */
void free_cluster_info(ClusterInfo* info, bool is_device);

#endif // PGVECTOR_CUDA_KMEANS_CUH

