#ifndef FUSION_L2_TOPK_CUH
#define FUSION_L2_TOPK_CUH

#include "pch.h"

/**
 * 融合算子：纯寄存器+双调排序 实现 topk
 **/
__global__ void fusion_l2_topk_warpsort_kernel(
    float* d_query_norm, 
    float* d_data_norm, 
    float* d_inner_product, 
    int* d_index,
    int* topk_index, 
    float* topk_dist,
    int n_query, 
    int n_batch, 
    int k
);

/**
 * 命名空间：pgvector::fusion_l2_topk_warpsort
 * 包含低级别的模板函数，用于直接操作GPU内存
 */
namespace pgvector {
namespace fusion_l2_topk_warpsort {

/**
 * 模板函数：从矩阵每一行中选取 top-k 最小或最大元素
 * 
 * 这是一个低级别的函数，直接操作GPU内存，不进行内存分配和传输
 * 
 * @param[in] d_query_norm query的L2范数 [batch_size]
 * @param[in] d_data_norm data的L2范数 [len]
 * @param[in] d_inner_product 内积矩阵 [batch_size, len]
 * @param[in] d_index 索引矩阵 [batch_size, len]
 * @param[in] batch_size 行数（批大小）
 * @param[in] len 每行的元素个数
 * @param[in] k 选取的元素个数
 * @param[out] output_vals 输出 top-k 值 [batch_size, k]
 * @param[out] output_idx 输出 top-k 对应的索引 [batch_size, k]
 * @param[in] select_min 若为 true 选取最小的 k 个，否则选取最大的 k 个
 * @param[in] stream CUDA流（可选，默认为0）
 * @return cudaError_t CUDA错误码
 */
template<typename T, typename IdxT>
cudaError_t fusion_l2_topk_warpsort(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product, const IdxT* d_index,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0
);

} // namespace fusion_l2_topk_warpsort
} // namespace pgvector
#endif // FUSION_L2_TOPK_CUH