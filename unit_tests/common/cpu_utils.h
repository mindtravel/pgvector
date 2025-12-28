#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include <cmath>
#include <cstring>
#include <algorithm>

/**
 * CPU 工具函数库
 * 包含典型的距离计算和长度计算函数
 */

/**
 * 计算向量的平方和（sum of squares）
 * @param vector 向量数据
 * @param n_dim 向量维度
 * @return 平方和
 */
inline float compute_squared_sum(const float* vector, int n_dim) {
    float sum = 0.0f;
    for (int d = 0; d < n_dim; d++) {
        sum += vector[d] * vector[d];
    }
    return sum;
}

/**
 * 计算向量的 L2 范数（L2 norm）
 * @param vector 向量数据
 * @param n_dim 向量维度
 * @return L2 范数（sqrt(sum of squares)）
 */
inline float compute_l2_norm(const float* vector, int n_dim) {
    float sum = compute_squared_sum(vector, n_dim);
    return std::sqrt(sum);
}

/**
 * L2 归一化向量（原地修改）
 * @param x 向量数据（会被修改）
 * @param dim 向量维度
 */
inline void l2_normalize_inplace(float* x, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) {
        s += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    double inv = (s > 1e-12) ? (1.0 / std::sqrt(s)) : 0.0;
    for (int i = 0; i < dim; ++i) {
        x[i] = static_cast<float>(x[i] * inv);
    }
}

/**
 * 计算两个向量的内积（dot product）
 * @param a 向量 a
 * @param b 向量 b
 * @param dim 向量维度
 * @return 内积值
 */
inline float dot_product(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * 计算两个向量的 L2 距离平方（squared L2 distance）
 * @param a 向量 a
 * @param b 向量 b
 * @param dim 向量维度
 * @return L2 距离平方
 */
inline float l2_distance_squared(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * 计算两个向量的 L2 距离（L2 distance）
 * @param a 向量 a
 * @param b 向量 b
 * @param dim 向量维度
 * @return L2 距离
 */
inline float l2_distance(const float* a, const float* b, int dim) {
    return std::sqrt(l2_distance_squared(a, b, dim));
}

/**
 * 计算两个向量的余弦距离（cosine distance）
 * 余弦距离 = 1 - 余弦相似度 = 1 - (a·b) / (||a|| * ||b||)
 * @param a 向量 a
 * @param b 向量 b
 * @param dim 向量维度
 * @return 余弦距离
 */
inline float cosine_distance(const float* a, const float* b, int dim) {
    float dot = dot_product(a, b, dim);
    float norm_a = compute_l2_norm(a, dim);
    float norm_b = compute_l2_norm(b, dim);
    
    if (norm_a < 1e-12f || norm_b < 1e-12f) {
        return 1.0f;  // 零向量，返回最大距离
    }
    
    float cosine_similarity = dot / (norm_a * norm_b);
    // 限制范围到 [-1, 1]，避免数值误差
    cosine_similarity = std::max(-1.0f, std::min(1.0f, cosine_similarity));
    
    return 1.0f - cosine_similarity;
}

/**
 * 计算两个向量的余弦相似度（cosine similarity）
 * 余弦相似度 = (a·b) / (||a|| * ||b||)
 * @param a 向量 a
 * @param b 向量 b
 * @param dim 向量维度
 * @return 余弦相似度
 */
inline float cosine_similarity(const float* a, const float* b, int dim) {
    float dot = dot_product(a, b, dim);
    float norm_a = compute_l2_norm(a, dim);
    float norm_b = compute_l2_norm(b, dim);
    
    if (norm_a < 1e-12f || norm_b < 1e-12f) {
        return 0.0f;  // 零向量，返回 0
    }
    
    float similarity = dot / (norm_a * norm_b);
    // 限制范围到 [-1, 1]，避免数值误差
    return std::max(-1.0f, std::min(1.0f, similarity));
}

/**
 * 批量计算向量的平方和
 * @param vectors 向量数组 [n_batch][n_dim] 或连续存储 [n_batch * n_dim]
 * @param squared_sums 输出的平方和数组 [n_batch]
 * @param n_batch 向量数量
 * @param n_dim 向量维度
 * @param stride 向量之间的步长（如果连续存储则为 n_dim）
 */
inline void compute_squared_sums_batch(
    const float* vectors, 
    float* squared_sums, 
    int n_batch, 
    int n_dim,
    int stride = 0
) {
    if (stride == 0) stride = n_dim;
    for (int i = 0; i < n_batch; i++) {
        squared_sums[i] = compute_squared_sum(vectors + i * stride, n_dim);
    }
}

/**
 * 批量计算向量的 L2 范数
 * @param vectors 向量数组 [n_batch][n_dim] 或连续存储 [n_batch * n_dim]
 * @param norms 输出的 L2 范数数组 [n_batch]
 * @param n_batch 向量数量
 * @param n_dim 向量维度
 * @param stride 向量之间的步长（如果连续存储则为 n_dim）
 */
inline void compute_l2_norms_batch(
    const float* vectors, 
    float* norms, 
    int n_batch, 
    int n_dim,
    int stride = 0
) {
    if (stride == 0) stride = n_dim;
    for (int i = 0; i < n_batch; i++) {
        norms[i] = compute_l2_norm(vectors + i * stride, n_dim);
    }
}

/**
 * 批量计算两个向量集合之间的 L2 距离平方
 * @param vectors_a 向量集合 a [n_batch][n_dim] 或连续存储 [n_batch * n_dim]
 * @param vectors_b 向量集合 b [n_batch][n_dim] 或连续存储 [n_batch * n_dim]
 * @param distances 输出的距离数组 [n_batch]
 * @param n_batch 向量数量
 * @param n_dim 向量维度
 * @param stride_a 向量 a 之间的步长（如果连续存储则为 n_dim）
 * @param stride_b 向量 b 之间的步长（如果连续存储则为 n_dim）
 */
inline void compute_l2_distances_squared_batch(
    const float* vectors_a,
    const float* vectors_b,
    float* distances,
    int n_batch,
    int n_dim,
    int stride_a = 0,
    int stride_b = 0
) {
    if (stride_a == 0) stride_a = n_dim;
    if (stride_b == 0) stride_b = n_dim;
    for (int i = 0; i < n_batch; i++) {
        distances[i] = l2_distance_squared(
            vectors_a + i * stride_a,
            vectors_b + i * stride_b,
            n_dim
        );
    }
}

/**
 * 批量计算两个向量集合之间的余弦距离
 * @param vectors_a 向量集合 a [n_batch][n_dim] 或连续存储 [n_batch * n_dim]
 * @param vectors_b 向量集合 b [n_batch][n_dim] 或连续存储 [n_batch * n_dim]
 * @param distances 输出的距离数组 [n_batch]
 * @param n_batch 向量数量
 * @param n_dim 向量维度
 * @param stride_a 向量 a 之间的步长（如果连续存储则为 n_dim）
 * @param stride_b 向量 b 之间的步长（如果连续存储则为 n_dim）
 */
inline void compute_cosine_distances_batch(
    const float* vectors_a,
    const float* vectors_b,
    float* distances,
    int n_batch,
    int n_dim,
    int stride_a = 0,
    int stride_b = 0
) {
    if (stride_a == 0) stride_a = n_dim;
    if (stride_b == 0) stride_b = n_dim;
    for (int i = 0; i < n_batch; i++) {
        distances[i] = cosine_distance(
            vectors_a + i * stride_a,
            vectors_b + i * stride_b,
            n_dim
        );
    }
}

#endif // CPU_UTILS_H

