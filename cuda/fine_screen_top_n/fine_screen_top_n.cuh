#ifndef FINE_SCREEN_TOP_N
#define FINE_SCREEN_TOP_N

#include "pch.h"


/**
 * 精筛topn融合算子: 
 * 使用连续存储结构（符合pgvector实现），向量按聚类物理连续存储
 * 
 * 参数说明:
 * - h_query_group: query向量组（二维指针）
 * - h_cluster_query_offset: cluster-query倒排索引的offset数组（标准格式：n_total_clusters+1个元素）
 * - h_cluster_query_data: cluster-query倒排索引的数据（连续存储）
 * - h_cluster_vector_index: 每个cluster在全局向量数组中的连续起始位置 [n_total_clusters]
 * - h_cluster_vector_num: 每个cluster的向量数量 [n_total_clusters]
 * - h_cluster_vector: 所有向量数据，按聚类物理连续存储 [n_total_vectors][n_dim]
 * - n_query: query总数
 * - n_cluster: 每个query精筛的cluster数量
 * - n_total_clusters: 所有distinct cluster数（用于offset数组大小）
 * - n_dim: 向量维度
 * - n_topn: 筛选Top-N（K）个
 * - n_total_vectors: 所有向量总数
 * 
 * 出参:
 * - h_query_topn_index: query对应的topn向量的原始索引（二维指针）
 * - h_query_topn_dist: query对应的topn向量的距离（二维指针）
 **/
void fine_screen_top_n(
    float** h_query_group, int* h_cluster_query_offset, int* h_cluster_query_data,
    int* h_cluster_vector_index, int* h_cluster_vector_num, float** h_cluster_vector,
    int n_query, int n_cluster, int n_total_clusters, int n_dim, int n_topn, int n_total_vectors,
    int** h_query_topn_index, float** h_query_topn_dist
);

/**
 * 精筛topn融合算子（v1版本：流式计算）
 * 
 * 使用流式计算方式，在kernel内部维护warp-sort queue，直接写入最终结果。
 * 内存占用从 O(n_query * max_candidates) 降至 O(n_query * k)。
 * 
 * 参数说明（与fine_screen_top_n相同）:
 * - h_query_group: query向量组（二维指针）
 * - h_cluster_query_offset: cluster-query倒排索引的offset数组（标准格式：n_total_clusters+1个元素）
 * - h_cluster_query_data: cluster-query倒排索引的数据（连续存储）
 * - h_cluster_vector_index: 每个cluster在全局向量数组中的连续起始位置 [n_total_clusters]
 * - h_cluster_vector_num: 每个cluster的向量数量 [n_total_clusters]
 * - h_cluster_vector: 所有向量数据，按聚类物理连续存储 [n_total_vectors][n_dim]
 * - n_query: query总数
 * - n_cluster: 每个query精筛的cluster数量
 * - n_total_clusters: 所有distinct cluster数（用于offset数组大小）
 * - n_dim: 向量维度
 * - n_topn: 筛选Top-N（K）个（必须 <= 256）
 * - n_total_vectors: 所有向量总数
 * 
 * 出参:
 * - h_query_topn_index: query对应的topn向量的原始索引（二维指针）
 * - h_query_topn_dist: query对应的topn向量的距离（二维指针）
 * 
 * 限制：
 * - n_topn <= 256 (warp-sort容量限制)
 */
void fine_screen_top_n_v1(
    float** h_query_group, int* h_cluster_query_offset, int* h_cluster_query_data,
    int* h_cluster_vector_index, int* h_cluster_vector_num, float** h_cluster_vector,
    int n_query, int n_cluster, int n_total_clusters, int n_dim, int n_topn, int n_total_vectors,
    int** h_query_topn_index, float** h_query_topn_dist
);

#endif // FINE_SCREEN_TOP_N