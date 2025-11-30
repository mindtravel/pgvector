#ifndef FINE_SCREEN_TOPK
#define FINE_SCREEN_TOPK

#include "pch.h"


void fine_screen_topk(
    float* h_query_group,

    float** h_block_vectors,
    int* h_block_vector_counts,
    int* h_block_query_offset,  // 大小为(n_total_clusters + 1)，包含所有cluster
    int* h_block_query_data,
    int* h_block_query_probe_indices,  // 新增：每个block-query对中probe在query中的索引
    int** h_query_topk_index,
    float** h_query_topk_dist,

    int n_query,
    int n_total_clusters,  // 总的cluster数量
    int n_probes,  // 每个query的probe数量
    int n_dim,
    int k

);

#endif // FINE_SCREEN_TOPK




