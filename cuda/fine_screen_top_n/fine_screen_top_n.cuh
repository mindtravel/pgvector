#ifndef FINE_SCREEN_TOP_N
#define FINE_SCREEN_TOP_N

#include "pch.h"


void fine_screen_top_n_blocks(
    float* h_query_group,

    float** h_block_vectors,
    int* h_block_vector_counts,
    int* h_block_query_offset,
    int* h_block_query_data,
    int* h_block_query_probe_indices,  // 新增：每个block-query对中probe在query中的索引
    int* h_query_topn_index,
    float* h_query_topn_dist,

    float** candidate_dist,
    int** candidate_index,

    int n_query,
    int n_probes,
    int n_dim,
    int k

);

#endif // FINE_SCREEN_TOP_N



