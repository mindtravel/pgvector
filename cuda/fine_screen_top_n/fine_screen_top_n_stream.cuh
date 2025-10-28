#ifndef FINE_SCREEN_TOP_N_STREAM
#define FINE_SCREEN_TOP_N_STREAM

#include "pch.h"
#include "../unit_tests/common/test_utils.cuh"

/**
 * 简单的双流流水线处理cluster数据
 * @param cluster_data_array cluster数据数组
 * @param num_batches batch数量
 * @param batch_size batch大小
 * @param h_query_group query向量数据
 * @param n_query query数量
 * @param n_dim 向量维度
 * @param n_topn top-n数量
 * @param h_query_topn_index 输出索引
 * @param h_query_topn_dist 输出距离
 */
void simple_dual_stream_pipeline(
    ClusterQueryData* cluster_data_array, int num_batches, int batch_size,
    float* h_query_group, int n_query, int n_dim, int n_topn,
    int* h_query_topn_index, float* h_query_topn_dist);

#endif // FINE_SCREEN_TOP_N_STREAM
