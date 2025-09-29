#ifndef FUSION_COS_TOPK_H
#define FUSION_COS_TOPK_H

#include "pch.h"

/*
内积->余弦距离->topk融合算子
*/
__global__ void fusion_cos_topk_kernel(
    float* d_query_norm, float* d_data_norm, float* d_inner_product,
    int* topk_index, float* topk_dist,
    int n_query, int n_batch, int k
);

#endif
