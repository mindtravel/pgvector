// filepath: cuda-distance-lib/include/distance.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void search_batch_cosine_distance_cuda(
    float* h_query, float* h_data,
    int N1, int N2, int D, int k,
    int* h_result
);

void search_batch_l2_distance_cuda(
    float* h_query, float* h_data,
    int N1, int N2, int D, int k,
    int* h_result
);

#ifdef __cplusplus
}
#endif