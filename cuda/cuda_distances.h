#include <stdio.h>
/*
* 计算余弦距离的核函数
*/
__global__ void compute_cosine_distances_kernel(
    float* query_vector,
    float* list_vectors,
    int* list_offsets,
    int* list_counts,
    float* distances,
    int vector_dim,
    int total_vectors
);