#include <stdio.h>

/*
* 全局变量用于GPU内存管理
*/ 
static float* d_query_vector = NULL;
static float* d_list_vectors = NULL;
static int* d_list_offsets = NULL;
static int* d_list_counts = NULL;
static float* d_distances = NULL;
static int* d_indices = NULL;
static int gpu_initialized = 0;

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