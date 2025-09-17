#include "../include/distance.h"
#include <cuda_runtime.h>
#include <stdio.h>

// 只做声明，不实现
extern "C" {
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
}
