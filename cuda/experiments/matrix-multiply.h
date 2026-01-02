#ifndef MATRIX_MULTIPLY_CUH
#define MATRIX_MULTIPLY_CUH

// CUDA矩阵乘法算子接口
void cuda_sgemmNN(float* h_A, float* h_B, float* h_C,
                  int M, int N, int K, float alpha, float beta);

void cuda_sgemmNN_ours(const float** h_A, const float** h_B, float** h_C,
    int M, int N, int K, float alpha, float beta);

void cuda_cosine_dist(
    const float** query_vector_group_cpu, const float** data_vector_group_cpu, float** h_cos_dist,
    int n_query, int n_batch, int n_dim, 
    float alpha, float beta
);
#endif 