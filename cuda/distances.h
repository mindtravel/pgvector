#pragma once
void cuda_cosine_dist(
    float** query_vector_group_cpu, float** data_vector_group_cpu, float** h_cos_dist,
    int n_query, int n_batch, int n_dim, 
    float alpha, float beta
);

void cuda_cosine_dist_topk(
    float** query_vector_group_cpu, float** data_vector_group_cpu, 
    int* data_index, int** topk_index,
    int n_query, int n_batch, int n_dim,
    int k /*查找的最近邻个数*/
);