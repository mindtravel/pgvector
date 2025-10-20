#ifndef DISTANCES_H
#define DISTANCES_H
void cuda_cosine_dist(
    float** query_vector_group_cpu, float** data_vector_group_cpu, float** h_cos_dist,
    int n_query, int n_batch, int n_dim, 
    float alpha, float beta
);

#endif