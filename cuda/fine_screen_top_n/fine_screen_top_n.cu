#include "fine_screen_top_n.cuh"


void fine_screen_top_n(
    float* h_query_group, int* h_query_cluster_group, int* h_cluster_query_offset, int* h_cluster_query_data,
    int* cluster_map,
    int* h_cluster_vector_index, int* h_cluster_vector_num, float** h_cluster_vector,
    int n_query, int n_cluster, int distinct_cluster_count, int n_dim, int n_topn, int max_cluster_id, int tol_vector,
    int* h_query_topn_index, float* h_query_topn_dist
) {
    // 计算内存大小
    size_t size_query_group = n_query * n_dim * sizeof(float);
    size_t size_query_cluster_group = n_query * n_cluster * sizeof(int); //每个query对应n个cluster
    size_t size_cluster_query_offset = distinct_cluster_count * sizeof(int);  // distinct cluster数量
    size_t size_cluster_query_data = n_query * n_cluster * sizeof(int);  // 每个query对应n个cluster
    size_t size_cluster_map = distinct_cluster_count * sizeof(int);  // distinct cluster数量
    size_t size_cluster_vector_index = distinct_cluster_count * sizeof(int);  // distinct cluster数量
    size_t size_cluster_vector_num = distinct_cluster_count * sizeof(int);  // distinct cluster数量
    size_t size_cluster_vector = tol_vector * n_dim * sizeof(float);  // 总向量数量
    size_t size_topn_index = n_query * n_topn * sizeof(int);
    size_t size_topn_dist = n_query * n_topn * sizeof(float);
    
    // 分配设备内存
    float *d_query_group, *d_cluster_vector, *d_topn_dist;
    int *d_query_cluster_group, *d_cluster_query_offset, *d_cluster_query_data;
    int *d_cluster_vector_index, *d_cluster_vector_num, *d_topn_index, *d_cluster_map;
    
    // GPU内存分配
    cudaMalloc(&d_query_group, size_query_group);
    cudaMalloc(&d_query_cluster_group, size_query_cluster_group);
    cudaMalloc(&d_cluster_query_offset, size_cluster_query_offset);
    cudaMalloc(&d_cluster_query_data, size_cluster_query_data);
    cudaMalloc(&d_cluster_map, size_cluster_map);
    cudaMalloc(&d_cluster_vector_index, size_cluster_vector_index);
    cudaMalloc(&d_cluster_vector_num, size_cluster_vector_num);
    cudaMalloc(&d_cluster_vector, size_cluster_vector);
    cudaMalloc(&d_topn_index, size_topn_index);
    cudaMalloc(&d_topn_dist, size_topn_dist);
    
    // 复制数据到设备内存
    cudaMemcpy(d_query_group, h_query_group, size_query_group, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_cluster_group, h_query_cluster_group, size_query_cluster_group, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_query_offset, h_cluster_query_offset, size_cluster_query_offset, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_query_data, h_cluster_query_data, size_cluster_query_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_map, cluster_map, size_cluster_map, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector_index, h_cluster_vector_index, size_cluster_vector_index, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector_num, h_cluster_vector_num, size_cluster_vector_num, cudaMemcpyHostToDevice);
    // 使用cudaMemcpy2D从二维指针复制cluster向量数据到设备内存
    // h_cluster_vector[i] 指向第i个cluster的向量数据
    cudaMemcpy2D(
        d_cluster_vector,                    // 目标设备内存
        n_dim * sizeof(float),              // 目标行间距
        h_cluster_vector[0],                // 源主机内存（第一个cluster的向量）
        n_dim * sizeof(float),              // 源行间距
        n_dim * sizeof(float),              // 每行字节数
        tol_vector,                          // 行数（cluster数量）
        cudaMemcpyHostToDevice
    );
    
    // TODO: 在这里添加实际的kernel计算逻辑
    
    // 复制结果回主机内存
    cudaMemcpy(h_query_topn_index, d_topn_index, size_topn_index, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_query_topn_dist, d_topn_dist, size_topn_dist, cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_query_group);
    cudaFree(d_query_cluster_group);
    cudaFree(d_cluster_query_offset);
    cudaFree(d_cluster_query_data);
    cudaFree(d_cluster_map);
    cudaFree(d_cluster_vector_index);
    cudaFree(d_cluster_vector_num);
    cudaFree(d_cluster_vector);
    cudaFree(d_topn_index);
    cudaFree(d_topn_dist);
}