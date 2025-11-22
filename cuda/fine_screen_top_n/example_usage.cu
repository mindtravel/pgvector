// 使用示例：如何使用修复后的双流流水线
#include "fine_screen_top_n_stream.cuh"
#include "../unit_tests/common/test_utils.cuh"

void example_usage() {
    // 1. 准备输入数据
    int n_query = 1000;
    int k = 5;
    int batch_size = 3;
    int* query_cluster_group = (int*)malloc(n_query * k * sizeof(int));
    
    // 填充测试数据
    for (int i = 0; i < n_query * k; i++) {
        query_cluster_group[i] = rand() % 100; // 假设有100个cluster
    }
    
    // 2. 生成cluster数据
    void* cluster_data_ptr = generate_cluster_query_data(query_cluster_group, n_query, k, batch_size);
    
    // 3. 准备query向量数据
    int n_dim = 128;
    float* h_query_group = (float*)malloc(n_query * n_dim * sizeof(float));
    for (int i = 0; i < n_query * n_dim; i++) {
        h_query_group[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    // 4. 准备输出缓冲区
    int n_topn = 10;
    int* h_query_topn_index = (int*)malloc(n_query * n_topn * sizeof(int));
    float* h_query_topn_dist = (float*)malloc(n_query * n_topn * sizeof(float));
    
    // 5. 计算batch数量
    int num_batches = (100 + batch_size - 1) / batch_size; // 假设有100个cluster
    
    // 6. 使用双流流水线处理
    simple_dual_stream_pipeline(
        cluster_data_ptr, num_batches, batch_size,
        h_query_group, n_query, n_dim, n_topn,
        h_query_topn_index, h_query_topn_dist
    );
    
    // 7. 清理内存
    free(query_cluster_group);
    free(h_query_group);
    free(h_query_topn_index);
    free(h_query_topn_dist);
    
    // 注意：cluster_data_ptr的内存需要在适当的时候释放
    // 这里需要根据实际的内存管理策略来处理
}
