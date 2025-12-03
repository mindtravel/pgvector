#include <stdexcept>
#include <cstdio>
#include <cstring>
#include "integrate_screen.cuh"

extern "C" {
/**
 * C包装函数，用于从C代码调用batch_search_pipeline（新接口：接受 device 指针）
 * 处理C++异常并返回错误码
 */
int batch_search_pipeline_wrapper(float* d_query_batch,
                                  int* d_cluster_size,
                                  float* d_cluster_vectors,
                                  float* d_cluster_centers,
                                  float* d_topk_dist,
                                  int* d_topk_index,
                                  int n_query, int n_dim, int n_total_cluster,
                                  int n_total_vectors, int n_probes, int k)
{
    /* 记录关键参数信息到PostgreSQL日志 */
    fprintf(stderr, "batch_search_pipeline_wrapper: ========== 参数验证 ==========\n");
    fprintf(stderr, "batch_search_pipeline_wrapper: n_query=%d, n_dim=%d, n_total_cluster=%d, n_total_vectors=%d, n_probes=%d, k=%d\n",
            n_query, n_dim, n_total_cluster, n_total_vectors, n_probes, k);
    
    /* 验证GPU内存指针 */
    fprintf(stderr, "batch_search_pipeline_wrapper: GPU内存指针:\n");
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_query_batch=%p (expected size: %zu bytes)\n",
            (void*)d_query_batch, (size_t)n_query * n_dim * sizeof(float));
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_cluster_size=%p (expected size: %zu bytes)\n",
            (void*)d_cluster_size, (size_t)n_total_cluster * sizeof(int));
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_cluster_vectors=%p (expected size: %zu bytes)\n",
            (void*)d_cluster_vectors, (size_t)n_total_vectors * n_dim * sizeof(float));
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_cluster_centers=%p (expected size: %zu bytes)\n",
            (void*)d_cluster_centers, (size_t)n_total_cluster * n_dim * sizeof(float));
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_topk_dist=%p (expected size: %zu bytes)\n",
            (void*)d_topk_dist, (size_t)n_query * k * sizeof(float));
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_topk_index=%p (expected size: %zu bytes)\n",
            (void*)d_topk_index, (size_t)n_query * k * sizeof(int));
    
    /* 验证参数有效性 */
    if (!d_query_batch || !d_cluster_size || !d_cluster_vectors || 
        !d_cluster_centers || !d_topk_dist || !d_topk_index) {
        fprintf(stderr, "batch_search_pipeline_wrapper: 错误: GPU内存指针为NULL\n");
        return -1;
    }
    
    if (n_query <= 0 || n_dim <= 0 || n_total_cluster <= 0 || 
        n_total_vectors <= 0 || n_probes <= 0 || k <= 0) {
        fprintf(stderr, "batch_search_pipeline_wrapper: 错误: 参数值无效\n");
        return -1;
    }
    
    if (n_probes > n_total_cluster) {
        fprintf(stderr, "batch_search_pipeline_wrapper: 警告: n_probes(%d) > n_total_cluster(%d)\n",
                n_probes, n_total_cluster);
    }
    
    fprintf(stderr, "batch_search_pipeline_wrapper: ========== 调用 batch_search_pipeline ==========\n");
    
    try {
        batch_search_pipeline(d_query_batch, d_cluster_size, d_cluster_vectors, d_cluster_centers,
                             d_topk_dist, d_topk_index,
                             n_query, n_dim, n_total_cluster, n_total_vectors, n_probes, k);
        fprintf(stderr, "batch_search_pipeline_wrapper: 执行成功\n");
        return 0;
    } catch (const std::exception& e) {
        /* 记录异常信息到PostgreSQL日志 */
        fprintf(stderr, "batch_search_pipeline_wrapper: 捕获到异常 - %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "batch_search_pipeline_wrapper: 捕获到未知异常\n");
        return -1;
    }
}
}

