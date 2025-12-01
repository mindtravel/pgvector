#include <stdexcept>
#include <cstdio>
#include <cstring>
#include "integrate_screen.cuh"

extern "C" {
/**
 * C包装函数，用于从C代码调用batch_search_pipeline
 * 处理C++异常并返回错误码
 */
int batch_search_pipeline_wrapper(float** query_batch, int* cluster_size,
                                  float*** cluster_vectors, float** cluster_center_data,
                                  float** topk_dist, int** topk_index, int* n_isnull,
                                  int n_query, int n_dim, int n_total_cluster,
                                  int n_total_vectors, int n_probes, int k)
{
    /* 记录关键参数信息到PostgreSQL日志 */
    fprintf(stderr, "batch_search_pipeline_wrapper: 开始执行, n_query=%d, n_dim=%d, n_total_cluster=%d, n_total_vectors=%d, n_probes=%d, k=%d\n",
            n_query, n_dim, n_total_cluster, n_total_vectors, n_probes, k);
    
    try {
        batch_search_pipeline(query_batch, cluster_size, cluster_vectors, cluster_center_data,
                             topk_dist, topk_index, n_isnull,
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

