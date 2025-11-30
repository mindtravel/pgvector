#include <stdbool.h>
#include <stddef.h>  // for size_t

#ifdef __cplusplus
extern "C" {
#endif

/*
* 测试cuda是否可用
*/ 
extern bool cuda_is_available(void);

/*
* CUDA基本功能测试
*/
extern bool cuda_basic_test(void);

// GPU聚类中心距离计算相关结构体和函数
typedef struct {
    float* d_centers;        // GPU上的聚类中心数据
    float* d_query_vector;   // GPU上的查询向量
    float* d_batch_queries;  // GPU上的批量查询向量
    float* d_distances;      // GPU上的距离结果
    float* d_batch_distances; // GPU上的批量距离结果
    float* h_centers_pinned; // 页锁定主机内存（零拷贝用）
    
    // 分离存储方案：向量数据和TID数据分离存储，提高性能和内存访问效率
    // 向量数据：对齐存储，连续内存，提高访问效率
    float* d_probe_vectors;     // GPU上的向量数据（对齐存储，连续内存）
    char* d_probe_tids;         // GPU上的TID数据（ItemPointerData，每个6字节，连续存储）
    int* d_probe_query_map;     // GPU上的查询映射（每个向量属于哪个查询）
    float* d_probe_distances;   // GPU上的距离数组
    int num_probe_candidates;   // 候选数量
    bool probes_uploaded;       // probes数据是否已上传
    
    // GPU TopK排序用的临时缓冲区
    int* d_topk_indices;        // GPU上的TopK索引
    float* d_topk_distances;    // GPU上的TopK距离
    char* d_topk_tids;          // GPU上的TopK TID结果（批量提取）
    
    int num_centers;         // 聚类中心数量
    int dimensions;          // 向量维度
    int max_batch_size;      // 最大批量大小
    bool initialized;        // 是否已初始化
    bool use_zero_copy;      // 是否使用零拷贝
    bool batch_support;      // 是否支持批量处理
} CudaCenterSearchContext;

// 函数声明
extern CudaCenterSearchContext* cuda_center_search_init(int num_centers, int dimensions, bool use_zero_copy);
extern void cuda_center_search_cleanup(CudaCenterSearchContext* ctx);
extern int cuda_compute_center_distances(CudaCenterSearchContext* ctx, 
                                        const float* query_vector, 
                                        float* distances);
extern int cuda_compute_batch_center_distances(CudaCenterSearchContext* ctx,
                                             const float* batch_query_vectors,
                                             int num_queries,
                                             float* batch_distances);
extern int cuda_upload_centers(CudaCenterSearchContext* ctx, 
                              const float* centers_data);
extern int cuda_upload_centers_zero_copy(CudaCenterSearchContext* ctx, 
                                        const float* centers_data);
extern int cuda_set_zero_copy_mode(CudaCenterSearchContext* ctx, bool enable);

// Probe候选数据上传和处理相关函数
// 分离存储方案：向量数据和TID数据分离上传，提高性能
// vectors: 向量数据数组（连续存储，每个向量dimensions个float）
// tids: TID数据数组（ItemPointerData，每个6字节，连续存储）
// query_ids: 查询ID映射（每个候选属于哪个查询）
// 注意：vectors[i], tids[i], query_ids[i] 必须对应同一个候选，保证索引一致性
extern int cuda_upload_probe_vectors(CudaCenterSearchContext* ctx,
                                     const float* vectors,
                                     const char* tids,  /* ItemPointerData数组，每个6字节 */
                                     const int* query_ids,
                                     int num_candidates,
                                     int dimensions);
extern int cuda_compute_batch_probe_distances(CudaCenterSearchContext* ctx,
                                             const float* batch_query_vectors,
                                             int num_queries);
extern int cuda_topk_probe_candidates(CudaCenterSearchContext* ctx,
                                      int k,
                                      int num_queries,
                                      int* topk_query_ids,
                                      void* topk_vector_ids,  /* ItemPointerData数组 */
                                      float* topk_distances,
                                      int* topk_counts);
// 一致性检查函数
extern int cuda_verify_probe_consistency(CudaCenterSearchContext* ctx,
                                         int num_candidates,
                                         int dimensions);

// 获取最后一个CUDA错误的字符串（用于PostgreSQL错误报告）
extern const char* cuda_get_last_error_string(void);

#ifdef __cplusplus
}
#endif