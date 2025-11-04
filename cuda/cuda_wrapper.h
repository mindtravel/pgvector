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
    
    // IndexTuple存储（与CPU端存储方式完全一致）
    // IndexTuple包含：t_info + t_tid + Vector数据
    char* d_probe_index_tuples; // GPU上的完整IndexTuple数据（按CPU方式存储）
    size_t index_tuple_size;    // 每个IndexTuple的大小（包括对齐），0表示变长
    int* d_probe_tuple_offsets; // 每个IndexTuple在连续内存中的偏移量（用于变长元组）
    int* d_probe_query_map;     // GPU上的查询映射（每个向量属于哪个查询）
    float* d_probe_distances;   // GPU上的距离数组
    int num_probe_candidates;   // 候选数量
    bool probes_uploaded;       // probes数据是否已上传
    
    // GPU TopK排序用的临时缓冲区
    int* d_topk_indices;        // GPU上的TopK索引
    float* d_topk_distances;    // GPU上的TopK距离
    
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

// Probe候选数据上传和处理相关函数（IndexTuple模式，与CPU端存储方式完全一致）
// 上传完整的IndexTuple数据到GPU（包含t_info + t_tid + Vector数据）
// index_tuples: 指向IndexTuple数组的指针（每个IndexTuple大小可能不同）
// tuple_sizes: 每个IndexTuple的大小数组（用于变长元组），如果为NULL且fixed_tuple_size>0则使用固定大小
// tuple_offsets: 每个IndexTuple在连续内存中的偏移量（可选），如果为NULL则假设连续
// fixed_tuple_size: 固定大小的IndexTuple大小，如果为0则使用tuple_sizes
extern int cuda_upload_probe_vectors(CudaCenterSearchContext* ctx,
                                     const char* index_tuples,
                                     const int* query_ids,
                                     const size_t* tuple_sizes,
                                     const int* tuple_offsets,
                                     int num_candidates,
                                     int dimensions,
                                     size_t fixed_tuple_size);
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

#ifdef __cplusplus
}
#endif