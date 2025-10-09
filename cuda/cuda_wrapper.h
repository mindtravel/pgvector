#ifdef __cplusplus
extern "C" {
#endif

extern int cuda_hello_world(void);
/*
* 测试cuda是否可用
*/ 
extern bool cuda_is_available(void);

/*
* CUDA基本功能测试
*/
extern bool cuda_basic_test(void);

/*
* GPU向量搜索相关接口
*/ 
extern bool gpu_ivf_search_init(void);
extern void gpu_ivf_search_cleanup(void);
/*
* gpu 批查询函数，先保留
*/
extern int gpu_ivf_search_l2_batch(
    float* query_vector,           // 查询向量
    float* list_vectors,           // 列表向量数据
    int* list_offsets,             // 每个列表的偏移量
    int* list_counts,              // 每个列表的向量数量
    int num_lists,                 // 列表数量
    int vector_dim,                // 向量维度
    float* distances,              // 输出距离
    int* indices,                  // 输出索引
    int k                          // 返回前k个结果
);
/*
* gpu 批余弦距离查询函数
*/
extern int gpu_ivf_search_cosine_batch(
    float* query_vector,           // 查询向量
    float* list_vectors,           // 列表向量数据
    int* list_offsets,             // 每个列表的偏移量
    int* list_counts,              // 每个列表的向量数量
    int num_lists,                 // 列表数量
    int vector_dim,                // 向量维度
    float* distances,              // 输出距离
    int* indices,                  // 输出索引
    int k                          // 返回前k个结果
);

// GPU聚类中心距离计算相关结构体和函数
typedef struct {
    float* d_centers;        // GPU上的聚类中心数据
    float* d_query_vector;   // GPU上的查询向量
    float* d_distances;      // GPU上的距离结果
    float* h_centers_pinned; // 页锁定主机内存（零拷贝用）
    int num_centers;         // 聚类中心数量
    int dimensions;          // 向量维度
    bool initialized;        // 是否已初始化
    bool use_zero_copy;      // 是否使用零拷贝
} CudaCenterSearchContext;

// 函数声明
extern CudaCenterSearchContext* cuda_center_search_init(int num_centers, int dimensions, bool use_zero_copy);
extern void cuda_center_search_cleanup(CudaCenterSearchContext* ctx);
extern int cuda_compute_center_distances(CudaCenterSearchContext* ctx, 
                                        const float* query_vector, 
                                        float* distances);
extern int cuda_upload_centers(CudaCenterSearchContext* ctx, 
                              const float* centers_data);
extern int cuda_upload_centers_zero_copy(CudaCenterSearchContext* ctx, 
                                        const float* centers_data);
extern int cuda_set_zero_copy_mode(CudaCenterSearchContext* ctx, bool enable);

#ifdef __cplusplus
}
#endif