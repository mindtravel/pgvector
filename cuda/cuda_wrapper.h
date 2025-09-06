#ifdef __cplusplus
extern "C" {
#endif

extern int cuda_hello_world(void);
/*
* 测试cuda是否可用
*/ 
extern bool cuda_is_available(void);

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
#ifdef __cplusplus
}
#endif