#ifndef IVFSCANBATCH_H
#define IVFSCANBATCH_H

#include "postgres.h"
#include "access/genam.h"
#include "access/relscan.h"
#include "lib/pairingheap.h"
#include "nodes/execnodes.h"
#include "utils/tuplesort.h"
#include "ivfflat.h"
#include "scanbatch.h"

#ifdef USE_CUDA
#include "cuda/cuda_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 批处理大小控制 - 设为 64K 个向量，假设 1024 维 float，约 256MB，适合 PCIe 传输粒度 */
#define PIPELINE_CHUNK_SIZE 65536

/* 
 * IVF 索引上下文：管理常驻显存的数据 (Dataset & Clusters)
 * 使用 void* 隐藏 CUDA 类型，保持 C 兼容性
 */
typedef struct IVFIndexContext {
    /* 原始数据 */
    float* d_cluster_vectors;
    float* d_cluster_vector_norm;
    
    /* 聚类元数据 */
    int* d_probe_vector_offset;
    int* d_probe_vector_count;
    
    /* 聚类中心 */
    float* d_cluster_centers;
    float* d_cluster_centers_norm;
    
    /* 维度信息 */
    int n_total_clusters;
    int n_total_vectors;
    int n_dim;
    bool is_initialized;
} IVFIndexContext;

/* 
 * 查询批次上下文：管理每个Batch的临时资源和Stream
 * 用于支持双缓冲，每个Buffer对应一个Context
 */
typedef struct IVFQueryBatchContext {
    /* 资源流 (使用 void* 隐藏 CUDA 类型) */
    void* stream;                    /* cudaStream_t */
    void* data_ready_event;          /* cudaEvent_t */
    void* compute_done_event;         /* cudaEvent_t */
    
    /* Query 数据 */
    float* d_queries;
    float* d_query_norm;
    
    /* 中间计算结果 (Coarse Search) */
    float* d_inner_product;
    float* d_top_nprobe_dist;
    int* d_top_nprobe_index;
    int* d_index_seq;                /* 粗筛用的序列索引 */
    
    /* 细筛中间数据 (Entry Based) */
    int* d_cluster_query_count;
    int* d_cluster_query_offset;
    int* d_cluster_query_data;
    int* d_cluster_query_probe_indices;
    int* d_cluster_write_pos;
    int* d_entry_count_per_cluster;
    int* d_entry_offset;
    int* d_entry_query_offset;
    
    int* d_entry_cluster_id;
    int* d_entry_query_start;
    int* d_entry_query_count;
    int* d_entry_queries;
    int* d_entry_probe_indices;
    
    /* 最终结果 (Fine Search Candidate & Final) */
    float* d_topk_dist_candidate;
    int* d_topk_index_candidate;
    float* d_topk_dist;
    int* d_topk_index;
    
    /* 配置 */
    int max_n_query;
    int n_dim;
    int max_n_probes;
    int max_k;
    int n_total_clusters;
} IVFQueryBatchContext;

/* 注意：C 代码应该只调用 wrapper 函数，不要直接调用 CUDA 函数 */

/* Batch 相关函数的 C wrapper（带异常处理） */
extern void* ivf_create_batch_context_wrapper(int max_n_query, int n_dim, int max_n_probes, int max_k, int n_total_clusters);
extern void ivf_destroy_batch_context_wrapper(void* ctx_ptr);
extern void ivf_pipeline_stage1_prepare_wrapper(
    void* batch_ctx_ptr,
    float* query_batch_host,
    int n_query
);
extern void ivf_pipeline_stage2_compute_wrapper(
    void* batch_ctx_ptr,
    void* idx_ctx_ptr,
    int n_query,
    int n_probes,
    int k,
    int distance_mode
);
extern void ivf_pipeline_get_results_wrapper(
    void* batch_ctx_ptr,
    float* topk_dist,
    int* topk_index,
    int n_query,
    int k
);
extern void ivf_pipeline_sync_batch_wrapper(void* batch_ctx_ptr);

/* 流式上传接口的 C wrapper（带异常处理） */
extern void* ivf_create_index_context_wrapper(void);
extern void ivf_destroy_index_context_wrapper(void* ctx_ptr);
extern int ivf_check_index_initialized_wrapper(void* idx_ctx_ptr);
extern void ivf_init_streaming_upload_wrapper(
    void* idx_ctx_ptr,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
);
extern void ivf_append_cluster_data_wrapper(
    void* idx_ctx_ptr,
    int cluster_id,
    float* host_vector_data,
    int count,
    int start_offset_idx
);
extern void ivf_finalize_streaming_upload_wrapper(
    void* idx_ctx_ptr,
    float* center_data_flat,
    int total_vectors_check
);

/* 索引句柄注册表接口 */
extern void ivf_register_index_instance(Oid index_oid, void* gpu_handle);
extern void* ivf_get_index_instance(Oid index_oid);
extern void ivf_unregister_index_instance(Oid index_oid);

#ifdef __cplusplus
}
#endif
#endif

/*
 * 排序配置结构
 */
typedef struct {
    bool need_sorting;              /* 是否需要排序 */
    int sort_direction;             /* 排序方向 */
    int top_k;                      /* 返回前k个结果 */
} SortConfig;

/*
 * 批量结果缓冲区 - 优化内存布局
 * 数据按列存储：所有query_id连续，所有vector_id连续，所有distance连续
 */
typedef struct {
    float* query_data;              /* 查询向量数据 */
    
    /* 结果数据 - 按列存储以提高缓存效率 */
    int* query_ids;                  /* 所有查询ID连续存储 */
    ItemPointerData* vector_ids;     /* 所有向量TID连续存储（ItemPointer） */
    int* global_vector_indices;      /* 所有全局向量索引连续存储（用于返回给用户） */
    float* distances;                /* 所有距离连续存储 */
    
    int n_queries;                  /* 查询数量 */
    int k;                          /* 每个查询的top-k */
    
    MemoryContext mem_ctx;          /* 内存上下文 */
} BatchBuffer;


/*
 * 批量扫描状态结构
 */
typedef struct IvfflatBatchScanOpaqueData
{
    /* 基础信息 - 只保留GPU批量查询需要的字段 */
    const IvfflatTypeInfo *typeInfo;  /* 类型信息 */
    int dimensions;                   /* 向量维度 */
    
    /* 批量查询数据 */
    ScanKeyBatch batch_keys;         /* 批量查询键 */
    
    /* 状态管理 */
    int current_query_index;         /* 当前查询索引 */
    bool batch_processing_complete; /* 批量处理是否完成 */
    
    BatchBuffer* result_buffer;         /* 结果缓冲区 */
    int maxProbes;                   /* 最大probe数量 */
    int probes;                      /* 实际probe数量 */

#ifdef USE_CUDA
    /* Lists相关字段（用于probe选择） */
    pairingheap *listQueue;          /* 列表优先级队列 */
    BlockNumber *listPages;          /* 每个查询选定的列表页面 */
    IvfflatScanList *lists;          /* 扫描列表数组 */

    int listIndex;                   /* 当前列表索引 */
#endif
} IvfflatBatchScanOpaqueData;

typedef IvfflatBatchScanOpaqueData* IvfflatBatchScanOpaque;



/* 公共函数声明 */
extern IndexScanDesc ivfflatbatchbeginscan(Relation index, int norderbys, ScanKeyBatch batch_keys);
// extern bool ivfflatbatchgettuple(IndexScanDesc scan, ScanDirection dir, Datum* values, bool* isnull, int k);
extern bool ivfflatbatchgettuple(IndexScanDesc scan, ScanDirection dir, Tuplestorestate *tupstore, TupleDesc tupdesc, int k, int distance_mode);
extern void ivfflatbatchendscan(IndexScanDesc scan);

/* 批量处理函数声明 */
extern BatchBuffer* CreateBatchBuffer(int n_queries, int k, int dimensions, MemoryContext mem_ctx);
extern void ProcessBatchQueriesGPU(IndexScanDesc scan, ScanKeyBatch batch_keys, int k, int distance_mode);
extern void GetBatchResults(BatchBuffer* buffer, Tuplestorestate *tupstore, TupleDesc tupdesc);



#endif /* IVFSCANBATCH_H */
