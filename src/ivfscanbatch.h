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
    int* vector_ids;                 /* 所有向量ID连续存储 */
    float* distances;                /* 所有距离连续存储 */
    
    int n_queries;                  /* 查询数量 */
    int k;                          /* 每个查询的top-k */
    int total_results;              /* 总结果数 = n_queries * k */
    
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
    MemoryContext tmpCtx;            /* 内存上下文 */
    
    /* 批量查询数据 */
    ScanKeyBatch batch_keys;         /* 批量查询键 */
    
    /* 状态管理 */
    int current_query_index;         /* 当前查询索引 */
    bool batch_processing_complete; /* 批量处理是否完成 */
    
    BatchBuffer* result_buffer;         /* 结果缓冲区 */
    
#ifdef USE_CUDA
    /* GPU相关字段 */
    CudaCenterSearchContext* cuda_ctx;  /* CUDA上下文 */
    bool centers_uploaded;              /* 聚类中心是否已上传 */
    float* gpu_batch_distances;         /* GPU批量距离结果 */
#endif
} IvfflatBatchScanOpaqueData;

typedef IvfflatBatchScanOpaqueData* IvfflatBatchScanOpaque;

/* 公共函数声明 */
extern IndexScanDesc ivfflatbatchbeginscan(Relation index, int norderbys, ScanKeyBatch batch_keys);
extern bool ivfflatbatchgettuple(IndexScanDesc scan, ScanDirection dir, Datum* values, bool* isnull, int max_tuples, int* returned_tuples, int k);
extern void ivfflatbatchendscan(IndexScanDesc scan);

/* 批量处理函数声明 */
extern BatchBuffer* CreateBatchBuffer(int n_queries, int k, MemoryContext mem_ctx);
extern void ProcessBatchQueriesGPU(IndexScanDesc scan, ScanKeyBatch batch_keys, int k);
extern void GetBatchResults(BatchBuffer* buffer, int query_index, int k, Datum* values, bool* isnull, int* returned_count);



#endif /* IVFSCANBATCH_H */
