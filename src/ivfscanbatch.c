#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/relscan.h"
#include "access/tableam.h"
#include "access/heapam.h"
#include "access/sysattr.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "utils/datum.h"
#include "utils/snapshot.h"
#include "fmgr.h"
#include "ivfscanbatch.h"
#include "scanbatch.h"
#include "ivfflat.h"
#include "vector.h"

/* 内部函数声明 */

#ifdef USE_CUDA
/* CUDA 运行时函数的前向声明（用于 C 代码） */
/* 注意：cuda_wrapper.h 已在 ivfscanbatch.h 中包含 */
/* 注意：这些函数由 CUDA 运行时库提供，链接时通过 -lcudart 解析 */
typedef enum cudaError_enum {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInvalidValue = 11
} cudaError_t;

extern cudaError_t cudaMalloc(void **devPtr, size_t size);
extern cudaError_t cudaFree(void *devPtr);
extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind);
extern const char* cudaGetErrorString(cudaError_t error);

#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2

#define GetScanList(ptr) pairingheap_container(IvfflatScanList, ph_node, ptr)
#define GetScanListConst(ptr) pairingheap_const_container(IvfflatScanList, ph_node, ptr)

static int CompareLists(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
    if (GetScanListConst(a)->distance > GetScanListConst(b)->distance)
        return 1;
    if (GetScanListConst(a)->distance < GetScanListConst(b)->distance)
        return -1;
    return 0;
}

static void GetScanLists_BatchGPU(IndexScanDesc scan, ScanKeyBatch batch_keys);
static int UploadCentersToGPU_Batch(IndexScanDesc scan);
static int UploadIndexTuplesToGPU_Batch(IndexScanDesc scan);
/** @deprecated 已废弃，请使用 ProcessBatchQueriesGPU */
static void GetScanItems_BatchGPU(IndexScanDesc scan, ScanKeyBatch batch_keys);

/* batch_search_pipeline相关函数 */
typedef struct {
    int cluster_id;
    int vector_index_in_cluster;
    ItemPointerData tid;
    int table_row_id;  /* 表行ID（id字段），如果索引不包含则从堆表读取 */
} VectorIndexMapping;

static int PrepareBatchDataForPipeline(IndexScanDesc scan, ScanKeyBatch batch_keys, int n_probes,
                                       float*** query_batch_out, int** cluster_size_out,
                                       float**** cluster_vectors_out, float*** cluster_center_data_out,
                                       VectorIndexMapping*** mapping_table_out,
                                       BlockNumber** cluster_pages_out,
                                       int* n_total_clusters_out, int* n_total_vectors_out);
static void ConvertBatchPipelineResults(IndexScanDesc scan, float** topk_dist, int** topk_index,
                                        int n_query, int k, BatchBuffer* result_buffer,
                                        VectorIndexMapping* mapping_table, int* cluster_size,
                                        BlockNumber* cluster_pages, int n_total_clusters);
// static void ProcessBatchQueriesGPU(IndexScanDesc scan, ScanKeyBatch batch_keys, int k);
#endif

/*
 * 批量扫描开始函数
 */
IndexScanDesc
ivfflatbatchbeginscan(Relation index, int norderbys, ScanKeyBatch batch_keys)
{
    IndexScanDesc scan;
    IvfflatBatchScanOpaque so;
    int lists;
    int dimensions;

    // elog(LOG, "ivfflatbatchbeginscan: 开始批量扫描, nkeys=%d", batch_keys->nkeys);

    /* 创建扫描描述符 */
    scan = RelationGetIndexScan(index, batch_keys->nkeys, norderbys);
    if (!scan) {
        elog(ERROR, "ivfflatbatchbeginscan: 无法创建扫描描述符");
    }
    
    /* 在当前内存上下文（SRF内存上下文）中分配批量扫描状态 */
    so = (IvfflatBatchScanOpaque)palloc0(sizeof(IvfflatBatchScanOpaqueData));
    if (!so) {
        elog(ERROR, "ivfflatbatchbeginscan: 无法分配批量扫描状态内存");
    }
    
    /* 初始化基础信息 */
    so->typeInfo = IvfflatGetTypeInfo(index);
    
    /* 从元页面获取维度和列表数 */
    lists = 0;
    dimensions = 0;

    IvfflatGetMetaPageInfo(index, &lists, &dimensions);
    so->dimensions = dimensions;
    
    /* 设置批量查询数据 */
    so->batch_keys = batch_keys;
    so->current_query_index = 0;
    so->batch_processing_complete = false;
    so->result_buffer = NULL;
    
#ifdef USE_CUDA
    /* 初始化GPU相关字段 */
    // so->centers_uploaded = false;
    // so->cuda_ctx = NULL;
    // so->gpu_batch_distances = NULL;
    
    /* 检查CUDA是否可用 */
    if (!cuda_is_available()) {
        elog(ERROR, "批量向量搜索需要GPU支持，但CUDA不可用");
    }
    
    /* 设置probe相关参数 */
    int probes = ivfflat_probes;  /* 从GUC参数获取 */
    int maxProbes = Max(ivfflat_max_probes, probes);
    if (probes > lists) probes = lists;
    if (maxProbes > lists) maxProbes = lists;
    
    /* 修复：对于批量查询，maxProbes应该等于probes，避免全表扫描 */
    maxProbes = probes;
    
    // elog(LOG, "ivfflatbatchbeginscan: probe配置 - lists=%d, probes=%d, maxProbes=%d, ivfflat_max_probes=%d", 
    //      lists, probes, maxProbes, ivfflat_max_probes);
    
    so->probes = probes;
    so->maxProbes = maxProbes;
    so->listIndex = 0;
    
    /* 初始化列表相关结构 */
    so->listQueue = pairingheap_allocate(CompareLists, scan);
    so->listPages = (BlockNumber*)palloc(batch_keys->nkeys * maxProbes * sizeof(BlockNumber));
    so->lists = (IvfflatScanList*)palloc(maxProbes * sizeof(IvfflatScanList));
    
    /* 初始化CUDA上下文 */
    // 注释掉：现在使用 integrate_screen 中的显存管理，不需要 cuda_center_search_init
    // so->cuda_ctx = cuda_center_search_init(lists, dimensions, false);
    so->cuda_ctx = NULL;  // 设置为 NULL，避免后续检查失败
    // if (!so->cuda_ctx) {
    //     elog(WARNING, "CUDA上下文初始化失败，将使用CPU模式");
    // } else {
    //     /* 分配GPU批量距离结果内存 */
    //     so->gpu_batch_distances = (float*)palloc(batch_keys->nkeys * lists * sizeof(float));
    // }
#endif

    /* 设置scan->opaque */
    scan->opaque = so;

    // elog(LOG, "ivfflatbatchbeginscan: 批量扫描初始化完成");
    return scan;
}

bool
ivfflatbatchgettuple(IndexScanDesc scan, ScanDirection dir, Datum* values, bool* isnull, int k)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    int nbatch;

    if (!so->batch_processing_complete) {
        ProcessBatchQueriesGPU(scan, so->batch_keys, k);
        so->batch_processing_complete = true;
    }
    
    if (!so->result_buffer) {
        elog(ERROR, "ivfflatbatchgettuple: result_buffer为空！");
    }
    
    nbatch = so->batch_keys->nkeys;
    
    /* 验证 result_buffer 中的查询数量是否匹配 */
    if (so->result_buffer->n_queries != nbatch) {
        elog(ERROR, "ivfflatbatchgettuple: result_buffer中的查询数量(%d)与batch_keys中的查询数量(%d)不匹配！"
                "这可能是因为扫描描述符被重用但result_buffer未重置。",
                so->result_buffer->n_queries, nbatch);
        return false;
    }
    
    elog(LOG, "ivfflatbatchgettuple: 开始处理结果, nbatch=%d, k=%d", 
            nbatch, k);
    
    GetBatchResults(so->result_buffer, values, isnull);
    
    elog(LOG, "ivfflatbatchgettuple: 完成");
    return true;
}

void
ivfflatbatchendscan(IndexScanDesc scan)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    
    if (!so) {
        /* 如果opaque已经被清理，直接返回 */
        return;
    }

#ifdef USE_CUDA
    /* 清理CUDA资源（如果使用了） */
    // 注释掉：现在使用 integrate_screen 中的显存管理，不需要 cuda_center_search_cleanup
    // if (so->cuda_ctx) {
    //     cuda_center_search_cleanup((CudaCenterSearchContext*)so->cuda_ctx);
    //     so->cuda_ctx = NULL;  // 避免重复清理
    // }
#endif
    
    /* 将scan->opaque设置为NULL，避免后续访问已删除的内存 */
    scan->opaque = NULL;
    
    /* 注意：so、batch_keys、result_buffer 等都在 SRF 内存上下文中分配，
     * 会在 SRF 结束时自动清理，无需手动释放 */
}

void
GetBatchResults(BatchBuffer* buffer, Datum* values, bool* isnull)
{
    int n_queries;
    int k;
    int result_idx;

    if (buffer == NULL) {
        elog(LOG, "GetBatchResults: buffer 为 NULL");
        return;
    }
    
    /* 验证 buffer 的字段 */
    if (buffer->query_ids == NULL) {
        elog(ERROR, "GetBatchResults: buffer->query_ids 为 NULL");
        return;
    }
    if (buffer->vector_ids == NULL) {
        elog(ERROR, "GetBatchResults: buffer->vector_ids 为 NULL");
        return;
    }
    
    if (buffer->global_vector_indices == NULL) {
        elog(ERROR, "GetBatchResults: buffer->global_vector_indices 为 NULL");
        return;
    }
    
    if (buffer->distances == NULL) {
        elog(ERROR, "GetBatchResults: buffer->distances 为 NULL");
        return;
    }
    
    n_queries = buffer->n_queries;
    k = buffer->k;
    
    elog(LOG, "GetBatchResults: buffer 字段验证通过, 开始遍历结果, n_queries=%d, k=%d", n_queries, k);
    
    /* 两重循环：外层遍历所有查询，内层遍历每个查询的k个结果 */
    result_idx = 0;
    for (int query_idx = 0; query_idx < n_queries; query_idx++) {
        elog(LOG, "GetBatchResults: 处理查询 %d", query_idx);
        
        for (int k_idx = 0; k_idx < k; k_idx++) {
            /* 计算在buffer中的索引：按列存储，每个查询有k个槽位 */
            int buffer_idx = query_idx * k + k_idx;
            
            /* 计算在输出数组中的索引：每个结果有3个字段（query_id, vector_id, distance） */
            int output_base_idx = result_idx * 3;
            
            /* 检查是否是null值（使用global_vector_indices检查，-1表示无效） */
            bool is_null = (buffer->global_vector_indices[buffer_idx] < 0);
            
            if (is_null) {
                /* null值：设置所有字段为null
                 * 注意：在PostgreSQL中，当isnull=true时，value的值不会被使用
                 * 但我们使用(Datum) 0作为标准约定，这是PostgreSQL的NULL Datum值
                 */
                values[output_base_idx + 0] = (Datum) 0;  /* query_id: NULL */
                values[output_base_idx + 1] = (Datum) 0;  /* vector_id: NULL */
                values[output_base_idx + 2] = (Datum) 0;  /* distance: NULL */
                
                isnull[output_base_idx + 0] = true;
                isnull[output_base_idx + 1] = true;
                isnull[output_base_idx + 2] = true;
            } else {
                /* 有效值：设置返回值 */
                values[output_base_idx + 0] = Int32GetDatum(buffer->query_ids[buffer_idx]); /* query_id */
                /* vector_id: 返回表行ID（存储在global_vector_indices中） */
                {
                    int table_row_id = buffer->global_vector_indices[buffer_idx];
                    elog(LOG, "GetBatchResults: 查询 %d, 结果 %d, buffer_idx=%d, table_row_id=%d", 
                         query_idx, k_idx, buffer_idx, table_row_id);
                    values[output_base_idx + 1] = Int32GetDatum(table_row_id); /* vector_id: 表行ID */
                }
                elog(LOG, "GetBatchResults: 查询 %d, 结果 %d, distance=%.6f", 
                     query_idx, k_idx, buffer->distances[buffer_idx]);
                values[output_base_idx + 2] = Float8GetDatum(buffer->distances[buffer_idx]); /* distance */
                
                /* 设置非空标志 */
                isnull[output_base_idx + 0] = false;
                isnull[output_base_idx + 1] = false;
                isnull[output_base_idx + 2] = false;
            }
            
            result_idx++;
        }
    }
    
    elog(LOG, "GetBatchResults: 完成, 总结果数=%d (n_queries=%d * k=%d)", result_idx, n_queries, k);
}


BatchBuffer*
CreateBatchBuffer(int n_queries, int k, int dimensions, MemoryContext ctx)
{
    
    int total_results;
    total_results = n_queries * k;
    BatchBuffer* buffer;
    buffer = (BatchBuffer*)palloc0(sizeof(BatchBuffer));
    
    /* 分配内存 - 按列存储，在当前内存上下文中分配 */
    buffer->query_data = (float*)palloc(n_queries * dimensions * sizeof(float));
    buffer->query_ids = (int*)palloc(total_results * sizeof(int));
    buffer->vector_ids = (ItemPointerData*)palloc(total_results * sizeof(ItemPointerData));
    buffer->global_vector_indices = (int*)palloc(total_results * sizeof(int));
    buffer->distances = (float*)palloc(total_results * sizeof(float));
    
    buffer->n_queries = n_queries;
    buffer->k = k;
    buffer->mem_ctx = ctx;
    
    return buffer;
}


#ifdef USE_CUDA
static void
GetScanLists_BatchGPU(IndexScanDesc scan, ScanKeyBatch batch_keys)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    
    /* 上传聚类中心数据到GPU（如果尚未上传） */
    if (!so->centers_uploaded && UploadCentersToGPU_Batch(scan) != 0) {
        elog(ERROR, "无法上传聚类中心数据到GPU");
        return;
    }
    
    /* 直接从元页面获取聚类中心总数 */
    int totalLists = 0;
    IvfflatGetMetaPageInfo(scan->indexRelation, &totalLists, NULL);
    
    BlockNumber *list_pages = palloc(totalLists * sizeof(BlockNumber));
    if (!list_pages) {
        elog(ERROR, "无法分配列表页面内存");
        return;
    }
    
    /* 收集列表页面信息 */
    BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
    int center_idx = 0;
    
    while (BlockNumberIsValid(nextblkno))
    {
        Buffer cbuf = ReadBuffer(scan->indexRelation, nextblkno);
        LockBuffer(cbuf, BUFFER_LOCK_SHARE);
        Page cpage = BufferGetPage(cbuf);
        OffsetNumber maxoffno = PageGetMaxOffsetNumber(cpage);

        for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
        {
            if (center_idx >= totalLists) {
                UnlockReleaseBuffer(cbuf);
                elog(ERROR, "列表索引超出范围");
                pfree(list_pages);
                return;
            }
            
            IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
            list_pages[center_idx++] = list->startPage;
        }

        nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;
        UnlockReleaseBuffer(cbuf);
    }
    
    /* 准备批量查询数据 - 使用与probe距离计算相同的方式 */
    int nbatch = batch_keys->nkeys;
    float *batch_query_data = (float*)palloc(nbatch * so->dimensions * sizeof(float));
    if (!batch_query_data) {
        elog(ERROR, "无法分配聚类中心距离计算的查询数据内存");
        pfree(list_pages);
        return;
    }
    
    /* 手动提取每个查询向量，确保与probe距离计算的顺序完全一致 */
    for (int i = 0; i < nbatch; i++) {
        Datum vector_datum = ScanKeyBatchGetVector(batch_keys, i);
        Vector *vec = DatumGetVector(vector_datum);
        /* 复制向量数据到连续数组 */
        memcpy(batch_query_data + i * so->dimensions, &vec->x[0], so->dimensions * sizeof(float));
        
        /* 调试：输出查询向量的前几个元素 */
        // elog(LOG, "GetScanLists_BatchGPU: 查询%d向量前3个元素: [%.6f, %.6f, %.6f] (聚类中心距离计算)", 
        //      i, vec->x[0], vec->x[1], vec->x[2]);
    }
    
    /* GPU批量计算距离 */
    int cuda_result = cuda_compute_batch_center_distances(
        so->cuda_ctx, 
        batch_query_data, 
        nbatch, 
        so->gpu_batch_distances);
    
    if (cuda_result != 0) {
        elog(ERROR, "GPU距离计算失败");
        pfree(batch_query_data);
        pfree(list_pages);
        return;
    }
    
    /* 为每个查询处理GPU计算结果 */
    for (int query_idx = 0; query_idx < nbatch; query_idx++) {
        int listCount = 0;
        double maxDistance = DBL_MAX;
        
        pairingheap_reset(so->listQueue);
        
        for (int i = 0; i < totalLists; i++) {
            double distance = so->gpu_batch_distances[query_idx * totalLists + i];
            
            if (listCount < so->maxProbes) {
                IvfflatScanList *scanlist = &so->lists[listCount];
                scanlist->startPage = list_pages[i];
                scanlist->distance = distance;
                listCount++;
                
                pairingheap_add(so->listQueue, &scanlist->ph_node);
                
                if (listCount == so->maxProbes)
                    maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
            }
            else if (distance < maxDistance) {
                IvfflatScanList *scanlist = GetScanList(pairingheap_remove_first(so->listQueue));
                
                scanlist->startPage = list_pages[i];
                scanlist->distance = distance;
                pairingheap_add(so->listQueue, &scanlist->ph_node);
                
                maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
            }
        }
        
        /* 输出排序结果 */
        int outputCount = Min(listCount, so->maxProbes);
        // elog(LOG, "GetScanLists_BatchGPU: 查询%d选择的probe列表:", query_idx);
        for (int i = outputCount - 1; i >= 0; i--) {
            IvfflatScanList *scanlist = GetScanList(pairingheap_remove_first(so->listQueue));
            so->listPages[query_idx * so->maxProbes + i] = scanlist->startPage;
            // elog(LOG, "  probe %d: 页面=%u, 距离=%.6f", i, scanlist->startPage, scanlist->distance);
        }
        
        Assert(pairingheap_is_empty(so->listQueue));
    }
    
    /* 释放手动分配的内存 */
    pfree(batch_query_data);
    pfree(list_pages);
}

// /**
//  * @deprecated 此函数已废弃，请使用 ProcessBatchQueriesGPU
//  * 
//  * 旧GPU路径：使用 cuda_compute_batch_probe_distances 和 cuda_topk_probe_candidates
//  * - 依赖 cuda_ctx (CUDA上下文)
//  * - 返回 TID (ItemPointerData)
//  * 
//  * 新GPU路径：使用 batch_search_pipeline_wrapper (ProcessBatchQueriesGPU)
//  * - 不依赖 cuda_ctx
//  * - 返回全局向量索引
//  * 
//  * 当前所有调用都通过 ProcessBatchQueriesGPU -> ProcessBatchQueriesGPU
//  */
// void
// GetScanItems_BatchGPU(IndexScanDesc scan, ScanKeyBatch batch_keys)
// {
//     IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
//     int nbatch = batch_keys->nkeys;
//     int k = so->result_buffer ? so->result_buffer->k : 10;  /* 默认top-k */
    
//     if (!so->cuda_ctx || !so->cuda_ctx->initialized) {
//         elog(ERROR, "GetScanItems_BatchGPU: CUDA上下文未初始化");
//         return;
//     }
    
//     /* 确保probes数据已上传 */
//     if (!so->cuda_ctx->probes_uploaded) {
//         // elog(LOG, "GetScanItems_BatchGPU: probes数据未上传，开始上传（分离存储方案）");
//         if (UploadIndexTuplesToGPU_Batch(scan) != 0) {
//             elog(ERROR, "GetScanItems_BatchGPU: probes数据上传失败");
//             return;
//         }
//     }
    
//     /* 确保result_buffer已创建 */
//     if (so->result_buffer == NULL) {
//         so->result_buffer = CreateBatchBuffer(nbatch, k, so->dimensions, so->tmpCtx);
//     }
    
//     /* 步骤1: GPU批量计算距离 */
//     /* 注意：向量是独立存储的，需要逐个提取并打包成连续数组 */
//     float *batch_query_data = (float*)palloc(nbatch * so->dimensions * sizeof(float));
//     if (!batch_query_data) {
//         elog(ERROR, "GetScanItems_BatchGPU: 无法分配批量查询数据内存");
//         return;
//     }
    
//     /* 提取每个查询向量的数据部分，打包成连续数组 */
//     for (int i = 0; i < nbatch; i++) {
//         Datum vector_datum = ScanKeyBatchGetVector(batch_keys, i);
//         Vector *vec = DatumGetVector(vector_datum);
//         /* 复制向量数据到连续数组 */
//         memcpy(batch_query_data + i * so->dimensions, &vec->x[0], so->dimensions * sizeof(float));
        
//         /* 调试：输出查询向量的前几个元素 */
//         // elog(LOG, "GetScanItems_BatchGPU: 查询%d向量前3个元素: [%.6f, %.6f, %.6f] (应该对应查询%d)", 
//         //      i, vec->x[0], vec->x[1], vec->x[2], i);
//     }
    
//     // elog(LOG, "GetScanItems_BatchGPU: 开始在GPU上计算批量probe距离");
//     int compute_result = cuda_compute_batch_probe_distances(so->cuda_ctx, 
//                                                            batch_query_data, 
//                                                            nbatch);
    
//     /* 释放临时内存 */
//     pfree(batch_query_data);
    
//     if (compute_result != 0) {
//         elog(ERROR, "GetScanItems_BatchGPU: GPU批量距离计算失败");
//         return;
//     }
    
//     /* 步骤2: GPU TopK选择 */
//     int *topk_query_ids = (int*)palloc(nbatch * k * sizeof(int));
//     ItemPointerData *topk_vector_ids = (ItemPointerData*)palloc(nbatch * k * sizeof(ItemPointerData));
//     float *topk_distances = (float*)palloc(nbatch * k * sizeof(float));
//     int *topk_counts = (int*)palloc0(nbatch * sizeof(int));
    
//     if (!topk_query_ids || !topk_vector_ids || !topk_distances || !topk_counts) {
//         elog(ERROR, "GetScanItems_BatchGPU: 无法分配TopK结果内存");
//         if (topk_query_ids) pfree(topk_query_ids);
//         if (topk_vector_ids) pfree(topk_vector_ids);
//         if (topk_distances) pfree(topk_distances);
//         if (topk_counts) pfree(topk_counts);
//         return;
//     }
    
//     // elog(LOG, "GetScanItems_BatchGPU: 开始在GPU上进行TopK选择 - k=%d, nbatch=%d, 候选数=%d", 
//     //      k, nbatch, so->cuda_ctx ? so->cuda_ctx->num_probe_candidates : -1);
//     int topk_result = cuda_topk_probe_candidates(so->cuda_ctx, 
//                                                  k, 
//                                                  nbatch,
//                                                  topk_query_ids, 
//                                                  topk_vector_ids, 
//                                                  topk_distances, 
//                                                  topk_counts);
    
//     if (topk_result != 0) {
// #ifdef USE_CUDA
//         const char* cuda_err = cuda_get_last_error_string();
//         elog(ERROR, "GetScanItems_BatchGPU: GPU TopK选择失败 - k=%d, nbatch=%d, 候选数=%d, 维度=%d, CUDA错误: %s", 
//              k, nbatch, 
//              so->cuda_ctx ? so->cuda_ctx->num_probe_candidates : -1,
//              so->dimensions,
//              cuda_err);
// #else
//         elog(ERROR, "GetScanItems_BatchGPU: GPU TopK选择失败 - k=%d, nbatch=%d, 候选数=%d, 维度=%d", 
//              k, nbatch, 
//              so->cuda_ctx ? so->cuda_ctx->num_probe_candidates : -1,
//              so->dimensions);
// #endif
//         pfree(topk_query_ids);
//         pfree(topk_vector_ids);
//         pfree(topk_distances);
//         pfree(topk_counts);
//         return;
//     }
    
//     /* 步骤3: 填充到result_buffer（按列存储） */
//     /* 每个查询必须填充k个位置，如果候选不足k个，用null值填充 */
//     int total_processed_results = 0;  /* 累计已处理的GPU结果数 */
//     int total_results = 0;            /* 总的缓冲区槽位数 */
//     for (int query_idx = 0; query_idx < nbatch; query_idx++) {
//         int count = topk_counts[query_idx];
        
//         /* 填充实际结果（如果有的话） */
//         for (int i = 0; i < count; i++) {
//             int buffer_idx = query_idx * k + i;           /* 目标缓冲区位置 */
//             int source_idx = total_processed_results + i; /* GPU结果数组中的源位置 */
            
//             so->result_buffer->query_ids[buffer_idx] = topk_query_ids[source_idx];
//             so->result_buffer->vector_ids[buffer_idx] = topk_vector_ids[source_idx];
//             /* 注意：GetScanItems_BatchGPU 使用旧的GPU路径，返回的是TID，不是全局向量索引
//              * 这里暂时设置为 -1，表示需要通过TID查找（但当前实现不支持）
//              * 或者可以尝试从TID推导出全局向量索引，但这需要额外的映射表
//              */
//             so->result_buffer->global_vector_indices[buffer_idx] = -1;  /* 旧路径暂不支持全局索引 */
//             so->result_buffer->distances[buffer_idx] = topk_distances[source_idx];
            
//             /* 调试日志：验证索引修复效果（仅前几个结果） */
//             // if (query_idx < 3 && i < 2) {
//             //     elog(LOG, "GetScanItems_BatchGPU: 查询%d结果%d - buffer_idx=%d, source_idx=%d, query_id=%d, distance=%.6f", 
//             //          query_idx, i, buffer_idx, source_idx, topk_query_ids[source_idx], topk_distances[source_idx]);
//             // }
//         }
        
//         /* 如果候选不足k个，用null值填充剩余位置 */
//         for (int i = count; i < k; i++) {
//             int buffer_idx = query_idx * k + i;
//             so->result_buffer->query_ids[buffer_idx] = query_idx;
//             ItemPointerSetInvalid(&so->result_buffer->vector_ids[buffer_idx]);  /* null标记：无效的TID */
//             so->result_buffer->distances[buffer_idx] = INFINITY;  /* null标记：无效的距离 */
//         }
        
//         total_processed_results += count;  /* 累计已处理的GPU结果数 */
//         total_results += k;                /* 每个查询贡献k个槽位 */
//     }
        
//     pfree(topk_query_ids);
//     pfree(topk_vector_ids);
//     pfree(topk_distances);
//     pfree(topk_counts);
    
//     // elog(LOG, "GetScanItems_BatchGPU: 成功获取 %d 个结果（GPU计算距离和TopK）", total_results);
// }

static int
UploadCentersToGPU_Batch(IndexScanDesc scan)
{
    if (!scan || !scan->opaque) {
        elog(ERROR, "UploadCentersToGPU_Batch: 扫描描述符无效");
        return -1;
    }

    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    
    /* 如果已经上传过，直接返回成功 */
    if (so->centers_uploaded) {
        return 0;
    }
    
    /* 检查CUDA上下文 */
    if (!so->cuda_ctx) {
        elog(ERROR, "UploadCentersToGPU_Batch: CUDA上下文为空");
        return -1;
    }
    
    /* 直接从元页面获取聚类中心总数 */
    int totalLists = 0;
    IvfflatGetMetaPageInfo(scan->indexRelation, &totalLists, NULL);
    
    if (totalLists <= 0) {
        elog(ERROR, "UploadCentersToGPU_Batch: 没有找到聚类中心数据");
        return -1;
    }
    
    int dimensions = so->dimensions;
    int center_idx = 0;
    BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
    
    /* 分配内存存储聚类中心数据 */
    float *centers_data = palloc(totalLists * dimensions * sizeof(float));
    if (!centers_data) {
        elog(ERROR, "UploadCentersToGPU_Batch: 无法分配内存 (%d个中心, %d维)", totalLists, dimensions);
        return -1;
    }
    
    /* 收集聚类中心数据 */
    while (BlockNumberIsValid(nextblkno))
    {
        Buffer cbuf;
        Page cpage;
        OffsetNumber maxoffno;

        cbuf = ReadBuffer(scan->indexRelation, nextblkno);
        LockBuffer(cbuf, BUFFER_LOCK_SHARE);
        cpage = BufferGetPage(cbuf);
        maxoffno = PageGetMaxOffsetNumber(cpage);

        for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
        {
            if (center_idx >= totalLists) {
                UnlockReleaseBuffer(cbuf);
                elog(ERROR, "UploadCentersToGPU_Batch: 聚类中心索引超出范围");
                pfree(centers_data);
                return -1;
            }
            
            IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
            Vector *center_vector = &list->center;
            float *center_data = &center_vector->x[0];
            
            memcpy(&centers_data[center_idx * dimensions], center_data, dimensions * sizeof(float));
            center_idx++;
        }

        nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;
        UnlockReleaseBuffer(cbuf);
    }
    
    /* 验证实际读取的中心数量 */
    if (center_idx != totalLists) {
        elog(ERROR, "UploadCentersToGPU_Batch: 实际读取的中心数量(%d)与预期(%d)不一致", center_idx, totalLists);
        pfree(centers_data);
        return -1;
    }
    
    elog(LOG, "UploadCentersToGPU_Batch: 成功读取%d个聚类中心，准备上传到GPU", center_idx);
    
    /* 上传到GPU */
    CudaCenterSearchContext* ctx = so->cuda_ctx;
    
    if (!ctx || !ctx->initialized) {
        elog(ERROR, "UploadCentersToGPU_Batch: CUDA上下文无效");
        pfree(centers_data);
        return -1;
    }
    
    int upload_result = (ctx->use_zero_copy) ? 
        cuda_upload_centers_zero_copy(ctx, centers_data) : 
        cuda_upload_centers(ctx, centers_data);
    
    pfree(centers_data);
    
    if (upload_result == 0) {
        so->centers_uploaded = true;
    } else {
        elog(ERROR, "UploadCentersToGPU_Batch: GPU上传失败");
    }
    
    return upload_result;
}

/*
 * 上传分离的向量数据和TID数据到GPU（分离存储方案，提高性能）
 * 关键：确保向量数据、TID数据和查询ID的索引一致性
 */
static int
UploadIndexTuplesToGPU_Batch(IndexScanDesc scan)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque) scan->opaque;
    ScanKeyBatch batch_keys = so->batch_keys;
    int nbatch = batch_keys->nkeys;
    int maxProbes = so->maxProbes;
    int dimensions = so->dimensions;
    int estimated_total_candidates = nbatch * maxProbes * 100;  // 估算值
    
    /* 分配临时数组收集分离的数据（方案1：分离存储） */
    float *packed_vectors = (float*)palloc(estimated_total_candidates * dimensions * sizeof(float));
    ItemPointerData *packed_tids = (ItemPointerData*)palloc(estimated_total_candidates * sizeof(ItemPointerData));
    int *query_ids = (int*)palloc(estimated_total_candidates * sizeof(int));
    
    if (!packed_vectors || !packed_tids || !query_ids) {
        elog(ERROR, "UploadIndexTuplesToGPU_Batch: 无法分配分离数据内存");
        if (packed_vectors) pfree(packed_vectors);
        if (packed_tids) pfree(packed_tids);
        if (query_ids) pfree(query_ids);
        return -1;
    }
    
    int total_candidates = 0;
    TupleDesc tupdesc = RelationGetDescr(scan->indexRelation);
    
    /* 遍历批量查询中的每个查询向量 */
    for (int query_idx = 0; query_idx < nbatch; query_idx++) {
        /* 遍历每个查询选定的列表 */
        for (int probe_idx = 0; probe_idx < maxProbes; probe_idx++) {
            BlockNumber list_page = so->listPages[query_idx * maxProbes + probe_idx];
            
            if (!BlockNumberIsValid(list_page)) {
                continue;  /* 无效的列表页面，跳过 */
            }
            
            /* 搜索该list的所有页面 */
            BlockNumber searchPage = list_page;
            while (BlockNumberIsValid(searchPage)) {
                Buffer buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, 
                                              searchPage, RBM_NORMAL, NULL);
                LockBuffer(buf, BUFFER_LOCK_SHARE);
                Page page = BufferGetPage(buf);
                OffsetNumber maxoffno = PageGetMaxOffsetNumber(page);
                
                /* 遍历页面中的所有向量项 */
                for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno)) {
                    IndexTuple itup = (IndexTuple) PageGetItem(page, PageGetItemId(page, offno));
                    
                    /* 获取向量数据以检查是否为null */
                    Datum vector_datum;
                    bool isnull;
                    vector_datum = index_getattr(itup, 1, tupdesc, &isnull);
                    
                    if (isnull) {
                        continue;
                    }
                    
                    /* 动态扩容检查 */
                    if (total_candidates >= estimated_total_candidates) {
                        estimated_total_candidates *= 2;
                        packed_vectors = (float*)repalloc(packed_vectors, estimated_total_candidates * dimensions * sizeof(float));
                        packed_tids = (ItemPointerData*)repalloc(packed_tids, estimated_total_candidates * sizeof(ItemPointerData));
                        query_ids = (int*)repalloc(query_ids, estimated_total_candidates * sizeof(int));
                    }
                    
                    /* 关键：使用相同的candidate_idx，确保索引一致性 */
                    int candidate_idx = total_candidates;
                    
                    /* 提取TID数据（从IndexTuple的t_tid字段） */
                    packed_tids[candidate_idx] = itup->t_tid;
                    
                    /* 提取向量数据（从IndexTuple中提取Vector，跳过t_tid + t_info） */
                    Vector *vec = DatumGetVector(vector_datum);
                    /* 复制向量数据到连续数组（对齐存储） */
                    memcpy(packed_vectors + candidate_idx * dimensions, 
                           &vec->x[0], 
                           dimensions * sizeof(float));
                    
                    /* 收集查询ID（确保索引一致性） */
                    query_ids[candidate_idx] = query_idx;
                    
                    /* 调试：验证数据（前几个候选） */
                    // if (candidate_idx < 3) {
                        // elog(LOG, "UploadIndexTuplesToGPU_Batch: 候选 %d - TID: (%u,%u), 向量维度: %d", 
                        //      candidate_idx,
                        //      ItemPointerGetBlockNumber(&packed_tids[candidate_idx]),
                        //      ItemPointerGetOffsetNumber(&packed_tids[candidate_idx]),
                        //      dimensions);
                    // }
                    
                    total_candidates++;
                }
                
                searchPage = IvfflatPageGetOpaque(page)->nextblkno;
                UnlockReleaseBuffer(buf);
            }
        }
    }
    
    if (total_candidates == 0) {
        elog(WARNING, "UploadIndexTuplesToGPU_Batch: 没有找到任何候选向量");
        pfree(packed_vectors);
        pfree(packed_tids);
        pfree(query_ids);
        return -1;
    }
    
    /* 上传分离的向量数据和TID数据到GPU（分离存储方案） */
    {
        int upload_result;
        upload_result = cuda_upload_probe_vectors(so->cuda_ctx,
                                                  packed_vectors,  // 向量数据（对齐存储）
                                                  (const char*)packed_tids,  // TID数据（ItemPointerData数组）
                                                  query_ids,      // 查询ID映射
                                                  total_candidates,
                                                  dimensions);
        
        pfree(packed_vectors);
        pfree(packed_tids);
        pfree(query_ids);
        
        if (upload_result != 0) {
            /* 输出详细的错误信息以帮助调试 */
            elog(ERROR, "UploadIndexTuplesToGPU_Batch: GPU上传失败（分离存储方案）- 候选数: %d, 维度: %d, 批量查询数: %d, 最大probes: %d。请检查CUDA错误日志（stderr）获取详细错误信息", 
                 total_candidates, dimensions, nbatch, maxProbes);
            return -1;
        }
    }
    
    // elog(LOG, "UploadIndexTuplesToGPU_Batch: 成功上传 %d 个候选（分离存储方案 - 向量数据 + TID数据），批量查询数: %d, 每个查询平均候选数: %.1f", 
    //      total_candidates, nbatch, (float)total_candidates / nbatch);
    return 0;
}

/*
 * 为batch_search_pipeline准备数据
 * 从PostgreSQL索引读取所有cluster的数据并组织为pipeline需要的格式
 */
static int
PrepareBatchDataForPipeline(IndexScanDesc scan, ScanKeyBatch batch_keys, int n_probes,
                            float*** query_batch_out, int** cluster_size_out,
                            float**** cluster_vectors_out, float*** cluster_center_data_out,
                            VectorIndexMapping*** mapping_table_out,
                            BlockNumber** cluster_pages_out,
                            int* n_total_clusters_out, int* n_total_vectors_out)
{
    elog(LOG, "PrepareBatchDataForPipeline: 开始执行, n_probes=%d", n_probes);
    
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    TupleDesc tupdesc = RelationGetDescr(scan->indexRelation);
    int totalLists = 0;
    IvfflatGetMetaPageInfo(scan->indexRelation, &totalLists, NULL);
    
    elog(LOG, "PrepareBatchDataForPipeline: totalLists=%d", totalLists);
    
    if (totalLists <= 0) {
        elog(ERROR, "PrepareBatchDataForPipeline: totalLists <= 0");
        
        return -1;
    }
    
    int dimensions = so->dimensions;
    int nbatch = batch_keys->nkeys;
    
    elog(LOG, "PrepareBatchDataForPipeline: dimensions=%d, nbatch=%d", dimensions, nbatch);
    
    /* 第一次遍历：统计每个cluster的向量数量 */
    int* cluster_sizes = (int*)palloc0(totalLists * sizeof(int));
    BlockNumber* cluster_start_pages = (BlockNumber*)palloc(totalLists * sizeof(BlockNumber));
    int* cluster_center_indices = (int*)palloc(totalLists * sizeof(int));
    
    BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
    int center_idx = 0;
    
    /* 收集cluster元数据 */
    while (BlockNumberIsValid(nextblkno))
    {
        Buffer cbuf = ReadBuffer(scan->indexRelation, nextblkno);
        LockBuffer(cbuf, BUFFER_LOCK_SHARE);
        Page cpage = BufferGetPage(cbuf);
        OffsetNumber maxoffno = PageGetMaxOffsetNumber(cpage);

        for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
        {
            if (center_idx >= totalLists) {
                UnlockReleaseBuffer(cbuf);
                pfree(cluster_sizes);
                pfree(cluster_start_pages);
                pfree(cluster_center_indices);
                return -1;
            }
            
            IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
            cluster_start_pages[center_idx] = list->startPage;
            cluster_center_indices[center_idx] = center_idx;
            center_idx++;
        }

        nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;
        UnlockReleaseBuffer(cbuf);
    }
    
    /* 统计每个cluster的向量数量 */
    elog(LOG, "PrepareBatchDataForPipeline: 开始统计每个cluster的向量数量");
    int total_vectors = 0;
    for (int i = 0; i < totalLists; i++) {
        if (i % 10 == 0 || i < 3) {
            elog(LOG, "PrepareBatchDataForPipeline: 统计 cluster %d/%d", i, totalLists);
        }
        BlockNumber searchPage = cluster_start_pages[i];
        elog(LOG, "PrepareBatchDataForPipeline: cluster %d, searchPage=%u, BlockNumberIsValid=%d", 
             i, searchPage, BlockNumberIsValid(searchPage));
        if (!BlockNumberIsValid(searchPage)) {
            elog(LOG, "PrepareBatchDataForPipeline: cluster %d 的起始页面无效", i);
            continue;
        }
        while (BlockNumberIsValid(searchPage)) {
            elog(LOG, "PrepareBatchDataForPipeline: cluster %d, 读取页面 %u", i, searchPage);
            Buffer buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, 
                                          searchPage, RBM_NORMAL, NULL);
            elog(LOG, "PrepareBatchDataForPipeline: cluster %d, ReadBufferExtended 完成, buf=%d", i, buf);
            if (buf == InvalidBuffer) {
                elog(ERROR, "PrepareBatchDataForPipeline: 无法读取页面 %u", searchPage);
                break;
            }
            elog(LOG, "PrepareBatchDataForPipeline: cluster %d, 锁定缓冲区", i);
            LockBuffer(buf, BUFFER_LOCK_SHARE);
            elog(LOG, "PrepareBatchDataForPipeline: cluster %d, 获取 Page 指针", i);
            Page page = BufferGetPage(buf);
            if (!page) {
                elog(ERROR, "PrepareBatchDataForPipeline: 页面 %u 的 Page 指针为 NULL", searchPage);
                UnlockReleaseBuffer(buf);
                break;
            }
            elog(LOG, "PrepareBatchDataForPipeline: cluster %d, 获取 maxoffno", i);
            OffsetNumber maxoffno = PageGetMaxOffsetNumber(page);
            elog(LOG, "PrepareBatchDataForPipeline: cluster %d, maxoffno=%d", i, maxoffno);
            
            for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno)) {
                if (offno <= 3 || offno == maxoffno) {
                    elog(LOG, "PrepareBatchDataForPipeline: cluster %d, 处理 offno=%d", i, offno);
                }
                ItemId itemId = PageGetItemId(page, offno);
                if (!itemId) {
                    elog(ERROR, "PrepareBatchDataForPipeline: ItemId 为 NULL (cluster %d, page %u, offno %d)", 
                         i, searchPage, offno);
                    continue;
                }
                elog(LOG, "PrepareBatchDataForPipeline: cluster %d, offno=%d, 获取 IndexTuple", i, offno);
                IndexTuple itup = (IndexTuple) PageGetItem(page, itemId);
                if (!itup) {
                    elog(ERROR, "PrepareBatchDataForPipeline: IndexTuple 为 NULL (cluster %d, page %u, offno %d)", 
                         i, searchPage, offno);
                    continue;
                }
                elog(LOG, "PrepareBatchDataForPipeline: cluster %d, offno=%d, 获取 vector_datum", i, offno);
                bool isnull = false;
                Datum vector_datum = index_getattr(itup, 1, tupdesc, &isnull);
                if (!isnull) {
                    cluster_sizes[i]++;
                    total_vectors++;
                }
            }
            
            elog(LOG, "PrepareBatchDataForPipeline: cluster %d, 获取下一页", i);
            searchPage = IvfflatPageGetOpaque(page)->nextblkno;
            elog(LOG, "PrepareBatchDataForPipeline: cluster %d, 释放缓冲区", i);
            UnlockReleaseBuffer(buf);
        }
        elog(LOG, "PrepareBatchDataForPipeline: cluster %d 统计完成, cluster_sizes[%d]=%d", i, i, cluster_sizes[i]);
    }
    elog(LOG, "PrepareBatchDataForPipeline: 统计完成, total_vectors=%d", total_vectors);
    
    elog(LOG, "PrepareBatchDataForPipeline: 开始分配输出数据结构, total_vectors=%d", total_vectors);
    
    /* 分配输出数据结构 */
    float** query_batch = (float**)palloc(nbatch * sizeof(float*));
    float* query_data_contiguous = (float*)palloc(nbatch * dimensions * sizeof(float));
    
    if (!query_batch || !query_data_contiguous) {
        elog(ERROR, "PrepareBatchDataForPipeline: 无法分配 query_batch 内存");
        return -1;
    }
    
    elog(LOG, "PrepareBatchDataForPipeline: 开始复制 query_batch 数据, nbatch=%d", nbatch);
    
    for (int i = 0; i < nbatch; i++) {
        elog(LOG, "PrepareBatchDataForPipeline: 处理 query %d", i);
        Datum vector_datum = ScanKeyBatchGetVector(batch_keys, i);
        if (vector_datum == (Datum) NULL) {
            elog(ERROR, "PrepareBatchDataForPipeline: query %d 的向量数据为 NULL", i);
            return -1;
        }
        Vector *vec = DatumGetVector(vector_datum);
        if (!vec) {
            elog(ERROR, "PrepareBatchDataForPipeline: query %d 的 Vector 指针为 NULL", i);
            return -1;
        }
        memcpy(query_data_contiguous + i * dimensions, &vec->x[0], dimensions * sizeof(float));
        query_batch[i] = query_data_contiguous + i * dimensions;
    }
    
    elog(LOG, "PrepareBatchDataForPipeline: query_batch 分配完成");
    
    int* cluster_size = (int*)palloc(totalLists * sizeof(int));
    memcpy(cluster_size, cluster_sizes, totalLists * sizeof(int));
    
    float*** cluster_vectors = (float***)palloc(totalLists * sizeof(float**));
    float** cluster_vectors_data = (float**)palloc(totalLists * sizeof(float*));
    
    if (!cluster_vectors || !cluster_vectors_data) {
        elog(ERROR, "PrepareBatchDataForPipeline: 无法分配 cluster_vectors 内存");
        return -1;
    }
    
    elog(LOG, "PrepareBatchDataForPipeline: cluster_vectors 分配完成");
    
    float** cluster_center_data = (float**)palloc(totalLists * sizeof(float*));
    float* cluster_centers_contiguous = (float*)palloc(totalLists * dimensions * sizeof(float));
    
    VectorIndexMapping* mapping_table = (VectorIndexMapping*)palloc(total_vectors * sizeof(VectorIndexMapping));
    int mapping_idx = 0;
    
    /* 第二次遍历：读取向量数据和cluster中心，构建映射表 */
    nextblkno = IVFFLAT_HEAD_BLKNO;
    center_idx = 0;
    
    while (BlockNumberIsValid(nextblkno))
    {
        Buffer cbuf = ReadBuffer(scan->indexRelation, nextblkno);
        LockBuffer(cbuf, BUFFER_LOCK_SHARE);
        Page cpage = BufferGetPage(cbuf);
        OffsetNumber maxoffno = PageGetMaxOffsetNumber(cpage);

        for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
        {
            IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
            Vector *center_vector = &list->center;
            
            memcpy(cluster_centers_contiguous + center_idx * dimensions, 
                   &center_vector->x[0], dimensions * sizeof(float));
            cluster_center_data[center_idx] = cluster_centers_contiguous + center_idx * dimensions;
            
            if (cluster_sizes[center_idx] > 0) {
                float* cluster_data = (float*)palloc(cluster_sizes[center_idx] * dimensions * sizeof(float));
                cluster_vectors_data[center_idx] = cluster_data;
                cluster_vectors[center_idx] = &cluster_vectors_data[center_idx];
                
                /* 记录当前聚类的mapping起始索引，用于批量读取表行ID */
                int cluster_mapping_start = mapping_idx;
                
                int vector_idx = 0;
                BlockNumber searchPage = cluster_start_pages[center_idx];
                while (BlockNumberIsValid(searchPage)) {
                    Buffer buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, 
                                                  searchPage, RBM_NORMAL, NULL);
                    LockBuffer(buf, BUFFER_LOCK_SHARE);
                    Page page = BufferGetPage(buf);
                    OffsetNumber maxoffno_page = PageGetMaxOffsetNumber(page);
                    
                    for (OffsetNumber offno_page = FirstOffsetNumber; offno_page <= maxoffno_page; offno_page = OffsetNumberNext(offno_page)) {
                        IndexTuple itup = (IndexTuple) PageGetItem(page, PageGetItemId(page, offno_page));
                        bool isnull = false;
                        Datum vector_datum = index_getattr(itup, 1, tupdesc, &isnull);
                        
                        if (!isnull) {
                            Vector *vec = DatumGetVector(vector_datum);
                            memcpy(cluster_data + vector_idx * dimensions, 
                                   &vec->x[0], dimensions * sizeof(float));
                            
                            mapping_table[mapping_idx].cluster_id = center_idx;
                            mapping_table[mapping_idx].vector_index_in_cluster = vector_idx;
                            mapping_table[mapping_idx].tid = itup->t_tid;
                            
                            /* 先设置为-1，稍后批量从堆表读取 */
                            mapping_table[mapping_idx].table_row_id = -1;
                            
                            mapping_idx++;
                            vector_idx++;
                        }
                    }
                    
                    searchPage = IvfflatPageGetOpaque(page)->nextblkno;
                    UnlockReleaseBuffer(buf);
                }
                
                /* 批量读取当前聚类的表行ID */
                int cluster_mapping_count = mapping_idx - cluster_mapping_start;
                if (cluster_mapping_count > 0) {
                    /* 打开堆表关系 */
                    Oid heapRelid = IndexGetRelation(RelationGetRelid(scan->indexRelation), false);
                    Relation heapRelation = table_open(heapRelid, AccessShareLock);
                    TupleDesc heapDesc = RelationGetDescr(heapRelation);
                    Snapshot snapshot = GetActiveSnapshot();
                    
                    /* 查找id字段的索引（假设id是第一个字段） */
                    int id_attnum = -1;
                    for (int i = 1; i <= heapDesc->natts; i++) {
                        Form_pg_attribute attr = TupleDescAttr(heapDesc, i - 1);
                        if (strcmp(NameStr(attr->attname), "id") == 0) {
                            id_attnum = i;
                            break;
                        }
                    }
                    
                    if (id_attnum > 0) {
                        /* 批量读取表行ID */
                        for (int i = 0; i < cluster_mapping_count; i++) {
                            int idx = cluster_mapping_start + i;
                            ItemPointer tid = &mapping_table[idx].tid;
                            
                            if (ItemPointerIsValid(tid)) {
                                Buffer buffer;
                                HeapTupleData tuple;
                                tuple.t_self = *tid;  /* 设置tuple的TID */
                                bool found = heap_fetch(heapRelation, snapshot, &tuple, &buffer, false);
                                if (found) {
                                    HeapTuple heapTuple = &tuple;
                                    bool isnull = false;
                                    Datum id_datum = heap_getattr(heapTuple, id_attnum, heapDesc, &isnull);
                                    if (!isnull) {
                                        mapping_table[idx].table_row_id = DatumGetInt32(id_datum);
                                    } else {
                                        mapping_table[idx].table_row_id = -1;
                                    }
                                    ReleaseBuffer(buffer);
                                } else {
                                    mapping_table[idx].table_row_id = -1;
                                }
                            } else {
                                mapping_table[idx].table_row_id = -1;
                            }
                        }
                    } else {
                        elog(WARNING, "PrepareBatchDataForPipeline: 未找到id字段，跳过表行ID读取");
                    }
                    
                    table_close(heapRelation, AccessShareLock);
                }
            } else {
                cluster_vectors[center_idx] = NULL;
            }
            
            center_idx++;
        }

        nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;
        UnlockReleaseBuffer(cbuf);
    }
    
    *query_batch_out = query_batch;
    *cluster_size_out = cluster_size;
    *cluster_vectors_out = cluster_vectors;
    *cluster_center_data_out = cluster_center_data;
    *mapping_table_out = mapping_table;
    *cluster_pages_out = cluster_start_pages;
    *n_total_clusters_out = totalLists;
    *n_total_vectors_out = total_vectors;
    
    elog(LOG, "PrepareBatchDataForPipeline: 数据准备完成, n_total_clusters=%d, n_total_vectors=%d", 
         totalLists, total_vectors);
    
    pfree(cluster_sizes);
    pfree(cluster_center_indices);
    
    return 0;
}

/*
 * 将batch_search_pipeline的输出转换为BatchBuffer格式
 * topk_index是全局向量索引（在所有向量中的索引，从0开始）
 */
static void
ConvertBatchPipelineResults(IndexScanDesc scan, float** topk_dist, int** topk_index,
                            int n_query, int k, BatchBuffer* result_buffer,
                            VectorIndexMapping* mapping_table, int* cluster_size,
                            BlockNumber* cluster_pages, int n_total_clusters)
{
    /* 验证 result_buffer 大小匹配 */
    if (result_buffer->n_queries != n_query || result_buffer->k != k) {
        elog(ERROR, "ConvertBatchPipelineResults: result_buffer大小不匹配 (n_queries=%d vs %d, k=%d vs %d)",
             result_buffer->n_queries, n_query, result_buffer->k, k);
        return;
    }
    
    int total_vectors = 0;
    for (int i = 0; i < n_total_clusters; i++) {
        total_vectors += cluster_size[i];
    }
    
    /* 重置 result_buffer：先清零所有字段，确保没有残留数据 */
    int buffer_size = result_buffer->n_queries * result_buffer->k;
    memset(result_buffer->query_ids, -1, buffer_size * sizeof(int));
    memset(result_buffer->global_vector_indices, -1, buffer_size * sizeof(int));
    memset(result_buffer->distances, 0, buffer_size * sizeof(float));
    for (int i = 0; i < buffer_size; i++) {
        ItemPointerSetInvalid(&result_buffer->vector_ids[i]);
    }
    
    /* 每个查询都有k个槽位，总共 n_query * k 个结果位置 */
    /* 注意：虽然数组大小是 n_query * k，但 total_results 只计算有效结果数 */
    for (int query_idx = 0; query_idx < n_query; query_idx++) {
        for (int i = 0; i < k; i++) {
            int buffer_idx = query_idx * k + i;
            int global_vector_idx = topk_index[query_idx][i];
            
            /* 边界检查：确保 buffer_idx 在数组范围内 */
            if (buffer_idx >= result_buffer->n_queries * result_buffer->k) {
                elog(ERROR, "ConvertBatchPipelineResults: buffer_idx=%d 超出范围 (n_queries=%d, k=%d, max=%d)", 
                     buffer_idx, result_buffer->n_queries, result_buffer->k, result_buffer->n_queries * result_buffer->k);
                continue;
            }
            
            if (global_vector_idx < 0 && global_vector_idx >= total_vectors) {
                /* null值或无效索引 */
                elog(LOG, "ConvertBatchPipelineResults: 无效的 global_vector_idx=%d (total_vectors=%d), query_idx=%d, i=%d", 
                     global_vector_idx, total_vectors, query_idx, i);
                result_buffer->query_ids[buffer_idx] = query_idx;
                result_buffer->global_vector_indices[buffer_idx] = -1;  /* 无效索引标记 */
                ItemPointerSetInvalid(&result_buffer->vector_ids[buffer_idx]);
                result_buffer->distances[buffer_idx] = INFINITY;
            }

            /* 直接使用全局索引在mapping_table中查找 */
            VectorIndexMapping* mapping = &mapping_table[global_vector_idx];
            result_buffer->query_ids[buffer_idx] = query_idx;
            /* 保存表行ID（用于返回给用户） */
            result_buffer->global_vector_indices[buffer_idx] = mapping->table_row_id;
            /* 确保 tid 是有效的 ItemPointer（保留用于内部使用） */
            if (ItemPointerIsValid(&mapping->tid)) {
                result_buffer->vector_ids[buffer_idx] = mapping->tid;
            } else {
                elog(WARNING, "ConvertBatchPipelineResults: mapping_table[%d].tid 无效, query_idx=%d, i=%d", 
                        global_vector_idx, query_idx, i);
                ItemPointerSetInvalid(&result_buffer->vector_ids[buffer_idx]);
            }
            result_buffer->distances[buffer_idx] = topk_dist[query_idx][i];
            
            /* 调试：打印前几个结果的详细信息 */
            if (query_idx < 3 && i < 3) {
                elog(LOG, "ConvertBatchPipelineResults: 查询%d结果%d - global_vector_idx=%d, table_row_id=%d, cluster_id=%d, vector_idx_in_cluster=%d, distance=%.10f, tid=(%u,%u)", 
                        query_idx, i, global_vector_idx, mapping->table_row_id, mapping->cluster_id, 
                        mapping->vector_index_in_cluster, topk_dist[query_idx][i],
                        ItemPointerGetBlockNumber(&mapping->tid),
                        ItemPointerGetOffsetNumber(&mapping->tid));
            }
            
        }
    }
}

/* C++包装函数声明（在C++文件中实现，通过extern "C"导出） */
/* 新接口：接受 device 指针 */
extern int batch_search_pipeline_wrapper(float* d_query_batch,
                                         int* d_cluster_size,
                                         float* d_cluster_vectors,
                                         float* d_cluster_centers,
                                         int* d_initial_indices,
                                         float* d_topk_dist,
                                         int* d_topk_index,
                                         int n_query, int n_dim, int n_total_cluster,
                                         int n_total_vectors, int n_probes, int k);

/*
 * 使用batch_search_pipeline的新GPU处理流程
 * 使用 cudaMemcpy 逐个复制 cluster 数据到 GPU
 */
/* 辅助函数：清理 GPU 内存 */
static void
CleanupGPUMemory(float* d_query_batch, int* d_cluster_size, float* d_cluster_vectors,
                 float* d_cluster_centers, int* d_initial_indices, float* d_topk_dist, int* d_topk_index)
{
    if (d_query_batch) cudaFree(d_query_batch);
    if (d_cluster_size) cudaFree(d_cluster_size);
    if (d_cluster_vectors) cudaFree(d_cluster_vectors);
    if (d_cluster_centers) cudaFree(d_cluster_centers);
    if (d_initial_indices) cudaFree(d_initial_indices);
    if (d_topk_dist) cudaFree(d_topk_dist);
    if (d_topk_index) cudaFree(d_topk_index);
}

/**
 * 
 * 特点：
 * - 使用 batch_search_pipeline_wrapper (调用 batch_search_pipeline)
 * - 不依赖 cuda_ctx，直接管理GPU内存
 * - 返回全局向量索引，通过 mapping_table 映射回 TID
 * - 支持传入初始索引 (d_initial_indices)
 * 
 */
void
ProcessBatchQueriesGPU(IndexScanDesc scan, ScanKeyBatch batch_keys, int k)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    int nbatch = batch_keys->nkeys;
    int dimensions = so->dimensions;
    
    /* 重置处理状态，确保每次查询都重新处理 */
    so->batch_processing_complete = false;
    
    /* 检查并重新创建 result_buffer（如果查询数量或k值改变） */
    if (so->result_buffer != NULL) {
        if (so->result_buffer->n_queries != nbatch || so->result_buffer->k != k) {
            elog(LOG, "ProcessBatchQueriesGPU: result_buffer大小不匹配 (n_queries=%d vs %d, k=%d vs %d)，重新创建",
                 so->result_buffer->n_queries, nbatch, so->result_buffer->k, k);
            /* 旧的result_buffer会在SRF结束时自动清理，这里只需要设置为NULL */
            so->result_buffer = NULL;
        }
    }
    
    /* 准备数据（所有 palloc 分配的内存都在 SRF 内存上下文中） */
    float** query_batch = NULL;
    int* cluster_size = NULL;
    float*** cluster_vectors = NULL;
    float** cluster_center_data = NULL;
    VectorIndexMapping* mapping_table = NULL;
    BlockNumber* cluster_pages = NULL;
    int n_total_clusters = 0;
    int n_total_vectors = 0;
    
    /* GPU 内存指针 */
    float* d_query_batch = NULL;
    int* d_cluster_size = NULL;
    float* d_cluster_vectors = NULL;
    float* d_cluster_centers = NULL;
    int* d_initial_indices = NULL;
    float* d_topk_dist = NULL;
    int* d_topk_index = NULL;
    
    /* 输出缓冲区 */
    float** topk_dist = NULL;
    int** topk_index = NULL;
    
    cudaError_t cuda_err;
    
    elog(LOG, "ProcessBatchQueriesGPU: 开始执行, nbatch=%d, dimensions=%d, probes=%d, k=%d",
         nbatch, dimensions, so->probes, k);
    
    /* ========== 数据准备 ========== */
    if (PrepareBatchDataForPipeline(scan, batch_keys, so->probes,
                                    &query_batch, &cluster_size,
                                    &cluster_vectors, &cluster_center_data,
                                    &mapping_table, &cluster_pages,
                                    &n_total_clusters, &n_total_vectors) != 0) {
        elog(ERROR, "ProcessBatchQueriesGPU: 数据准备失败");
        return;
    }
    
    elog(LOG, "ProcessBatchQueriesGPU: 数据准备完成, n_total_clusters=%d, n_total_vectors=%d",
         n_total_clusters, n_total_vectors);
    
    /* 检查数据有效性 */
    if (!query_batch || !query_batch[0] || !cluster_size || 
        !cluster_center_data || !cluster_center_data[0] ||
        n_total_clusters <= 0 || n_total_vectors <= 0) {
        elog(ERROR, "ProcessBatchQueriesGPU: 数据验证失败");
        
        return;
    }
    
    /* 计算并记录每个分配的大小 */
    size_t query_batch_size = (size_t)nbatch * (size_t)dimensions * sizeof(float);
    size_t cluster_size_size = (size_t)n_total_clusters * sizeof(int);
    size_t cluster_vectors_size = (size_t)n_total_vectors * (size_t)dimensions * sizeof(float);
    size_t cluster_centers_size = (size_t)n_total_clusters * (size_t)dimensions * sizeof(float);
    size_t initial_indices_size = (size_t)nbatch * (size_t)n_total_clusters * sizeof(int);
    size_t topk_dist_size = (size_t)nbatch * (size_t)k * sizeof(float);
    size_t topk_index_size = (size_t)nbatch * (size_t)k * sizeof(int);
    
    elog(LOG, "ProcessBatchQueriesGPU: GPU 内存分配计划 - query_batch=%zu bytes, cluster_size=%zu bytes, cluster_vectors=%zu bytes, cluster_centers=%zu bytes, topk_dist=%zu bytes, topk_index=%zu bytes",
         query_batch_size, cluster_size_size, cluster_vectors_size, cluster_centers_size, topk_dist_size, topk_index_size);
    elog(LOG, "ProcessBatchQueriesGPU: GPU 内存分配计划 - 总计=%zu bytes (%.2f MB)",
         query_batch_size + cluster_size_size + cluster_vectors_size + cluster_centers_size + topk_dist_size + topk_index_size,
         (query_batch_size + cluster_size_size + cluster_vectors_size + cluster_centers_size + topk_dist_size + topk_index_size) / (1024.0 * 1024.0));
    
    /* ========== GPU 内存分配 ========== */
    cuda_err = cudaMalloc(&d_query_batch, query_batch_size);
    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: 无法分配 query batch GPU 内存 (%zu bytes): %s", 
            query_batch_size, cudaGetErrorString(cuda_err));
        
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU: d_query_batch 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_query_batch, query_batch_size);
    cuda_err = cudaMalloc(&d_cluster_size, cluster_size_size);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: 无法分配 cluster_size GPU 内存 (%zu bytes): %s", 
            cluster_size_size, cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, NULL, NULL, NULL, NULL, NULL, NULL);
        
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU: d_cluster_size 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_cluster_size, cluster_size_size);
    cuda_err = cudaMalloc(&d_cluster_vectors, cluster_vectors_size);


    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: 无法分配 cluster_vectors GPU 内存 (%zu bytes): %s", 
            cluster_vectors_size, cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, NULL, NULL, NULL, NULL, NULL);
        
        return;
    }      
    
    elog(LOG, "ProcessBatchQueriesGPU: d_cluster_vectors 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_cluster_vectors, cluster_vectors_size);
    
    cuda_err = cudaMalloc(&d_cluster_centers, cluster_centers_size);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: 无法分配 cluster_centers GPU 内存 (%zu bytes): %s", 
            cluster_centers_size, cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        NULL, NULL, NULL, NULL);
        
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU: d_cluster_centers 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_cluster_centers, cluster_centers_size);
    
    cuda_err = cudaMalloc(&d_topk_dist, topk_dist_size);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: 无法分配 topk_dist GPU 内存 (%zu bytes): %s", 
            topk_dist_size, cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, NULL, NULL, NULL);
        
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU: d_topk_dist 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_topk_dist, topk_dist_size);
    
    /* 分配 d_initial_indices: 每个查询对应 n_total_clusters 个初始索引 */
    cuda_err = cudaMalloc(&d_initial_indices, initial_indices_size);
    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: 无法分配 initial_indices GPU 内存 (%zu bytes): %s", 
            initial_indices_size, cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, NULL, d_topk_dist, NULL);
        
        return;
    }
    
    elog(LOG, "ProcessBatchQueriesGPU: d_initial_indices 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_initial_indices, initial_indices_size);
    
    cuda_err = cudaMalloc(&d_topk_index, topk_index_size);    


    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: 无法分配 topk_index GPU 内存 (%zu bytes): %s", 
            topk_index_size, cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_initial_indices, d_topk_dist, NULL);
        
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU: d_topk_index 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_topk_index, topk_index_size);
    elog(LOG, "ProcessBatchQueriesGPU: GPU 内存分配完成");
    
    /* ========== CPU -> GPU 数据复制 ========== */
    if (!query_batch || !query_batch[0]) {
        elog(ERROR, "ProcessBatchQueriesGPU: query_batch 为空，无法复制");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU: 准备复制 query_batch 到GPU, nbatch=%d, dimensions=%d, 总大小=%zu bytes", 
         nbatch, dimensions, (size_t)nbatch * dimensions * sizeof(float));
    elog(LOG, "ProcessBatchQueriesGPU: query_batch[0]=%p, query_data_contiguous=%p", 
         (void*)query_batch[0], (void*)query_batch[0]);
    cuda_err = cudaMemcpy(d_query_batch, query_batch[0], 
        nbatch * dimensions * sizeof(float), 
        cudaMemcpyHostToDevice);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: query_batch 复制失败: %s", 
                cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    /*H2D: cluster size*/
    cuda_err = cudaMemcpy(d_cluster_size, cluster_size, 
        n_total_clusters * sizeof(int), 
        cudaMemcpyHostToDevice);
    
    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: cluster_size 复制失败: %s", 
            cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    /*H2D: cluster_vectors：使用 cudaMemcpy 逐个复制每个 cluster */
    int gpu_offset = 0;
    int copied_clusters = 0;
    int copied_vectors = 0;
    elog(LOG, "ProcessBatchQueriesGPU: 开始复制 cluster_vectors, n_total_clusters=%d, n_total_vectors=%d", 
            n_total_clusters, n_total_vectors);
    
    int copy_success = 1;
    for (int cid = 0; cid < n_total_clusters && copy_success; cid++) {
        if (cluster_size[cid] <= 0 || !cluster_vectors || !cluster_vectors[cid] || !cluster_vectors[cid][0]) {
            elog(LOG, "ProcessBatchQueriesGPU: 跳过 cluster %d (cluster_size=%d, cluster_vectors=%p), gpu_offset保持不变=%d", 
                    cid, cluster_size[cid], cluster_vectors ? (cluster_vectors[cid] ? cluster_vectors[cid][0] : NULL) : NULL, gpu_offset);
            /* 重要：即使跳过空cluster，gpu_offset也要增加0（保持不变），确保后续cluster的位置正确 */
            /* 但这里 cluster_size[cid] <= 0，所以 gpu_offset += 0 是正确的 */
            continue;
        }

        size_t cluster_bytes = cluster_size[cid] * dimensions * sizeof(float);
        elog(LOG, "ProcessBatchQueriesGPU: 复制 cluster %d, cluster_size=%d, gpu_offset=%d, bytes=%zu", 
                cid, cluster_size[cid], gpu_offset, cluster_bytes);
        cuda_err = cudaMemcpy(d_cluster_vectors + gpu_offset * dimensions,
                                cluster_vectors[cid][0],
                                cluster_bytes,
                                cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            elog(ERROR, "ProcessBatchQueriesGPU: cluster_vectors[%d] 复制失败: %s", 
                cid, cudaGetErrorString(cuda_err));
            copy_success = 0;
        } else {
        gpu_offset += cluster_size[cid];
        copied_clusters++;
        copied_vectors += cluster_size[cid];
        }
    }

    if (!copy_success) {
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    if (gpu_offset != n_total_vectors) {
        elog(ERROR, "ProcessBatchQueriesGPU: cluster_vectors 复制数量不匹配! gpu_offset=%d, n_total_vectors=%d", 
            gpu_offset, n_total_vectors);
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU: cluster_vectors 复制完成, copied_clusters=%d, copied_vectors=%d, gpu_offset=%d, expected_total=%d", 
            copied_clusters, copied_vectors, gpu_offset, n_total_vectors);

    if (!cluster_center_data || !cluster_center_data[0]) {
        elog(ERROR, "ProcessBatchQueriesGPU: cluster_center_data 为空，无法复制");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    /* H2D: cluster_centers */
    cuda_err = cudaMemcpy(d_cluster_centers, cluster_center_data[0], 
        n_total_clusters * dimensions * sizeof(float), 
        cudaMemcpyHostToDevice);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: cluster_centers 复制失败: %s", 
            cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    /* H2D: 初始化 d_initial_indices - 每个查询对应 n_total_clusters 个初始索引 (0 到 n_total_clusters-1) */
    int* initial_indices_host = (int*)palloc(nbatch * n_total_clusters * sizeof(int));
    if (!initial_indices_host) {
        elog(ERROR, "ProcessBatchQueriesGPU: 无法分配 initial_indices_host 内存");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }
    
    /* 初始化：每个查询的初始索引为 0 到 n_total_clusters-1 */
    for (int query_idx = 0; query_idx < nbatch; query_idx++) {
        for (int cluster_idx = 0; cluster_idx < n_total_clusters; cluster_idx++) {
            initial_indices_host[query_idx * n_total_clusters + cluster_idx] = cluster_idx;
        }
    }
    
    cuda_err = cudaMemcpy(d_initial_indices, initial_indices_host,
                          nbatch * n_total_clusters * sizeof(int),
                          cudaMemcpyHostToDevice);
    pfree(initial_indices_host);
    
    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: initial_indices 复制失败: %s", 
            cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU: CPU->GPU 数据复制完成, nbatch=%d, n_total_clusters=%d, n_total_vectors=%d, probes=%d, k=%d",
            nbatch, n_total_clusters, n_total_vectors, so->probes, k);
    
    /* ========== 调用 GPU 计算 ========== */
    elog(LOG, "ProcessBatchQueriesGPU: 准备调用 batch_search_pipeline_wrapper");
    
    /* 详细参数验证 */
    elog(LOG, "ProcessBatchQueriesGPU: ========== GPU内存指针验证 ==========");
    elog(LOG, "ProcessBatchQueriesGPU: d_query_batch=%p (size=%zu bytes, %d queries × %d dims)", 
            (void*)d_query_batch, (size_t)nbatch * dimensions * sizeof(float), nbatch, dimensions);
    elog(LOG, "ProcessBatchQueriesGPU: d_cluster_size=%p (size=%zu bytes, %d clusters)", 
            (void*)d_cluster_size, (size_t)n_total_clusters * sizeof(int), n_total_clusters);
    elog(LOG, "ProcessBatchQueriesGPU: d_cluster_vectors=%p (size=%zu bytes, %d vectors × %d dims)", 
            (void*)d_cluster_vectors, (size_t)n_total_vectors * dimensions * sizeof(float), n_total_vectors, dimensions);
    elog(LOG, "ProcessBatchQueriesGPU: d_cluster_centers=%p (size=%zu bytes, %d clusters × %d dims)", 
            (void*)d_cluster_centers, (size_t)n_total_clusters * dimensions * sizeof(float), n_total_clusters, dimensions);
    elog(LOG, "ProcessBatchQueriesGPU: d_topk_dist=%p (size=%zu bytes, %d queries × %d k)", 
            (void*)d_topk_dist, (size_t)nbatch * k * sizeof(float), nbatch, k);
    elog(LOG, "ProcessBatchQueriesGPU: d_topk_index=%p (size=%zu bytes, %d queries × %d k)", 
            (void*)d_topk_index, (size_t)nbatch * k * sizeof(int), nbatch, k);
    
    elog(LOG, "ProcessBatchQueriesGPU: ========== 参数值验证 ==========");
    elog(LOG, "ProcessBatchQueriesGPU: n_query=%d, n_dim=%d, n_total_cluster=%d, n_total_vectors=%d, n_probes=%d, k=%d",
            nbatch, dimensions, n_total_clusters, n_total_vectors, so->probes, k);
    
    /* 验证 cluster_size 数组 */
    int sum_cluster_sizes = 0;
    elog(LOG, "ProcessBatchQueriesGPU: cluster_size数组内容 (前10个):");
    for (int i = 0; i < n_total_clusters && i < 10; i++) {
        sum_cluster_sizes += cluster_size[i];
        elog(LOG, "ProcessBatchQueriesGPU:   cluster_size[%d]=%d", i, cluster_size[i]);
    }
    if (n_total_clusters > 10) {
        elog(LOG, "ProcessBatchQueriesGPU:   ... (共%d个clusters)", n_total_clusters);
    }
    for (int i = 10; i < n_total_clusters; i++) {
        sum_cluster_sizes += cluster_size[i];
    }
    elog(LOG, "ProcessBatchQueriesGPU: cluster_size 数组总和=%d, n_total_vectors=%d", 
            sum_cluster_sizes, n_total_vectors);
    
    /* 验证 query_batch 数据（前几个值） */
    if (query_batch && query_batch[0]) {
        elog(LOG, "ProcessBatchQueriesGPU: query_batch[0] 前5个值: %.6f, %.6f, %.6f, %.6f, %.6f",
                query_batch[0][0], query_batch[0][1], query_batch[0][2], query_batch[0][3], query_batch[0][4]);
    }
    
    /* 验证 cluster_centers 数据（第一个cluster的前几个值） */
    if (cluster_center_data && cluster_center_data[0]) {
        elog(LOG, "ProcessBatchQueriesGPU: cluster_centers[0] 前5个值: %.6f, %.6f, %.6f, %.6f, %.6f",
                cluster_center_data[0][0], cluster_center_data[0][1], cluster_center_data[0][2], 
                cluster_center_data[0][3], cluster_center_data[0][4]);
    }
    
    elog(LOG, "ProcessBatchQueriesGPU: ========== 调用 batch_search_pipeline_wrapper ==========");
    
    if (sum_cluster_sizes != n_total_vectors) {
        elog(ERROR, "ProcessBatchQueriesGPU: cluster_size 数组总和(%d) 与 n_total_vectors(%d) 不匹配!", 
            sum_cluster_sizes, n_total_vectors);
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    int result = batch_search_pipeline_wrapper(d_query_batch, d_cluster_size, d_cluster_vectors,
                                                d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index,
                                                nbatch, dimensions, n_total_clusters,
                                                n_total_vectors, so->probes, k);
    
    if (result != 0) {
        elog(ERROR, "ProcessBatchQueriesGPU: batch_search_pipeline执行失败 (返回码: %d), 请查看PostgreSQL日志获取详细信息", result);
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    /* ========== GPU -> CPU 结果回传 ========== */
    /* 分配连续的内存块用于存储所有查询的结果 */
    float* topk_dist_flat = (float*)palloc(nbatch * k * sizeof(float));
    int* topk_index_flat = (int*)palloc(nbatch * k * sizeof(int));
    
    if (!topk_dist_flat || !topk_index_flat){
        elog(ERROR, "ProcessBatchQueriesGPU: 无法分配输出缓冲区");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }
    /* 复制结果到 host（连续数组） */
    cuda_err = cudaMemcpy(topk_dist_flat, d_topk_dist, 
                            nbatch * k * sizeof(float), 
                            cudaMemcpyDeviceToHost);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: topk_index 回传失败: %s", 
            cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    cuda_err = cudaMemcpy(topk_index_flat, d_topk_index, 
                        nbatch * k * sizeof(int), 
                        cudaMemcpyDeviceToHost);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU: topk_dist 回传失败: %s", 
            cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    /* 分配指针数组 */
    topk_dist = (float**)palloc(nbatch * sizeof(float*));
    topk_index = (int**)palloc(nbatch * sizeof(int*));

    if(!topk_dist || !topk_index){
        elog(ERROR, "ProcessBatchQueriesGPU: 无法分配指针数组");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    /* 重新组织为指针数组格式（用于 ConvertBatchPipelineResults） */
    elog(LOG, "ProcessBatchQueriesGPU: 开始重新组织指针数组, nbatch=%d", nbatch);
    for (int i = 0; i < nbatch; i++) {
        topk_dist[i] = topk_dist_flat + i * k;
        topk_index[i] = topk_index_flat + i * k;
    }
    elog(LOG, "ProcessBatchQueriesGPU: 指针数组重新组织完成");
    
    /* 调试：打印GPU返回的前几个结果 */
    elog(LOG, "ProcessBatchQueriesGPU: GPU返回的结果（前3个查询，每个查询前3个结果）:");
    for (int query_idx = 0; query_idx < nbatch && query_idx < 3; query_idx++) {
        elog(LOG, "  查询 %d:", query_idx);
        for (int i = 0; i < k && i < 3; i++) {
            int idx = query_idx * k + i;
            elog(LOG, "    结果 %d: global_vector_idx=%d, distance=%.10f", 
                 i, topk_index_flat[idx], topk_dist_flat[idx]);
        }
    }
    
    /* ========== 转换结果到 BatchBuffer ========== */
    elog(LOG, "ProcessBatchQueriesGPU: 开始转换结果到BatchBuffer");
                    
    if (!mapping_table || !cluster_size) {
        elog(ERROR, "ProcessBatchQueriesGPU: mapping_table 或 cluster_size 为 NULL");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    /* 创建或重新创建 result_buffer */
    /* 注意：每次查询都必须重新创建 result_buffer，避免复用上一次查询的结果 */
    /* 即使 nbatch 和 k 相同，也要重新创建，因为查询向量可能不同 */
    if (so->result_buffer != NULL) {
        elog(LOG, "ProcessBatchQueriesGPU: 清理旧的 result_buffer (n_queries=%d, k=%d)，重新创建 (n_queries=%d, k=%d)",
             so->result_buffer->n_queries, so->result_buffer->k, nbatch, k);
        /* 旧的result_buffer会在SRF结束时自动清理，这里只需要设置为NULL */
        so->result_buffer = NULL;
    }
        so->result_buffer = CreateBatchBuffer(nbatch, k, dimensions, CurrentMemoryContext);
                        
    if (!so->result_buffer) {
        elog(ERROR, "ProcessBatchQueriesGPU: 无法创建BatchBuffer");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU: 调用 ConvertBatchPipelineResults");
    ConvertBatchPipelineResults(scan, topk_dist, topk_index, nbatch, k,
                                so->result_buffer, mapping_table, cluster_size,
                                cluster_pages, n_total_clusters);
    elog(LOG, "ProcessBatchQueriesGPU: ConvertBatchPipelineResults 完成");
    
    /* ========== 清理 GPU 内存 ========== */
    CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                    d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
    
    return;
}

#endif
