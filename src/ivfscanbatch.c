#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/relscan.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "utils/datum.h"
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
static void GetScanItems_BatchGPU(IndexScanDesc scan, ScanKeyBatch batch_keys);

/* batch_search_pipeline相关函数 */
typedef struct {
    int cluster_id;
    int vector_index_in_cluster;
    ItemPointerData tid;
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
static void ProcessBatchQueriesGPU_NewPipeline(IndexScanDesc scan, ScanKeyBatch batch_keys, int k);
#endif

/*
 * 批量扫描开始函数
 */
IndexScanDesc
ivfflatbatchbeginscan(Relation index, int norderbys, ScanKeyBatch batch_keys)
{
    IndexScanDesc scan;
    IvfflatBatchScanOpaque so;
    MemoryContext tmpCtx;

    // elog(LOG, "ivfflatbatchbeginscan: 开始批量扫描, nkeys=%d", batch_keys->nkeys);

    /* 创建扫描描述符 */
    scan = RelationGetIndexScan(index, batch_keys->nkeys, norderbys);
    if (!scan) {
        elog(ERROR, "ivfflatbatchbeginscan: 无法创建扫描描述符");
    }
    
    /* 先创建临时内存上下文 */
    /* 注意：tmpCtx应该在CurrentMemoryContext中创建，而不是在SRF内存上下文中
     * 这样可以确保tmpCtx的生命周期独立于SRF内存上下文
     * 但是，如果CurrentMemoryContext是SRF内存上下文，tmpCtx也会在SRF清理时被删除
     * 所以，我们需要确保在SRF清理之前，tmpCtx已经被删除
     */
    tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
        "Ivfflat batch scan temporary context",
        ALLOCSET_DEFAULT_SIZES);
    
    /* 在临时上下文中分配批量扫描状态 */
    so = (IvfflatBatchScanOpaque)MemoryContextAllocZero(tmpCtx, sizeof(IvfflatBatchScanOpaqueData));
    if (!so) {
        MemoryContextDelete(tmpCtx);
        elog(ERROR, "ivfflatbatchbeginscan: 无法分配批量扫描状态内存");
    }
    
    /* 初始化基础信息 */
    so->typeInfo = IvfflatGetTypeInfo(index);
    
    /* 从元页面获取维度和列表数 */
    int lists = 0;
    int dimensions = 0;
    IvfflatGetMetaPageInfo(index, &lists, &dimensions);
    so->dimensions = dimensions;
    so->tmpCtx = tmpCtx;
    
    /* 设置批量查询数据 */
    so->batch_keys = batch_keys;
    so->current_query_index = 0;
    so->batch_processing_complete = false;
    so->result_buffer = NULL;
    
#ifdef USE_CUDA
    /* 初始化GPU相关字段 */
    so->centers_uploaded = false;
    so->cuda_ctx = NULL;
    so->gpu_batch_distances = NULL;
    
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
    so->listPages = (BlockNumber*)MemoryContextAlloc(tmpCtx, 
                                                     batch_keys->nkeys * maxProbes * sizeof(BlockNumber));
    so->lists = (IvfflatScanList*)MemoryContextAlloc(tmpCtx, 
                                                     maxProbes * sizeof(IvfflatScanList));
    
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
ivfflatbatchgettuple(IndexScanDesc scan, ScanDirection dir, Datum* values, bool* isnull, int max_tuples, int* returned_tuples, int k)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    
    if (!so->batch_processing_complete) {
        ProcessBatchQueriesGPU(scan, so->batch_keys, k);
        so->batch_processing_complete = true;
    }
    
    if (so->result_buffer) {
        // 处理所有查询的结果，而不是只处理当前查询
        int total_results = 0;
        int nbatch = so->batch_keys->nkeys;
        
        elog(LOG, "ivfflatbatchgettuple: 开始处理结果, nbatch=%d, k=%d, max_tuples=%d", 
             nbatch, k, max_tuples);
        
        /* max_tuples 实际上是 max_values（数组大小），每个结果有3个字段 */
        int max_results = max_tuples / 3;
        
        for (int query_idx = 0; query_idx < nbatch; query_idx++) {
            /* 检查写入后是否会超出限制 */
            int expected_total = total_results + k;  /* 每个查询最多k个结果 */
            if (expected_total > max_results) {
                elog(WARNING, "ivfflatbatchgettuple: 结果数量将超出限制, total_results=%d, expected_total=%d, max_results=%d", 
                     total_results, expected_total, max_results);
                /* 只处理能容纳的结果 */
                int remaining_slots = max_results - total_results;
                if (remaining_slots <= 0) {
                    break;
                }
            }
            
            int query_results = 0;
            elog(LOG, "ivfflatbatchgettuple: 处理查询 %d, total_results=%d, max_results=%d", 
                 query_idx, total_results, max_results);
            GetBatchResults(so->result_buffer, query_idx, k, 
                          values + total_results * 3, 
                          isnull + total_results * 3, 
                          &query_results);
            elog(LOG, "ivfflatbatchgettuple: 查询 %d 返回 %d 个结果", query_idx, query_results);
            total_results += query_results;
            
            /* 再次检查是否超出（实际写入后） */
            if (total_results * 3 > max_tuples) {
                elog(WARNING, "ivfflatbatchgettuple: 结果数量超出 max_tuples, total_results=%d, max_tuples=%d", 
                     total_results, max_tuples);
                break;
            }
        }
        
        elog(LOG, "ivfflatbatchgettuple: 完成, 总结果数=%d", total_results);
        *returned_tuples = total_results;
        return total_results > 0;
    }
    
    elog(LOG, "ivfflatbatchgettuple: result_buffer 为 NULL");
    return false;
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
    
    /* 保存tmpCtx指针，因为删除tmpCtx后so也会被删除 */
    MemoryContext tmpCtx = so->tmpCtx;
    
    /* 在删除tmpCtx之前，先将scan->opaque设置为NULL，避免后续访问已删除的内存 */
    scan->opaque = NULL;
    
    /* 删除临时内存上下文，自动清理所有在tmpCtx中分配的内存 */
    if (tmpCtx) {
        MemoryContextDelete(tmpCtx);
    }
    
    /* 注意：so、batch_ctx、result_buffer 等都在 tmpCtx 中分配，
     * 删除 tmpCtx 时会自动清理，无需手动释放 */
}

void
ProcessBatchQueriesGPU(IndexScanDesc scan, ScanKeyBatch batch_keys, int k)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    int nbatch = batch_keys->nkeys;
    
#ifdef USE_CUDA
    // 注释掉：现在直接使用 integrate_screen，不依赖 cuda_ctx
    // if (so->cuda_ctx && so->cuda_ctx->initialized) {
        /* 使用新的batch_search_pipeline */
        ProcessBatchQueriesGPU_NewPipeline(scan, batch_keys, k);
        return;
    // }
#endif
}

void
GetBatchResults(BatchBuffer* buffer, int query_index, int k, Datum* values, bool* isnull, int* returned_count)
{
    int i;
    int count = 0;
    
    elog(LOG, "GetBatchResults: 开始, buffer=%p, query_index=%d, k=%d, total_results=%d", 
         buffer, query_index, k, buffer ? buffer->total_results : 0);
    
    if (buffer == NULL || buffer->total_results == 0) {
        elog(LOG, "GetBatchResults: buffer 为 NULL 或 total_results=0");
        *returned_count = 0;
        return;
    }
    
    /* 验证 buffer 的字段 */
    if (buffer->query_ids == NULL) {
        elog(ERROR, "GetBatchResults: buffer->query_ids 为 NULL");
        *returned_count = 0;
        return;
    }
    if (buffer->vector_ids == NULL) {
        elog(ERROR, "GetBatchResults: buffer->vector_ids 为 NULL");
        *returned_count = 0;
        return;
    }
    if (buffer->distances == NULL) {
        elog(ERROR, "GetBatchResults: buffer->distances 为 NULL");
        *returned_count = 0;
        return;
    }
    
    elog(LOG, "GetBatchResults: buffer 字段验证通过, 开始遍历结果");
    
    /* 直接从按列存储的数组中获取结果 */
    /* 注意：每个查询都有k个槽位，可能包含null值（vector_id == -1） */
    for (i = 0; i < buffer->total_results && count < k; i++) {
        /* 检查是否是当前查询的结果 */
        if (buffer->query_ids[i] == query_index) {
            /* 边界检查 */
            if (count >= k) {
                break;
            }
            
            elog(LOG, "GetBatchResults: 找到匹配结果, i=%d, count=%d", i, count);
            
            /* 检查是否是null值（使用ItemPointerIsValid检查） */
            bool is_null = !ItemPointerIsValid(&buffer->vector_ids[i]);
            
            if (is_null) {
                /* null值：设置所有字段为null
                 * 注意：在PostgreSQL中，当isnull=true时，value的值不会被使用
                 * 但我们使用(Datum) 0作为标准约定，这是PostgreSQL的NULL Datum值
                 * 即使真实值可能是0，只要isnull=false就不会混淆
                 */
                values[count * 3 + 0] = (Datum) 0;  /* query_id: NULL */
                values[count * 3 + 1] = (Datum) 0;  /* vector_id: NULL */
                values[count * 3 + 2] = (Datum) 0;  /* distance: NULL */
                
                isnull[count * 3 + 0] = true;
                isnull[count * 3 + 1] = true;
                isnull[count * 3 + 2] = true;
            } else {
                /* 有效值：设置返回值 */
                values[count * 3 + 0] = Int32GetDatum(buffer->query_ids[i]); /* query_id */
                /* vector_id: 将 ItemPointer 转换为 integer
                 * ItemPointer 包含 block number 和 offset number
                 * 在 PostgreSQL 中，通常使用 block number 和 offset number 的组合来表示 TID
                 * 但这里我们需要一个整数，可以使用 offset number
                 * 注意：如果 vector_id 需要是全局唯一的，可能需要使用 block number 和 offset number 的组合
                 */
                {
                    /* 使用 offset number 作为 vector_id */
                    /* 注意：ItemPointerGetOffsetNumber 接受 ItemPointer 参数（ItemPointerData*） */
                    elog(LOG, "GetBatchResults: 访问 vector_ids[%d], 地址=%p", i, (void*)&buffer->vector_ids[i]);
                    int vector_id_value = (int)ItemPointerGetOffsetNumber(&buffer->vector_ids[i]);
                    elog(LOG, "GetBatchResults: vector_id_value=%d", vector_id_value);
                    values[count * 3 + 1] = Int32GetDatum(vector_id_value); /* vector_id: integer */
                }
                elog(LOG, "GetBatchResults: 访问 distances[%d], 值=%.6f", i, buffer->distances[i]);
                values[count * 3 + 2] = Float8GetDatum(buffer->distances[i]); /* distance */
                
                /* 设置非空标志 */
                isnull[count * 3 + 0] = false;
                isnull[count * 3 + 1] = false;
                isnull[count * 3 + 2] = false;
                
                elog(LOG, "GetBatchResults: 结果设置完成, count=%d", count);
            }
            
            count++;
        }
    }
    
    elog(LOG, "GetBatchResults: 完成, 返回 count=%d", count);
    *returned_count = count;
}


BatchBuffer*
CreateBatchBuffer(int n_queries, int k, int dimensions, MemoryContext ctx)
{
    BatchBuffer* buffer = MemoryContextAllocZero(ctx, sizeof(BatchBuffer));
    int total_results = n_queries * k;
    
    /* 分配内存 - 按列存储，在指定上下文中分配 */
    buffer->query_data = MemoryContextAlloc(ctx, n_queries * dimensions * sizeof(float));
    buffer->query_ids = MemoryContextAlloc(ctx, total_results * sizeof(int));
    buffer->vector_ids = MemoryContextAlloc(ctx, total_results * sizeof(ItemPointerData));
    buffer->distances = MemoryContextAlloc(ctx, total_results * sizeof(float));
    
    buffer->n_queries = n_queries;
    buffer->k = k;
    buffer->total_results = total_results;
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

void
GetScanItems_BatchGPU(IndexScanDesc scan, ScanKeyBatch batch_keys)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    int nbatch = batch_keys->nkeys;
    int k = so->result_buffer ? so->result_buffer->k : 10;  /* 默认top-k */
    
    if (!so->cuda_ctx || !so->cuda_ctx->initialized) {
        elog(ERROR, "GetScanItems_BatchGPU: CUDA上下文未初始化");
        return;
    }
    
    /* 确保probes数据已上传 */
    if (!so->cuda_ctx->probes_uploaded) {
        // elog(LOG, "GetScanItems_BatchGPU: probes数据未上传，开始上传（分离存储方案）");
        if (UploadIndexTuplesToGPU_Batch(scan) != 0) {
            elog(ERROR, "GetScanItems_BatchGPU: probes数据上传失败");
            return;
        }
    }
    
    /* 确保result_buffer已创建 */
    if (so->result_buffer == NULL) {
        so->result_buffer = CreateBatchBuffer(nbatch, k, so->dimensions, so->tmpCtx);
    }
    
    /* 步骤1: GPU批量计算距离 */
    /* 注意：向量是独立存储的，需要逐个提取并打包成连续数组 */
    float *batch_query_data = (float*)palloc(nbatch * so->dimensions * sizeof(float));
    if (!batch_query_data) {
        elog(ERROR, "GetScanItems_BatchGPU: 无法分配批量查询数据内存");
        return;
    }
    
    /* 提取每个查询向量的数据部分，打包成连续数组 */
    for (int i = 0; i < nbatch; i++) {
        Datum vector_datum = ScanKeyBatchGetVector(batch_keys, i);
        Vector *vec = DatumGetVector(vector_datum);
        /* 复制向量数据到连续数组 */
        memcpy(batch_query_data + i * so->dimensions, &vec->x[0], so->dimensions * sizeof(float));
        
        /* 调试：输出查询向量的前几个元素 */
        // elog(LOG, "GetScanItems_BatchGPU: 查询%d向量前3个元素: [%.6f, %.6f, %.6f] (应该对应查询%d)", 
        //      i, vec->x[0], vec->x[1], vec->x[2], i);
    }
    
    // elog(LOG, "GetScanItems_BatchGPU: 开始在GPU上计算批量probe距离");
    int compute_result = cuda_compute_batch_probe_distances(so->cuda_ctx, 
                                                           batch_query_data, 
                                                           nbatch);
    
    /* 释放临时内存 */
    pfree(batch_query_data);
    
    if (compute_result != 0) {
        elog(ERROR, "GetScanItems_BatchGPU: GPU批量距离计算失败");
        return;
    }
    
    /* 步骤2: GPU TopK选择 */
    int *topk_query_ids = (int*)palloc(nbatch * k * sizeof(int));
    ItemPointerData *topk_vector_ids = (ItemPointerData*)palloc(nbatch * k * sizeof(ItemPointerData));
    float *topk_distances = (float*)palloc(nbatch * k * sizeof(float));
    int *topk_counts = (int*)palloc0(nbatch * sizeof(int));
    
    if (!topk_query_ids || !topk_vector_ids || !topk_distances || !topk_counts) {
        elog(ERROR, "GetScanItems_BatchGPU: 无法分配TopK结果内存");
        if (topk_query_ids) pfree(topk_query_ids);
        if (topk_vector_ids) pfree(topk_vector_ids);
        if (topk_distances) pfree(topk_distances);
        if (topk_counts) pfree(topk_counts);
        return;
    }
    
    // elog(LOG, "GetScanItems_BatchGPU: 开始在GPU上进行TopK选择 - k=%d, nbatch=%d, 候选数=%d", 
    //      k, nbatch, so->cuda_ctx ? so->cuda_ctx->num_probe_candidates : -1);
    int topk_result = cuda_topk_probe_candidates(so->cuda_ctx, 
                                                 k, 
                                                 nbatch,
                                                 topk_query_ids, 
                                                 topk_vector_ids, 
                                                 topk_distances, 
                                                 topk_counts);
    
    if (topk_result != 0) {
#ifdef USE_CUDA
        const char* cuda_err = cuda_get_last_error_string();
        elog(ERROR, "GetScanItems_BatchGPU: GPU TopK选择失败 - k=%d, nbatch=%d, 候选数=%d, 维度=%d, CUDA错误: %s", 
             k, nbatch, 
             so->cuda_ctx ? so->cuda_ctx->num_probe_candidates : -1,
             so->dimensions,
             cuda_err);
#else
        elog(ERROR, "GetScanItems_BatchGPU: GPU TopK选择失败 - k=%d, nbatch=%d, 候选数=%d, 维度=%d", 
             k, nbatch, 
             so->cuda_ctx ? so->cuda_ctx->num_probe_candidates : -1,
             so->dimensions);
#endif
        pfree(topk_query_ids);
        pfree(topk_vector_ids);
        pfree(topk_distances);
        pfree(topk_counts);
        return;
    }
    
    /* 步骤3: 填充到result_buffer（按列存储） */
    /* 每个查询必须填充k个位置，如果候选不足k个，用null值填充 */
    int total_processed_results = 0;  /* 累计已处理的GPU结果数 */
    int total_results = 0;            /* 总的缓冲区槽位数 */
    for (int query_idx = 0; query_idx < nbatch; query_idx++) {
        int count = topk_counts[query_idx];
        
        /* 填充实际结果（如果有的话） */
        for (int i = 0; i < count; i++) {
            int buffer_idx = query_idx * k + i;           /* 目标缓冲区位置 */
            int source_idx = total_processed_results + i; /* GPU结果数组中的源位置 */
            
            so->result_buffer->query_ids[buffer_idx] = topk_query_ids[source_idx];
            so->result_buffer->vector_ids[buffer_idx] = topk_vector_ids[source_idx];
            so->result_buffer->distances[buffer_idx] = topk_distances[source_idx];
            
            /* 调试日志：验证索引修复效果（仅前几个结果） */
            // if (query_idx < 3 && i < 2) {
            //     elog(LOG, "GetScanItems_BatchGPU: 查询%d结果%d - buffer_idx=%d, source_idx=%d, query_id=%d, distance=%.6f", 
            //          query_idx, i, buffer_idx, source_idx, topk_query_ids[source_idx], topk_distances[source_idx]);
            // }
        }
        
        /* 如果候选不足k个，用null值填充剩余位置 */
        for (int i = count; i < k; i++) {
            int buffer_idx = query_idx * k + i;
            so->result_buffer->query_ids[buffer_idx] = query_idx;
            ItemPointerSetInvalid(&so->result_buffer->vector_ids[buffer_idx]);  /* null标记：无效的TID */
            so->result_buffer->distances[buffer_idx] = INFINITY;  /* null标记：无效的距离 */
        }
        
        total_processed_results += count;  /* 累计已处理的GPU结果数 */
        total_results += k;                /* 每个查询贡献k个槽位 */
    }
    
    so->result_buffer->total_results = total_results;
    
    pfree(topk_query_ids);
    pfree(topk_vector_ids);
    pfree(topk_distances);
    pfree(topk_counts);
    
    // elog(LOG, "GetScanItems_BatchGPU: 成功获取 %d 个结果（GPU计算距离和TopK）", total_results);
}

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
    MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);
    
    int totalLists = 0;
    IvfflatGetMetaPageInfo(scan->indexRelation, &totalLists, NULL);
    
    elog(LOG, "PrepareBatchDataForPipeline: totalLists=%d", totalLists);
    
    if (totalLists <= 0) {
        elog(ERROR, "PrepareBatchDataForPipeline: totalLists <= 0");
        MemoryContextSwitchTo(oldCtx);
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
                MemoryContextSwitchTo(oldCtx);
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
        MemoryContextSwitchTo(oldCtx);
        return -1;
    }
    
    elog(LOG, "PrepareBatchDataForPipeline: 开始复制 query_batch 数据, nbatch=%d", nbatch);
    
    for (int i = 0; i < nbatch; i++) {
        elog(LOG, "PrepareBatchDataForPipeline: 处理 query %d", i);
        Datum vector_datum = ScanKeyBatchGetVector(batch_keys, i);
        if (vector_datum == (Datum) NULL) {
            elog(ERROR, "PrepareBatchDataForPipeline: query %d 的向量数据为 NULL", i);
            MemoryContextSwitchTo(oldCtx);
            return -1;
        }
        Vector *vec = DatumGetVector(vector_datum);
        if (!vec) {
            elog(ERROR, "PrepareBatchDataForPipeline: query %d 的 Vector 指针为 NULL", i);
            MemoryContextSwitchTo(oldCtx);
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
        MemoryContextSwitchTo(oldCtx);
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
                            mapping_idx++;
                            vector_idx++;
                        }
                    }
                    
                    searchPage = IvfflatPageGetOpaque(page)->nextblkno;
                    UnlockReleaseBuffer(buf);
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
    
    MemoryContextSwitchTo(oldCtx);
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
    int total_vectors = 0;
    for (int i = 0; i < n_total_clusters; i++) {
        total_vectors += cluster_size[i];
    }
    
    int total_results = 0;
    for (int query_idx = 0; query_idx < n_query; query_idx++) {
        for (int i = 0; i < k; i++) {
            int buffer_idx = query_idx * k + i;
            int global_vector_idx = topk_index[query_idx][i];
            
            if (global_vector_idx >= 0 && global_vector_idx < total_vectors) {
                /* 直接使用全局索引在mapping_table中查找 */
                VectorIndexMapping* mapping = &mapping_table[global_vector_idx];
                result_buffer->query_ids[buffer_idx] = query_idx;
                /* 确保 tid 是有效的 ItemPointer */
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
                    elog(LOG, "ConvertBatchPipelineResults: 查询%d结果%d - global_vector_idx=%d, cluster_id=%d, vector_idx_in_cluster=%d, distance=%.10f, tid=(%u,%u)", 
                         query_idx, i, global_vector_idx, mapping->cluster_id, 
                         mapping->vector_index_in_cluster, topk_dist[query_idx][i],
                         ItemPointerGetBlockNumber(&mapping->tid),
                         ItemPointerGetOffsetNumber(&mapping->tid));
                }
                
                total_results++;
            } else {
                /* null值或无效索引 */
                elog(LOG, "ConvertBatchPipelineResults: 无效的 global_vector_idx=%d (total_vectors=%d), query_idx=%d, i=%d", 
                     global_vector_idx, total_vectors, query_idx, i);
                result_buffer->query_ids[buffer_idx] = query_idx;
                ItemPointerSetInvalid(&result_buffer->vector_ids[buffer_idx]);
                result_buffer->distances[buffer_idx] = INFINITY;
            }
        }
    }
    
    result_buffer->total_results = total_results > 0 ? total_results : n_query * k;
}

/* C++包装函数声明（在C++文件中实现，通过extern "C"导出） */
/* 新接口：接受 device 指针 */
extern int batch_search_pipeline_wrapper(float* d_query_batch,
                                         int* d_cluster_size,
                                         float* d_cluster_vectors,
                                         float* d_cluster_centers,
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
                 float* d_cluster_centers, float* d_topk_dist, int* d_topk_index)
{
    if (d_query_batch) cudaFree(d_query_batch);
    if (d_cluster_size) cudaFree(d_cluster_size);
    if (d_cluster_vectors) cudaFree(d_cluster_vectors);
    if (d_cluster_centers) cudaFree(d_cluster_centers);
    if (d_topk_dist) cudaFree(d_topk_dist);
    if (d_topk_index) cudaFree(d_topk_index);
}

static void
ProcessBatchQueriesGPU_NewPipeline(IndexScanDesc scan, ScanKeyBatch batch_keys, int k)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    int nbatch = batch_keys->nkeys;
    int dimensions = so->dimensions;
    
    /* 切换到 tmpCtx，确保所有 palloc 分配的内存都在 tmpCtx 中 */
    /* 这样在 tmpCtx 被删除时会自动清理，无需手动释放 */
    MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);
    
    /* 准备数据 */
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
    float* d_topk_dist = NULL;
    int* d_topk_index = NULL;
    
    /* 输出缓冲区 */
    float** topk_dist = NULL;
    int** topk_index = NULL;
    
    cudaError_t cuda_err;
    
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: 开始执行, nbatch=%d, dimensions=%d, probes=%d, k=%d",
         nbatch, dimensions, so->probes, k);
    
    /* ========== 数据准备 ========== */
    if (PrepareBatchDataForPipeline(scan, batch_keys, so->probes,
                                    &query_batch, &cluster_size,
                                    &cluster_vectors, &cluster_center_data,
                                    &mapping_table, &cluster_pages,
                                    &n_total_clusters, &n_total_vectors) != 0) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: 数据准备失败");
        MemoryContextSwitchTo(oldCtx);
        return;
    }
    
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: 数据准备完成, n_total_clusters=%d, n_total_vectors=%d",
         n_total_clusters, n_total_vectors);
    
    /* 检查数据有效性 */
    if (!query_batch || !query_batch[0] || !cluster_size || 
        !cluster_center_data || !cluster_center_data[0] ||
        n_total_clusters <= 0 || n_total_vectors <= 0) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: 数据验证失败");
        MemoryContextSwitchTo(oldCtx);
        return;
    }
    
    /* 计算并记录每个分配的大小 */
    size_t query_batch_size = (size_t)nbatch * (size_t)dimensions * sizeof(float);
    size_t cluster_size_size = (size_t)n_total_clusters * sizeof(int);
    size_t cluster_vectors_size = (size_t)n_total_vectors * (size_t)dimensions * sizeof(float);
    size_t cluster_centers_size = (size_t)n_total_clusters * (size_t)dimensions * sizeof(float);
    size_t topk_dist_size = (size_t)nbatch * (size_t)k * sizeof(float);
    size_t topk_index_size = (size_t)nbatch * (size_t)k * sizeof(int);
    
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: GPU 内存分配计划 - query_batch=%zu bytes, cluster_size=%zu bytes, cluster_vectors=%zu bytes, cluster_centers=%zu bytes, topk_dist=%zu bytes, topk_index=%zu bytes",
         query_batch_size, cluster_size_size, cluster_vectors_size, cluster_centers_size, topk_dist_size, topk_index_size);
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: GPU 内存分配计划 - 总计=%zu bytes (%.2f MB)",
         query_batch_size + cluster_size_size + cluster_vectors_size + cluster_centers_size + topk_dist_size + topk_index_size,
         (query_batch_size + cluster_size_size + cluster_vectors_size + cluster_centers_size + topk_dist_size + topk_index_size) / (1024.0 * 1024.0));
    
    /* ========== GPU 内存分配 ========== */
    cuda_err = cudaMalloc(&d_query_batch, query_batch_size);
    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: 无法分配 query batch GPU 内存 (%zu bytes): %s", 
            query_batch_size, cudaGetErrorString(cuda_err));
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: d_query_batch 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_query_batch, query_batch_size);
    cuda_err = cudaMalloc(&d_cluster_size, cluster_size_size);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: 无法分配 cluster_size GPU 内存 (%zu bytes): %s", 
            cluster_size_size, cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, NULL, NULL, NULL, NULL, NULL);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: d_cluster_size 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_cluster_size, cluster_size_size);
    cuda_err = cudaMalloc(&d_cluster_vectors, cluster_vectors_size);


    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: 无法分配 cluster_vectors GPU 内存 (%zu bytes): %s", 
            cluster_vectors_size, cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, NULL, NULL, NULL, NULL);
        MemoryContextSwitchTo(oldCtx);
        return;
    }      
    
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: d_cluster_vectors 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_cluster_vectors, cluster_vectors_size);
    
    cuda_err = cudaMalloc(&d_cluster_centers, cluster_centers_size);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: 无法分配 cluster_centers GPU 内存 (%zu bytes): %s", 
            cluster_centers_size, cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        NULL, NULL, NULL);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: d_cluster_centers 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_cluster_centers, cluster_centers_size);
    
    cuda_err = cudaMalloc(&d_topk_dist, topk_dist_size);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: 无法分配 topk_dist GPU 内存 (%zu bytes): %s", 
            topk_dist_size, cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, NULL, NULL);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: d_topk_dist 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_topk_dist, topk_dist_size);
    
    cuda_err = cudaMalloc(&d_topk_index, topk_index_size);    


    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: 无法分配 topk_index GPU 内存 (%zu bytes): %s", 
            topk_index_size, cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_topk_dist, NULL);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: d_topk_index 分配成功, 地址=%p, 大小=%zu bytes", 
        (void*)d_topk_index, topk_index_size);
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: GPU 内存分配完成");
    
    /* ========== CPU -> GPU 数据复制 ========== */
    if (!query_batch || !query_batch[0]) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: query_batch 为空，无法复制");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    cuda_err = cudaMemcpy(d_query_batch, query_batch[0], 
        nbatch * dimensions * sizeof(float), 
        cudaMemcpyHostToDevice);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: query_batch 复制失败: %s", 
                cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    /*H2D: cluster size*/
    cuda_err = cudaMemcpy(d_cluster_size, cluster_size, 
        n_total_clusters * sizeof(int), 
        cudaMemcpyHostToDevice);
    
    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: cluster_size 复制失败: %s", 
            cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    /*H2D: cluster_vectors：使用 cudaMemcpy 逐个复制每个 cluster */
    int gpu_offset = 0;
    int copied_clusters = 0;
    int copied_vectors = 0;
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: 开始复制 cluster_vectors, n_total_clusters=%d, n_total_vectors=%d", 
            n_total_clusters, n_total_vectors);
    
    int copy_success = 1;
    for (int cid = 0; cid < n_total_clusters && copy_success; cid++) {
        if (cluster_size[cid] <= 0 || !cluster_vectors || !cluster_vectors[cid] || !cluster_vectors[cid][0]) {
            elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: 跳过 cluster %d (cluster_size=%d, cluster_vectors=%p), gpu_offset保持不变=%d", 
                    cid, cluster_size[cid], cluster_vectors ? (cluster_vectors[cid] ? cluster_vectors[cid][0] : NULL) : NULL, gpu_offset);
            /* 重要：即使跳过空cluster，gpu_offset也要增加0（保持不变），确保后续cluster的位置正确 */
            /* 但这里 cluster_size[cid] <= 0，所以 gpu_offset += 0 是正确的 */
            continue;
        }

        size_t cluster_bytes = cluster_size[cid] * dimensions * sizeof(float);
        elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: 复制 cluster %d, cluster_size=%d, gpu_offset=%d, bytes=%zu", 
                cid, cluster_size[cid], gpu_offset, cluster_bytes);
        cuda_err = cudaMemcpy(d_cluster_vectors + gpu_offset * dimensions,
                                cluster_vectors[cid][0],
                                cluster_bytes,
                                cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: cluster_vectors[%d] 复制失败: %s", 
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
            d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    if (gpu_offset != n_total_vectors) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: cluster_vectors 复制数量不匹配! gpu_offset=%d, n_total_vectors=%d", 
            gpu_offset, n_total_vectors);
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: cluster_vectors 复制完成, copied_clusters=%d, copied_vectors=%d, gpu_offset=%d, expected_total=%d", 
            copied_clusters, copied_vectors, gpu_offset, n_total_vectors);

    if (!cluster_center_data || !cluster_center_data[0]) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: cluster_center_data 为空，无法复制");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    /* H2D: cluster_centers */
    cuda_err = cudaMemcpy(d_cluster_centers, cluster_center_data[0], 
        n_total_clusters * dimensions * sizeof(float), 
        cudaMemcpyHostToDevice);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: cluster_centers 复制失败: %s", 
            cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: CPU->GPU 数据复制完成, nbatch=%d, n_total_clusters=%d, n_total_vectors=%d, probes=%d, k=%d",
            nbatch, n_total_clusters, n_total_vectors, so->probes, k);
    
    /* ========== 调用 GPU 计算 ========== */
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: 准备调用 batch_search_pipeline_wrapper");
    
    /* 详细参数验证 */
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: ========== GPU内存指针验证 ==========");
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: d_query_batch=%p (size=%zu bytes, %d queries × %d dims)", 
            (void*)d_query_batch, (size_t)nbatch * dimensions * sizeof(float), nbatch, dimensions);
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: d_cluster_size=%p (size=%zu bytes, %d clusters)", 
            (void*)d_cluster_size, (size_t)n_total_clusters * sizeof(int), n_total_clusters);
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: d_cluster_vectors=%p (size=%zu bytes, %d vectors × %d dims)", 
            (void*)d_cluster_vectors, (size_t)n_total_vectors * dimensions * sizeof(float), n_total_vectors, dimensions);
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: d_cluster_centers=%p (size=%zu bytes, %d clusters × %d dims)", 
            (void*)d_cluster_centers, (size_t)n_total_clusters * dimensions * sizeof(float), n_total_clusters, dimensions);
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: d_topk_dist=%p (size=%zu bytes, %d queries × %d k)", 
            (void*)d_topk_dist, (size_t)nbatch * k * sizeof(float), nbatch, k);
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: d_topk_index=%p (size=%zu bytes, %d queries × %d k)", 
            (void*)d_topk_index, (size_t)nbatch * k * sizeof(int), nbatch, k);
    
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: ========== 参数值验证 ==========");
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: n_query=%d, n_dim=%d, n_total_cluster=%d, n_total_vectors=%d, n_probes=%d, k=%d",
            nbatch, dimensions, n_total_clusters, n_total_vectors, so->probes, k);
    
    /* 验证 cluster_size 数组 */
    int sum_cluster_sizes = 0;
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: cluster_size数组内容 (前10个):");
    for (int i = 0; i < n_total_clusters && i < 10; i++) {
        sum_cluster_sizes += cluster_size[i];
        elog(LOG, "ProcessBatchQueriesGPU_NewPipeline:   cluster_size[%d]=%d", i, cluster_size[i]);
    }
    if (n_total_clusters > 10) {
        elog(LOG, "ProcessBatchQueriesGPU_NewPipeline:   ... (共%d个clusters)", n_total_clusters);
    }
    for (int i = 10; i < n_total_clusters; i++) {
        sum_cluster_sizes += cluster_size[i];
    }
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: cluster_size 数组总和=%d, n_total_vectors=%d", 
            sum_cluster_sizes, n_total_vectors);
    
    /* 验证 query_batch 数据（前几个值） */
    if (query_batch && query_batch[0]) {
        elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: query_batch[0] 前5个值: %.6f, %.6f, %.6f, %.6f, %.6f",
                query_batch[0][0], query_batch[0][1], query_batch[0][2], query_batch[0][3], query_batch[0][4]);
    }
    
    /* 验证 cluster_centers 数据（第一个cluster的前几个值） */
    if (cluster_center_data && cluster_center_data[0]) {
        elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: cluster_centers[0] 前5个值: %.6f, %.6f, %.6f, %.6f, %.6f",
                cluster_center_data[0][0], cluster_center_data[0][1], cluster_center_data[0][2], 
                cluster_center_data[0][3], cluster_center_data[0][4]);
    }
    
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: ========== 调用 batch_search_pipeline_wrapper ==========");
    
    if (sum_cluster_sizes != n_total_vectors) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: cluster_size 数组总和(%d) 与 n_total_vectors(%d) 不匹配!", 
            sum_cluster_sizes, n_total_vectors);
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    int result = batch_search_pipeline_wrapper(d_query_batch, d_cluster_size, d_cluster_vectors,
                                                d_cluster_centers, d_topk_dist, d_topk_index,
                                                nbatch, dimensions, n_total_clusters,
                                                n_total_vectors, so->probes, k);
    
    if (result != 0) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: batch_search_pipeline执行失败 (返回码: %d), 请查看PostgreSQL日志获取详细信息", result);
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    /* ========== GPU -> CPU 结果回传 ========== */
    /* 分配连续的内存块用于存储所有查询的结果 */
    float* topk_dist_flat = (float*)palloc(nbatch * k * sizeof(float));
    int* topk_index_flat = (int*)palloc(nbatch * k * sizeof(int));
    
    if (!topk_dist_flat || !topk_index_flat){
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: 无法分配输出缓冲区");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }
    /* 复制结果到 host（连续数组） */
    cuda_err = cudaMemcpy(topk_dist_flat, d_topk_dist, 
                            nbatch * k * sizeof(float), 
                            cudaMemcpyDeviceToHost);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: topk_index 回传失败: %s", 
            cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    cuda_err = cudaMemcpy(topk_index_flat, d_topk_index, 
                        nbatch * k * sizeof(int), 
                        cudaMemcpyDeviceToHost);

    if (cuda_err != cudaSuccess) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: topk_dist 回传失败: %s", 
            cudaGetErrorString(cuda_err));
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    /* 分配指针数组 */
    topk_dist = (float**)palloc(nbatch * sizeof(float*));
    topk_index = (int**)palloc(nbatch * sizeof(int*));

    if(!topk_dist || !topk_index){
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: 无法分配指针数组");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    /* 重新组织为指针数组格式（用于 ConvertBatchPipelineResults） */
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: 开始重新组织指针数组, nbatch=%d", nbatch);
    for (int i = 0; i < nbatch; i++) {
        topk_dist[i] = topk_dist_flat + i * k;
        topk_index[i] = topk_index_flat + i * k;
    }
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: 指针数组重新组织完成");
    
    /* 调试：打印GPU返回的前几个结果 */
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: GPU返回的结果（前3个查询，每个查询前3个结果）:");
    for (int query_idx = 0; query_idx < nbatch && query_idx < 3; query_idx++) {
        elog(LOG, "  查询 %d:", query_idx);
        for (int i = 0; i < k && i < 3; i++) {
            int idx = query_idx * k + i;
            elog(LOG, "    结果 %d: global_vector_idx=%d, distance=%.10f", 
                 i, topk_index_flat[idx], topk_dist_flat[idx]);
        }
    }
    
    /* ========== 转换结果到 BatchBuffer ========== */
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: 开始转换结果到BatchBuffer");
                    
    if (!mapping_table || !cluster_size) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: mapping_table 或 cluster_size 为 NULL");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    if (so->result_buffer == NULL) {/*这个检查可能没有什么用*/
        so->result_buffer = CreateBatchBuffer(nbatch, k, dimensions, so->tmpCtx);
    }
                        
    if (!so->result_buffer) {
        elog(ERROR, "ProcessBatchQueriesGPU_NewPipeline: 无法创建BatchBuffer");
        CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
            d_cluster_centers, d_topk_dist, d_topk_index);
        MemoryContextSwitchTo(oldCtx);
        return;
    }

    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: 调用 ConvertBatchPipelineResults");
    ConvertBatchPipelineResults(scan, topk_dist, topk_index, nbatch, k,
                                so->result_buffer, mapping_table, cluster_size,
                                cluster_pages, n_total_clusters);
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: ConvertBatchPipelineResults 完成");
    elog(LOG, "ProcessBatchQueriesGPU_NewPipeline: 结果转换完成, 总结果数=%d", 
            so->result_buffer ? so->result_buffer->total_results : 0);
    
    /* ========== 清理 GPU 内存 ========== */
    CleanupGPUMemory(d_query_batch, d_cluster_size, d_cluster_vectors,
                    d_cluster_centers, d_topk_dist, d_topk_index);
    MemoryContextSwitchTo(oldCtx);
    return;
}

#endif
