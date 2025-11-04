#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/relscan.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "ivfscanbatch.h"
#include "scanbatch.h"
#include "ivfflat.h"
#include "vector.h"

/* 内部函数声明 */

#ifdef USE_CUDA
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

    elog(LOG, "ivfflatbatchbeginscan: 开始批量扫描, nkeys=%d", batch_keys->nkeys);

    /* 创建扫描描述符 */
    scan = RelationGetIndexScan(index, batch_keys->nkeys, norderbys);
    if (!scan) {
        elog(ERROR, "ivfflatbatchbeginscan: 无法创建扫描描述符");
    }
    
    /* 先创建临时内存上下文 */
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
    so->cuda_ctx = cuda_center_search_init(lists, dimensions, false);
    if (!so->cuda_ctx) {
        elog(WARNING, "CUDA上下文初始化失败，将使用CPU模式");
    } else {
        /* 分配GPU批量距离结果内存 */
        so->gpu_batch_distances = (float*)palloc(batch_keys->nkeys * lists * sizeof(float));
    }
#endif

    /* 设置scan->opaque */
    scan->opaque = so;

    elog(LOG, "ivfflatbatchbeginscan: 批量扫描初始化完成");
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
        
        for (int query_idx = 0; query_idx < nbatch; query_idx++) {
            int query_results = 0;
            GetBatchResults(so->result_buffer, query_idx, k, 
                          values + total_results * 3, 
                          isnull + total_results * 3, 
                          &query_results);
            total_results += query_results;
        }
        
        *returned_tuples = total_results;
        return total_results > 0;
    }
    
    return false;
}

void
ivfflatbatchendscan(IndexScanDesc scan)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;

#ifdef USE_CUDA
    /* 清理CUDA资源（如果使用了） */
    if (so->cuda_ctx) {
        cuda_center_search_cleanup((CudaCenterSearchContext*)so->cuda_ctx);
    }
#endif
    
    /* 删除临时内存上下文，自动清理所有在tmpCtx中分配的内存 */
    if (so->tmpCtx) {
        MemoryContextDelete(so->tmpCtx);
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
    if (so->cuda_ctx && so->cuda_ctx->initialized) {
        /* 步骤1: 选择最近的列表（probes） */
        elog(LOG, "ProcessBatchQueriesGPU: 步骤1 - 选择最近的列表");
        GetScanLists_BatchGPU(scan, batch_keys);
        
        /* 步骤2: 上传probe候选数据到GPU（使用IndexTuple模式） */
        elog(LOG, "ProcessBatchQueriesGPU: 步骤2 - 上传probe候选数据（IndexTuple模式）");
        if (UploadIndexTuplesToGPU_Batch(scan) != 0) {
            elog(ERROR, "ProcessBatchQueriesGPU: 上传probe候选数据失败");
            return;
        }
        
        /* 步骤3: 从GPU获取结果并填充到result_buffer */
        elog(LOG, "ProcessBatchQueriesGPU: 步骤3 - 获取GPU结果");
        GetScanItems_BatchGPU(scan, batch_keys);
        
        elog(LOG, "ProcessBatchQueriesGPU: GPU处理完成");
        return;
    }
#endif
    
    /* GPU不可用，返回错误 */
    elog(ERROR, "ProcessBatchQueriesGPU: GPU功能不可用，CUDA上下文未初始化");
}

void
GetBatchResults(BatchBuffer* buffer, int query_index, int k, Datum* values, bool* isnull, int* returned_count)
{
    int i;
    int count = 0;
    
    elog(LOG, "GetBatchResults: buffer=%p, query_index=%d, k=%d, total_results=%d", 
         buffer, query_index, k, buffer ? buffer->total_results : 0);
    
    if (buffer == NULL || buffer->total_results == 0) {
        *returned_count = 0;
        return;
    }
    
    /* 直接从按列存储的数组中获取结果 */
    /* 注意：每个查询都有k个槽位，可能包含null值（vector_id == -1） */
    for (i = 0; i < buffer->total_results && count < k; i++) {
        /* 检查是否是当前查询的结果 */
        if (buffer->query_ids[i] == query_index) {
            /* 边界检查 */
            if (count >= k) {
                break;
            }
            
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
                values[count * 3 + 1] = PointerGetDatum(&buffer->vector_ids[i]); /* vector_id: ItemPointer */
                values[count * 3 + 2] = Float8GetDatum(buffer->distances[i]); /* distance */
                
                /* 设置非空标志 */
                isnull[count * 3 + 0] = false;
                isnull[count * 3 + 1] = false;
                isnull[count * 3 + 2] = false;
            }
            
            count++;
        }
    }
    
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
    
    /* 准备批量查询数据 */
    float *batch_query_data = (float *) ScanKeyBatchGetContinuousData(batch_keys);
    int nbatch = batch_keys->nkeys;
    
    if (!batch_query_data) {
        elog(ERROR, "无法获取批量查询数据");
        pfree(list_pages);
        return;
    }
    
    /* GPU批量计算距离 */
    int cuda_result = cuda_compute_batch_center_distances(
        so->cuda_ctx, 
        batch_query_data, 
        nbatch, 
        so->gpu_batch_distances);
    
    if (cuda_result != 0) {
        elog(ERROR, "GPU距离计算失败");
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
        for (int i = outputCount - 1; i >= 0; i--) {
            so->listPages[query_idx * so->maxProbes + i] = GetScanList(pairingheap_remove_first(so->listQueue))->startPage;
        }
        
        Assert(pairingheap_is_empty(so->listQueue));
    }
    
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
        elog(LOG, "GetScanItems_BatchGPU: probes数据未上传，开始上传（IndexTuple模式）");
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
    }
    
    elog(LOG, "GetScanItems_BatchGPU: 开始在GPU上计算批量probe距离");
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
    
    elog(LOG, "GetScanItems_BatchGPU: 开始在GPU上进行TopK选择");
    int topk_result = cuda_topk_probe_candidates(so->cuda_ctx, 
                                                 k, 
                                                 nbatch,
                                                 topk_query_ids, 
                                                 topk_vector_ids, 
                                                 topk_distances, 
                                                 topk_counts);
    
    if (topk_result != 0) {
        elog(ERROR, "GetScanItems_BatchGPU: GPU TopK选择失败");
        pfree(topk_query_ids);
        pfree(topk_vector_ids);
        pfree(topk_distances);
        pfree(topk_counts);
        return;
    }
    
    /* 步骤3: 填充到result_buffer（按列存储） */
    /* 每个查询必须填充k个位置，如果候选不足k个，用null值填充 */
    int total_results = 0;
    for (int query_idx = 0; query_idx < nbatch; query_idx++) {
        int count = topk_counts[query_idx];
        
        /* 填充实际结果（如果有的话） */
        for (int i = 0; i < count; i++) {
            int buffer_idx = query_idx * k + i;
            so->result_buffer->query_ids[buffer_idx] = topk_query_ids[buffer_idx];
            so->result_buffer->vector_ids[buffer_idx] = topk_vector_ids[buffer_idx];
            so->result_buffer->distances[buffer_idx] = topk_distances[buffer_idx];
        }
        
        /* 如果候选不足k个，用null值填充剩余位置 */
        for (int i = count; i < k; i++) {
            int buffer_idx = query_idx * k + i;
            so->result_buffer->query_ids[buffer_idx] = query_idx;
            ItemPointerSetInvalid(&so->result_buffer->vector_ids[buffer_idx]);  /* null标记：无效的TID */
            so->result_buffer->distances[buffer_idx] = INFINITY;  /* null标记：无效的距离 */
        }
        
        total_results += k;  /* 每个查询贡献k个槽位 */
    }
    
    so->result_buffer->total_results = total_results;
    
    pfree(topk_query_ids);
    pfree(topk_vector_ids);
    pfree(topk_distances);
    pfree(topk_counts);
    
    elog(LOG, "GetScanItems_BatchGPU: 成功获取 %d 个结果（GPU计算距离和TopK）", total_results);
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
 * 上传完整的IndexTuple数据到GPU（与CPU端存储方式完全一致）
 * IndexTuple包含：t_tid + t_info + Vector数据
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
    
    /* 分配临时数组收集完整的IndexTuple数据 */
    char **index_tuples = (char**)palloc(estimated_total_candidates * sizeof(char*));
    size_t *tuple_sizes = (size_t*)palloc(estimated_total_candidates * sizeof(size_t));
    int *query_ids = (int*)palloc(estimated_total_candidates * sizeof(int));
    
    if (!index_tuples || !tuple_sizes || !query_ids) {
        elog(ERROR, "UploadIndexTuplesToGPU_Batch: 无法分配IndexTuple数据内存");
        if (index_tuples) pfree(index_tuples);
        if (tuple_sizes) pfree(tuple_sizes);
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
                        index_tuples = (char**)repalloc(index_tuples, estimated_total_candidates * sizeof(char*));
                        tuple_sizes = (size_t*)repalloc(tuple_sizes, estimated_total_candidates * sizeof(size_t));
                        query_ids = (int*)repalloc(query_ids, estimated_total_candidates * sizeof(int));
                    }
                    
                    /* 获取IndexTuple的大小 */
                    Size tuple_size = IndexTupleSize(itup);
                    tuple_sizes[total_candidates] = tuple_size;
                    
                    /* 分配内存并复制完整的IndexTuple */
                    index_tuples[total_candidates] = (char*)palloc(tuple_size);
                    memcpy(index_tuples[total_candidates], itup, tuple_size);
                    
                    /* 调试：验证 TID 值（前几个候选） */
                    if (total_candidates < 3) {
                        ItemPointer tid = &itup->t_tid;
                        elog(LOG, "UploadIndexTuplesToGPU_Batch: 候选 %d - TID: (%u,%u), tuple_size=%zu", 
                             total_candidates,
                             ItemPointerGetBlockNumber(tid),
                             ItemPointerGetOffsetNumber(tid),
                             tuple_size);
                    }
                    
                    /* 收集元数据 */
                    query_ids[total_candidates] = query_idx;
                    
                    total_candidates++;
                }
                
                searchPage = IvfflatPageGetOpaque(page)->nextblkno;
                UnlockReleaseBuffer(buf);
            }
        }
    }
    
    if (total_candidates == 0) {
        elog(WARNING, "UploadIndexTuplesToGPU_Batch: 没有找到任何候选向量");
        for (int i = 0; i < total_candidates; i++) {
            if (index_tuples[i]) pfree(index_tuples[i]);
        }
        pfree(index_tuples);
        pfree(tuple_sizes);
        pfree(query_ids);
        return -1;
    }
    
    /* 将IndexTuple打包到连续内存（用于上传） */
    /* 计算总大小 */
    size_t total_size = 0;
    for (int i = 0; i < total_candidates; i++) {
        total_size += tuple_sizes[i];
    }
    
    /* 分配连续内存并打包IndexTuple */
    char *packed_tuples = (char*)palloc(total_size);
    int *tuple_offsets = (int*)palloc((total_candidates + 1) * sizeof(int));
    
    tuple_offsets[0] = 0;
    for (int i = 0; i < total_candidates; i++) {
        memcpy(packed_tuples + tuple_offsets[i], index_tuples[i], tuple_sizes[i]);
        tuple_offsets[i + 1] = tuple_offsets[i] + tuple_sizes[i];
        
        /* 释放临时内存 */
        pfree(index_tuples[i]);
    }
    pfree(index_tuples);
    
    /* 上传完整的IndexTuple数据到GPU */
    {
        int upload_result;
        upload_result = cuda_upload_probe_vectors(so->cuda_ctx,
                                                  packed_tuples,  // index_tuples
                                                  query_ids,
                                                  tuple_sizes,     // tuple_sizes
                                                  tuple_offsets,   // tuple_offsets
                                                  total_candidates,
                                                  dimensions,
                                                  0);     // fixed_tuple_size = 0 (变长)
        
        pfree(packed_tuples);
        pfree(tuple_sizes);
        pfree(tuple_offsets);
        pfree(query_ids);
        
        if (upload_result != 0) {
            elog(ERROR, "UploadIndexTuplesToGPU_Batch: GPU上传失败");
            return -1;
        }
    }
    
    elog(LOG, "UploadIndexTuplesToGPU_Batch: 成功上传 %d 个IndexTuple（包含t_tid + t_info + Vector数据）", 
         total_candidates);
    return 0;
}
#endif
