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
#include "utils/tuplestore.h"
#include "executor/tuptable.h"
#include "catalog/index.h"
#include "utils/relcache.h"
#include "fmgr.h"
#include "ivfscanbatch.h"
#include "scanbatch.h"
#include "ivfflat.h"
#include "vector.h"

#include "cuda/cuda_wrapper.h"

/* 引入新的 Pipeline 接口 */
#ifdef USE_CUDA
/* 函数声明已在 ivfscanbatch.h 中 */
#include "utils/hsearch.h"
#endif

/* ========================================================================= */
/*                              GPU 索引缓存管理                              */
/* ========================================================================= */

#ifdef USE_CUDA
/* 缓存键：使用 Index 的 OID */
typedef struct GpuIndexCacheKey {
    Oid index_oid;
} GpuIndexCacheKey;

/* 缓存条目 */
typedef struct GpuIndexCacheEntry {
    GpuIndexCacheKey key;      /* Hash Key */
    
    /* 缓存的数据 */
    void* idx_handle;          /* GPU Handle */
    ItemPointer global_tids;   /* CPU端 TID 映射表 */
    int n_total_vectors;       /* 向量总数 (用于校验) */
    int n_total_clusters;      /* 聚类总数 */
    int dimensions;            /* 维度 */
} GpuIndexCacheEntry;

/* 静态全局变量，存储当前会话的缓存 */
static HTAB *g_gpu_index_cache = NULL;

/* 初始化缓存表 */
static void
InitGpuIndexCache(void)
{
    HASHCTL info;
    
    memset(&info, 0, sizeof(info));
    info.keysize = sizeof(GpuIndexCacheKey);
    info.entrysize = sizeof(GpuIndexCacheEntry);
    info.hcxt = TopMemoryContext;  /* 使用 TopMemoryContext 确保缓存持久化 */
    
    g_gpu_index_cache = hash_create("GPU Index Cache",
                                    16, /* 初始大小 */
                                    &info,
                                    HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
}
#endif /* USE_CUDA */

/* ========================================================================= */
/*                               辅助宏与结构定义                            */
/* ========================================================================= */

#define GetScanList(ptr) pairingheap_container(IvfflatScanList, ph_node, ptr)
#define GetScanListConst(ptr) pairingheap_const_container(IvfflatScanList, ph_node, ptr)
/*
 * Compare list distances
 */
static int CompareLists(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (GetScanListConst(a)->distance > GetScanListConst(b)->distance)
		return 1;

	if (GetScanListConst(a)->distance < GetScanListConst(b)->distance)
		return -1;

	return 0;
}
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
    if (!cuda_is_available()) {
        elog(ERROR, "批量向量搜索需要GPU支持，但CUDA不可用");
    }
    
    /* 设置probe相关参数 */
    int probes = ivfflat_probes;  /* 从GUC参数获取 */
    int maxProbes = Max(ivfflat_max_probes, probes);
    if (probes > lists) probes = lists;
    if (maxProbes > lists) maxProbes = lists;
    
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
#endif
    scan->opaque = so;

    return scan;
}


bool
ivfflatbatchgettuple(IndexScanDesc scan, ScanDirection dir, Tuplestorestate *tupstore, TupleDesc tupdesc, int k)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    int nbatch;

    if (!so->batch_processing_complete) {
        ProcessBatchQueriesGPU(scan, so->batch_keys, k);
        so->batch_processing_complete = true;
    }
    
    if (!so->result_buffer) {
        elog(ERROR, "ivfflatbatchgettuple: result_buffer为空！");
        return false;
    }
    
    nbatch = so->batch_keys->nkeys;
    
    if (so->result_buffer->n_queries != nbatch) {
        elog(ERROR, "ivfflatbatchgettuple: result_buffer中的查询数量(%d)与batch_keys中的查询数量(%d)不匹配！",
             so->result_buffer->n_queries, nbatch);
        return false;
    }
    
    GetBatchResults(so->result_buffer, tupstore, tupdesc);
    
    return true;
}

void
ivfflatbatchendscan(IndexScanDesc scan)
{
    IvfflatBatchScanOpaque so;
    so = (IvfflatBatchScanOpaque) scan->opaque;
    
    if(so != NULL){
        if(so->result_buffer){
            if (so->result_buffer->query_data) pfree(so->result_buffer->query_data);
            if (so->result_buffer->query_ids) pfree(so->result_buffer->query_ids);
            if (so->result_buffer->distances) pfree(so->result_buffer->distances);
            pfree(so->result_buffer);
        }
    }
    scan->opaque = NULL;
}

BatchBuffer*
CreateBatchBuffer(int n_queries, int k, int dimensions, MemoryContext ctx)
{
    int total_results = n_queries * k;
    BatchBuffer* buffer = (BatchBuffer*)palloc0(sizeof(BatchBuffer));
    
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

void
GetBatchResults(BatchBuffer* buffer, Tuplestorestate *tupstore, TupleDesc tupdesc)
{
    int n_queries;
    int k;
    Datum values[3];
    bool tuple_nulls[3];

    if (buffer == NULL || tupstore == NULL || tupdesc == NULL) {
        elog(ERROR, "GetBatchResults: 参数为 NULL");
        return;
    }
    
    n_queries = buffer->n_queries;
    k = buffer->k;
    
    for (int query_idx = 0; query_idx < n_queries; query_idx++) {
        for (int k_idx = 0; k_idx < k; k_idx++) {
            int buffer_idx = query_idx * k + k_idx;
            bool is_null = (buffer->global_vector_indices[buffer_idx] < 0);
            
            if (is_null) {
                values[0] = (Datum) 0;
                values[1] = (Datum) 0;
                values[2] = (Datum) 0;
                tuple_nulls[0] = true;
                tuple_nulls[1] = true;
                tuple_nulls[2] = true;
            } else {
                values[0] = Int32GetDatum(buffer->query_ids[buffer_idx]);
                values[1] = Int32GetDatum(buffer->global_vector_indices[buffer_idx]);
                values[2] = Float8GetDatum(buffer->distances[buffer_idx]);
                tuple_nulls[0] = false;
                tuple_nulls[1] = false;
                tuple_nulls[2] = false;
            }
            
            tuplestore_putvalues(tupstore, tupdesc, values, tuple_nulls);
        }
    }
}

/*
 * 向ScankeyBatch添加vectorbatch
 */

/* 简化的元数据准备函数声明 */
static int PrepareMetaInfoOnly(IndexScanDesc scan,
                                int** cluster_size_out,
                                BlockNumber** cluster_pages_out,
                                float** cluster_centers_flat_out,
                                int* n_total_clusters_out, 
                                int* n_total_vectors_out);

static void ConvertBatchPipelineResults(IndexScanDesc scan, float** topk_dist, int** topk_index,
                                        int n_query, int k, BatchBuffer* result_buffer,
                                        ItemPointer global_tids, int n_total_vectors, Relation indexRelation);


/* ========================================================================= */
/*                              GPU 索引缓存函数                              */
/* ========================================================================= */

#ifdef USE_CUDA
/* 
 * 获取或加载 GPU 索引
 * 如果缓存中有，直接返回；否则执行加载流程。
 */
static GpuIndexCacheEntry*
GetOrLoadGpuIndex(IndexScanDesc scan)
{
    Oid index_oid = RelationGetRelid(scan->indexRelation);
    GpuIndexCacheEntry *entry;
    bool found;
    
    /* 1. 初始化缓存表 */
    if (g_gpu_index_cache == NULL) {
        InitGpuIndexCache();
    }
    
    /* 2. 查找缓存 */
    GpuIndexCacheKey key;
    key.index_oid = index_oid;
    
    entry = (GpuIndexCacheEntry *) hash_search(g_gpu_index_cache,
                                               &key,
                                               HASH_ENTER,
                                               &found);
    
    /* 3. 如果命中缓存，直接返回 */
    if (found && entry->idx_handle != NULL) {
        /* TODO: 这里可以添加逻辑检查索引是否过期 (例如比较文件大小或版本号) */
        /* 目前简单起见，假设 Session 期间索引不变 */
        return entry;
    }
    
    /* 3.5. 尝试从注册表获取（如果是在 Build 阶段创建的） */
    void* registered_handle = ivf_get_index_instance(index_oid);
    
    /* ================== 缓存未命中，执行加载流程 ================== */
    
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    int dimensions = so->dimensions;
    
    /* 临时变量声明 */
    int* predicted_cluster_sizes = NULL;
    BlockNumber* cluster_start_pages = NULL;
    float* cluster_centers_flat = NULL;
    int n_total_clusters = 0;
    int n_total_vectors = 0;
    
    /* A. 准备元数据 */
    if (PrepareMetaInfoOnly(scan, &predicted_cluster_sizes, &cluster_start_pages, 
                            &cluster_centers_flat, &n_total_clusters, &n_total_vectors) != 0) {
        hash_search(g_gpu_index_cache, &key, HASH_REMOVE, NULL); /* 移除占位符 */
        elog(ERROR, "GPU Index Loading: 元数据准备失败");
        return NULL;
    }
    
    if (n_total_clusters <= 0) {
        hash_search(g_gpu_index_cache, &key, HASH_REMOVE, NULL);
        elog(WARNING, "GPU Index Loading: 索引为空");
        return NULL;
    }
    
    /* B. 初始化 GPU Handle */
    void* idx_handle = NULL;
    bool use_registered_handle = false;
    
    /* 如果从注册表获取到了句柄，直接使用 */
    if (registered_handle != NULL) {
        idx_handle = registered_handle;
        use_registered_handle = true;
        elog(INFO, "GPU Index Loading: 使用注册表中的句柄 (OID %u)", index_oid);
    } else {
        /* 否则创建新的句柄 */
        /* 注意：这里的内存 context 应该是 TopMemoryContext，
           以保证 global_tids 在函数返回后不被释放 */
        MemoryContext cacheCtx = TopMemoryContext; 
        MemoryContext oldCtx = MemoryContextSwitchTo(cacheCtx);
        
        idx_handle = ivf_create_index_context_wrapper();
        if (idx_handle == NULL) {
            MemoryContextSwitchTo(oldCtx);
            hash_search(g_gpu_index_cache, &key, HASH_REMOVE, NULL);
            elog(ERROR, "GPU Index Loading: 无法创建 GPU Handle");
            return NULL;
        }
        
        int max_vectors_capacity = (int)(n_total_vectors * 1.1) + 1000;
        ivf_init_streaming_upload_wrapper(idx_handle, n_total_clusters, max_vectors_capacity, dimensions);
        
        MemoryContextSwitchTo(oldCtx); /* 切回原来的 Context */
    }
    
    /* 分配 TID 映射表（无论是否从注册表获取，都需要 TID 映射） */
    MemoryContext cacheCtx = TopMemoryContext; 
    MemoryContext oldCtx = MemoryContextSwitchTo(cacheCtx);
    int max_vectors_capacity = (int)(n_total_vectors * 1.1) + 1000;
    ItemPointer global_tids = (ItemPointer)palloc0(max_vectors_capacity * sizeof(ItemPointerData));
    MemoryContextSwitchTo(oldCtx);
    
    /* C. 扫描数据并流式上传 (包含内存泄漏修复) */
    TupleDesc tupdesc = RelationGetDescr(scan->indexRelation);
    int current_global_offset = 0;
    size_t vec_size = dimensions * sizeof(float);
    
    /* === 关键修复：创建临时内存上下文用于解压 Vector === */
    MemoryContext loopTmpCtx = AllocSetContextCreate(CurrentMemoryContext,
                                                     "GPU Load Loop Context",
                                                     ALLOCSET_DEFAULT_SIZES);
    
    /* 临时 Buffer 分配 (在 oldCtx 中分配，确保在循环外可用) */
    MemoryContextSwitchTo(oldCtx);
    int tmp_buf_cap = 1024;
    float* tmp_cluster_buffer = (float*)palloc(tmp_buf_cap * vec_size);
    MemoryContextSwitchTo(loopTmpCtx);
    
    for (int i = 0; i < n_total_clusters; i++) {
        BlockNumber searchPage = cluster_start_pages[i];
        int cluster_vec_count = 0;
        int cluster_start_offset = current_global_offset;
        
        while (BlockNumberIsValid(searchPage)) {
            /* 在处理新 Buffer 前重置内存上下文 */
            MemoryContextReset(loopTmpCtx);
            MemoryContextSwitchTo(loopTmpCtx);
            
            Buffer buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, 
                                          searchPage, RBM_NORMAL, NULL);
            LockBuffer(buf, BUFFER_LOCK_SHARE);
            Page page = BufferGetPage(buf);
            OffsetNumber maxoffno = PageGetMaxOffsetNumber(page);
            
            for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno)) {
                ItemId itemId = PageGetItemId(page, offno);
                if (!ItemIdIsUsed(itemId)) continue;
                
                IndexTuple itup = (IndexTuple) PageGetItem(page, itemId);
                if (!itup) continue;
                
                bool isnull = false;
                /* index_getattr 在 loopTmpCtx 中分配内存，不会泄露 */
                Datum vector_datum = index_getattr(itup, 1, tupdesc, &isnull);
                
                if (!isnull) {
                    if (current_global_offset >= max_vectors_capacity) {
                        elog(WARNING, "GPU Index Loading: 向量数量超过预分配上限，截断处理");
                        break;
                    }
                    
                    /* 记录 TID (内存是在 TopContext 分配的，安全) */
                    MemoryContextSwitchTo(cacheCtx);
                    global_tids[current_global_offset] = itup->t_tid;
                    MemoryContextSwitchTo(loopTmpCtx);
                    
                    /* 如果使用注册表句柄，数据已上传，只需要记录 TID */
                    if (!use_registered_handle) {
                        Vector *vec = DatumGetVector(vector_datum);
                        
                        /* 动态扩容 (需要切回 oldCtx 进行 repalloc) */
                        if (cluster_vec_count >= tmp_buf_cap) {
                            MemoryContextSwitchTo(oldCtx);
                            tmp_buf_cap *= 2;
                            tmp_cluster_buffer = (float*)repalloc(tmp_cluster_buffer, tmp_buf_cap * vec_size);
                            MemoryContextSwitchTo(loopTmpCtx);
                        }
                        
                        memcpy(tmp_cluster_buffer + (cluster_vec_count * dimensions), 
                               vec->x, vec_size);
                        cluster_vec_count++;
                    }
                    
                    current_global_offset++;
                }
            }
            
            searchPage = IvfflatPageGetOpaque(page)->nextblkno;
            UnlockReleaseBuffer(buf);
        }
        
        /* 上传数据 (数据在 tmp_cluster_buffer 中) */
        /* 如果使用注册表的句柄，数据已经在 Build 阶段上传，这里只需要扫描 TID */
        if (!use_registered_handle) {
            ivf_append_cluster_data_wrapper(
                idx_handle, i, tmp_cluster_buffer, cluster_vec_count, cluster_start_offset
            );
        }
    }
    
    /* 清理加载过程的临时资源 */
    MemoryContextDelete(loopTmpCtx);
    MemoryContextSwitchTo(oldCtx);
    
    pfree(tmp_cluster_buffer);
    pfree(predicted_cluster_sizes);
    pfree(cluster_start_pages);
    
    /* D. 完成上传（如果使用注册表句柄，数据已上传，跳过） */
    if (!use_registered_handle) {
        ivf_finalize_streaming_upload_wrapper(idx_handle, cluster_centers_flat, current_global_offset);
    }
    pfree(cluster_centers_flat);
    
    /* E. 填充缓存条目 */
    entry->idx_handle = idx_handle;
    entry->global_tids = global_tids;
    entry->n_total_vectors = current_global_offset; /* 使用实际值 */
    entry->n_total_clusters = n_total_clusters;
    entry->dimensions = dimensions;
    
    return entry;
}
#endif /* USE_CUDA */

/* ========================================================================= */
/*                              核心处理逻辑                                  */
/* ========================================================================= */

/*
 * ProcessBatchQueriesGPU
 * 
 * 使用缓存机制优化后的版本：
 * 1. 获取或加载 GPU 索引 (GetOrLoadGpuIndex) - 带缓存复用
 * 2. 准备 Batch Query 并执行 (ivf_pipeline_stage*)
 * 3. 获取结果并转换
 */
void
ProcessBatchQueriesGPU(IndexScanDesc scan, ScanKeyBatch batch_keys, int k)
{
#ifdef USE_CUDA
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    int nbatch = batch_keys->nkeys;
    int dimensions = so->dimensions;
    so->batch_processing_complete = false;
    
    /* 重新创建 result_buffer */
    if (so->result_buffer != NULL) {
        if (so->result_buffer->n_queries != nbatch || so->result_buffer->k != k) {
            so->result_buffer = NULL;
        }
    }
    
    /* 1. 获取 GPU 索引 (复用或加载) */
    GpuIndexCacheEntry *index_entry = GetOrLoadGpuIndex(scan);
    if (!index_entry || index_entry->idx_handle == NULL) {
        /* 错误已在内部记录 */
        return;
    }
    
    /* 2. 准备 Query Data */
    float* query_batch_flat = (float*)palloc(nbatch * dimensions * sizeof(float));
    for (int i = 0; i < nbatch; i++) {
        Datum vec_datum = ScanKeyBatchGetVector(batch_keys, i);
        Vector *vec = DatumGetVector(vec_datum);
        memcpy(query_batch_flat + i * dimensions, vec->x, dimensions * sizeof(float));
    }
    
    /* 3. 创建临时 Batch Handle (每次查询独立) */
    void* batch_handle = ivf_create_batch_context_wrapper(
        nbatch, dimensions, so->maxProbes, k, 
        index_entry->n_total_clusters
    );
    
    if (!batch_handle) {
        pfree(query_batch_flat);
        elog(ERROR, "ProcessBatchQueriesGPU: 无法创建 GPU Batch Handle");
        return;
    }
    
    /* 4. 执行 Pipeline */
    ivf_pipeline_stage1_prepare_wrapper(batch_handle, query_batch_flat, nbatch);
    
    ivf_pipeline_stage2_compute_wrapper(batch_handle, index_entry->idx_handle, 
                                        nbatch, so->probes, k);
    
    /* 5. 获取结果 */
    float* topk_dist_flat = (float*)palloc(nbatch * k * sizeof(float));
    int* topk_index_flat = (int*)palloc(nbatch * k * sizeof(int));
    
    ivf_pipeline_get_results_wrapper(batch_handle, topk_dist_flat, topk_index_flat, nbatch, k);
    
    /* 指针数组转换 */
    float** topk_dist = (float**)palloc(nbatch * sizeof(float*));
    int** topk_index = (int**)palloc(nbatch * sizeof(int*));
    for (int i = 0; i < nbatch; i++) {
        topk_dist[i] = topk_dist_flat + i * k;
        topk_index[i] = topk_index_flat + i * k;
    }
    
    /* 6. 结果转换 (ID Mapping) */
    so->result_buffer = CreateBatchBuffer(nbatch, k, dimensions, CurrentMemoryContext);
    
    ConvertBatchPipelineResults(scan, topk_dist, topk_index, nbatch, k,
                                so->result_buffer, 
                                index_entry->global_tids, /* 使用缓存的 TID 表 */
                                index_entry->n_total_vectors, 
                                scan->indexRelation);
    
    /* 7. 资源清理 (只清理 Batch 相关的，保留 Index 相关的) */
    ivf_destroy_batch_context_wrapper(batch_handle);
    /* 注意：千万不要调用 ivf_destroy_index_context_wrapper(index_entry->idx_handle) */
    
    pfree(query_batch_flat);
    pfree(topk_dist_flat);
    pfree(topk_index_flat);
    pfree(topk_dist);
    pfree(topk_index);
#else
    elog(ERROR, "ProcessBatchQueriesGPU: CUDA support not compiled");
#endif /* USE_CUDA */
}


/* ========================================================================= */
/*                              元数据准备函数                                */
/* ========================================================================= */

/* 简化的元数据准备：只统计数量，不读向量数据 */
static int
PrepareMetaInfoOnly(IndexScanDesc scan,
                    int** cluster_size_out,
                    BlockNumber** cluster_pages_out,
                    float** cluster_centers_flat_out,
                    int* n_total_clusters_out, 
                    int* n_total_vectors_out)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    TupleDesc tupdesc = RelationGetDescr(scan->indexRelation);
    int totalLists = 0;
    IvfflatGetMetaPageInfo(scan->indexRelation, &totalLists, NULL);
    
    if (totalLists <= 0) return -1;
    
    int dimensions = so->dimensions;
    
    /* 分配输出数组 */
    int* cluster_sizes = (int*)palloc0(totalLists * sizeof(int));
    BlockNumber* cluster_start_pages = (BlockNumber*)palloc(totalLists * sizeof(BlockNumber));
    
    /* 1. 遍历 Meta Pages 获取 Cluster 起始页 */
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
            if (center_idx >= totalLists) break;
            IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
            cluster_start_pages[center_idx] = list->startPage;
            center_idx++;
        }
        nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;
        UnlockReleaseBuffer(cbuf);
    }
    
    /* 2. 统计每个 cluster 的向量数量 */
    int total_vectors = 0;
    for (int i = 0; i < totalLists; i++) {
        BlockNumber searchPage = cluster_start_pages[i];
        while (BlockNumberIsValid(searchPage)) {
            Buffer buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, 
                                          searchPage, RBM_NORMAL, NULL);
            LockBuffer(buf, BUFFER_LOCK_SHARE);
            Page page = BufferGetPage(buf);
            OffsetNumber maxoffno = PageGetMaxOffsetNumber(page);
            
            for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno)) {
                ItemId itemId = PageGetItemId(page, offno);
                /* 关键修正: 严谨检查 Item 是否可用 */
                if (!ItemIdIsUsed(itemId)) continue;
                IndexTuple itup = (IndexTuple) PageGetItem(page, itemId);
                if (!itup) continue;
                
                /* 这里只检查 Datum 是否存在，不解压 Vector，速度较快 */
                bool isnull = false;
                index_getattr(itup, 1, tupdesc, &isnull);
                if (!isnull) {
                    cluster_sizes[i]++;
                    total_vectors++;
                }
            }
            searchPage = IvfflatPageGetOpaque(page)->nextblkno;
            UnlockReleaseBuffer(buf);
        }
    }
    
    /* 3. 获取 Cluster Centers (扁平化) */
    float* cluster_centers_flat = (float*)palloc(totalLists * dimensions * sizeof(float));
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
            if (center_idx >= totalLists) break;
            IndexTuple itup = (IndexTuple) PageGetItem(cpage, PageGetItemId(cpage, offno));
            bool isnull = false;
            Datum center_datum = index_getattr(itup, 1, tupdesc, &isnull);
            
            if (!isnull) {
                Vector *center_vec = DatumGetVector(center_datum);
                memcpy(cluster_centers_flat + center_idx * dimensions, 
                       center_vec->x, dimensions * sizeof(float));
            }
            center_idx++;
        }
        nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;
        UnlockReleaseBuffer(cbuf);
    }
    
    *cluster_size_out = cluster_sizes;
    *cluster_pages_out = cluster_start_pages;
    *cluster_centers_flat_out = cluster_centers_flat;
    *n_total_clusters_out = totalLists;
    *n_total_vectors_out = total_vectors;
    
    return 0;
}

/* ========================================================================= */
/*                              结果转换函数 (延迟回表)                        */
/* ========================================================================= */

static void
ConvertBatchPipelineResults(IndexScanDesc scan, float** topk_dist, int** topk_index,
                            int n_query, int k, BatchBuffer* result_buffer,
                            ItemPointer global_tids, int n_total_vectors, Relation indexRelation)
{
    /* 打开堆表（只在这里打开一次） */
    Oid heapRelid = IndexGetRelation(RelationGetRelid(indexRelation), false);
    Relation heapRelation = table_open(heapRelid, AccessShareLock);
    TupleDesc heapDesc = RelationGetDescr(heapRelation);
    Snapshot snapshot = GetActiveSnapshot();
    
    /* 查找 ID 列 */
    int id_attnum = -1;
    for (int i = 1; i <= heapDesc->natts; i++) {
        Form_pg_attribute attr = TupleDescAttr(heapDesc, i - 1);
        if (strcmp(NameStr(attr->attname), "id") == 0) {
            id_attnum = i;
            break;
        }
    }
    
    /* 初始化 buffer */
    int buffer_size = result_buffer->n_queries * result_buffer->k;
    memset(result_buffer->query_ids, -1, buffer_size * sizeof(int));
    memset(result_buffer->global_vector_indices, -1, buffer_size * sizeof(int));
    memset(result_buffer->distances, 0, buffer_size * sizeof(float));
    
    /* 填充结果 */
    for (int query_idx = 0; query_idx < n_query; query_idx++) {
        for (int i = 0; i < k; i++) {
            int buffer_idx = query_idx * k + i;
            int vec_idx = topk_index[query_idx][i];
            
            /* 增加对 vec_idx 的有效性检查 */
            if (vec_idx < 0) {
                result_buffer->query_ids[buffer_idx] = query_idx;
                ItemPointerSetInvalid(&result_buffer->vector_ids[buffer_idx]);
                result_buffer->distances[buffer_idx] = INFINITY;
                continue;
            }
            
            /* 边界检查：确保 vec_idx 在有效范围内 */
            if (vec_idx >= n_total_vectors) {
                elog(WARNING, "ConvertBatchPipelineResults: vec_idx %d >= n_total_vectors %d, skipping", 
                     vec_idx, n_total_vectors);
                result_buffer->query_ids[buffer_idx] = query_idx;
                ItemPointerSetInvalid(&result_buffer->vector_ids[buffer_idx]);
                result_buffer->distances[buffer_idx] = INFINITY;
                continue;
            }
            
            /* 1. 从 CPU 数组获取 TID */
            /* 这里的 global_tids 现在和 GPU 的 d_cluster_vectors 是严格对应的 */
            ItemPointer tid = &global_tids[vec_idx];
            
            /* 调试日志：如果发现结果还是不对，可以打开这个查看 TID 是否合理 */
            /* elog(LOG, "Query %d Top %d: FlatIdx %d -> TID (%u, %u)", 
                 query_idx, i, vec_idx, 
                 ItemPointerGetBlockNumber(tid), ItemPointerGetOffsetNumber(tid)); */
            result_buffer->query_ids[buffer_idx] = query_idx;
            result_buffer->vector_ids[buffer_idx] = *tid;
            result_buffer->distances[buffer_idx] = topk_dist[query_idx][i];
            
            /* 2. 延迟回表：只对 Top-K 结果执行 heap_fetch */
            int32_t row_id = -1;
            if (id_attnum > 0) {
                Buffer buffer;
                HeapTupleData tuple;
                tuple.t_self = *tid;
                if (heap_fetch(heapRelation, snapshot, &tuple, &buffer, false)) {
                    bool isnull;
                    Datum d = heap_getattr(&tuple, id_attnum, heapDesc, &isnull);
                    if (!isnull) row_id = DatumGetInt32(d);
                    ReleaseBuffer(buffer);
                }
            }
            result_buffer->global_vector_indices[buffer_idx] = row_id;
        }
    }
    
    table_close(heapRelation, AccessShareLock);
}
