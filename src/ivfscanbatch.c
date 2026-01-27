#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/relscan.h"
#include "access/tableam.h"
#include "access/heapam.h"
#include "access/htup_details.h"  /* 用于 MaxHeapTuplesPerPage */
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
ivfflatbatchgettuple(IndexScanDesc scan, ScanDirection dir, Tuplestorestate *tupstore, TupleDesc tupdesc, int k, int distance_mode)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    int nbatch;

    if (!so->batch_processing_complete) {
        ProcessBatchQueriesGPU(scan, so->batch_keys, k, distance_mode);
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
/* ========================================================================= */
/*                              GPU 索引缓存函数 (核心加载逻辑)                */
/* ========================================================================= */

#ifdef USE_CUDA
static GpuIndexCacheEntry*
GetOrLoadGpuIndex(IndexScanDesc scan)
{
    Oid index_oid = RelationGetRelid(scan->indexRelation);
    GpuIndexCacheEntry *entry;
    bool found;
    
    /* 1. 初始化并查找缓存 */
    if (g_gpu_index_cache == NULL) InitGpuIndexCache();
    
    GpuIndexCacheKey key;
    key.index_oid = index_oid;
    
    entry = (GpuIndexCacheEntry *) hash_search(g_gpu_index_cache, &key, HASH_ENTER, &found);
    
    /* 命中缓存直接返回 */
    if (found && entry->idx_handle != NULL) return entry;
    
    /* 尝试从 Build 阶段的注册表获取 */
    void* registered_handle = ivf_get_index_instance(index_oid);
    
    /* ================== 开始加载流程 ================== */
    
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    int dimensions = so->dimensions;
    int* predicted_cluster_sizes = NULL; 
    BlockNumber* cluster_start_pages = NULL;
    float* cluster_centers_flat = NULL;
    int n_total_clusters = 0;
    int n_estimated_vectors = 0;
    
    /* 1. 快速准备元数据 (仅扫描 Meta Pages) */
    if (PrepareMetaInfoOnly(scan, &predicted_cluster_sizes, &cluster_start_pages, 
                            &cluster_centers_flat, &n_total_clusters, &n_estimated_vectors) != 0) {
        hash_search(g_gpu_index_cache, &key, HASH_REMOVE, NULL);
        elog(ERROR, "GPU Index Loading: 元数据准备失败");
        return NULL;
    }
    
    /* 2. 基于文件大小估算总向量数 (消除双重扫描) */
    BlockNumber total_blocks = RelationGetNumberOfBlocks(scan->indexRelation);
    /* 估算公式：总块数 * (块大小 / (向量大小 + 索引头大小)) * 填充率 */
    size_t est_tuple_size = dimensions * sizeof(float) + sizeof(IndexTupleData) + sizeof(ItemIdData);
    double fill_factor = 1.0; /* 假设填充率 */
    n_estimated_vectors = (int)((total_blocks * BLCKSZ * fill_factor) / est_tuple_size);
    n_estimated_vectors = Max(n_estimated_vectors, 10000); /* 最小保底 */
    
    elog(LOG, "GPU Index Loading: 估算向量总数 %d (Blocks: %u)", n_estimated_vectors, total_blocks);

    /* 3. 初始化 GPU Context */
    void* idx_handle = NULL;
    bool use_registered_handle = false;
    
    if (registered_handle != NULL) {
        /* 安全检查：验证注册的 handle 是否已初始化 */
        /* 注意：由于无法直接访问 C++ 结构体，我们通过尝试访问来验证 */
        /* 如果 handle 未初始化，在后续调用时会抛出异常 */
        idx_handle = registered_handle;
        use_registered_handle = true;
        
        /* 验证维度一致性：需要从元数据中获取维度信息 */
        /* 注意：这里假设 Build 阶段和 Scan 阶段的维度应该一致 */
        elog(INFO, "GPU Index Loading: 使用 Build 阶段注册的 handle (OID %u), n_total_clusters=%d, dimensions=%d", 
             index_oid, n_total_clusters, dimensions);
        
        /* 等待 Build 阶段的异步操作完成（如果有） */
        /* 注意：ivf_finalize_streaming_upload 内部已经同步，但为了安全起见，这里再次同步 */
        cuda_device_synchronize();
        
        /* 检查 CUDA 错误状态 */
        if (cuda_check_last_error()) {
            elog(WARNING, "GPU Index Loading: CUDA 错误状态 (可能来自 Build 阶段): %s", 
                 cuda_get_last_error_string());
            /* 清除错误状态，继续执行 */
            cuda_check_last_error();
        }
    } else {
        /* 使用估算值分配显存 */
        idx_handle = ivf_create_index_context_wrapper();
        if (!idx_handle) {
            hash_search(g_gpu_index_cache, &key, HASH_REMOVE, NULL);
            elog(ERROR, "GPU Index Loading: 无法创建 GPU Handle (可能是显存不足)");
            return NULL;
        }
        /* 此处 n_estimated_vectors 决定了显存 malloc 大小，宁大勿小 */
        ivf_init_streaming_upload_wrapper(idx_handle, n_total_clusters, n_estimated_vectors, dimensions);
    }
    
    /* 4. 分配 CPU 端全局 TID 数组 (使用 TopMemoryContext 持久化) */
    MemoryContext cacheCtx = TopMemoryContext;
    MemoryContext oldCtx = MemoryContextSwitchTo(cacheCtx);
    
    /* 关键修复：确保 ItemPointerData 对齐正确 */
    /* ItemPointerData 是 6 字节，但可能被编译器对齐到 8 字节 */
    size_t tid_array_size = n_estimated_vectors * sizeof(ItemPointerData);
    ItemPointer global_tids = (ItemPointer)palloc0(tid_array_size);
    
    /* 验证对齐：确保数组元素之间的偏移量正确 */
    if (n_estimated_vectors > 1) {
        size_t actual_offset = (char*)&global_tids[1] - (char*)&global_tids[0];
        if (actual_offset != sizeof(ItemPointerData)) {
            elog(WARNING, "GetOrLoadGpuIndex: ItemPointerData 对齐异常 - sizeof=%zu, actual_offset=%zu", 
                 sizeof(ItemPointerData), actual_offset);
        }
    }
    
    MemoryContextSwitchTo(oldCtx);
    
    /* =======================================================
     * 优化 2: 使用 Pinned Memory + 异常安全处理
     * ======================================================= */
    
    /* 分配 Pinned Memory */
    int tmp_buf_cap = 4096; /* 每次缓冲 4K 个向量 */
    size_t pinned_mem_size = tmp_buf_cap * dimensions * sizeof(float);
    float* pinned_cluster_buffer = (float*)cuda_alloc_pinned(pinned_mem_size);
    
    if (!pinned_cluster_buffer) {
        /* 回滚：清理资源 */
        if (!use_registered_handle) ivf_destroy_index_context_wrapper(idx_handle);
        pfree(global_tids);
        hash_search(g_gpu_index_cache, &key, HASH_REMOVE, NULL);
        elog(ERROR, "GPU Index Loading: 无法分配 Pinned Memory");
        return NULL;
    }

    int current_global_offset = 0;
    TupleDesc tupdesc = RelationGetDescr(scan->indexRelation);
    
    /* 临时内存上下文，用于循环内解压 Vector */
    MemoryContext loopTmpCtx = AllocSetContextCreate(CurrentMemoryContext, "GPU Load Loop", ALLOCSET_DEFAULT_SIZES);
    
    /* 批量读取策略，防止刷爆 Shared Buffers */
    BufferAccessStrategy bas = GetAccessStrategy(BAS_BULKREAD);

    /* 使用 PG_TRY 保护 Pinned Memory 的释放 */
    PG_TRY();
    {
        for (int i = 0; i < n_total_clusters; i++) {
            BlockNumber searchPage = cluster_start_pages[i];
            int cluster_vec_count = 0;
            int cluster_start_offset = current_global_offset;
            
            while (BlockNumberIsValid(searchPage)) {
                MemoryContextReset(loopTmpCtx);
                MemoryContextSwitchTo(loopTmpCtx);
                
                Buffer buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, searchPage, RBM_NORMAL, bas);
                LockBuffer(buf, BUFFER_LOCK_SHARE);
                Page page = BufferGetPage(buf);
                OffsetNumber maxoffno = PageGetMaxOffsetNumber(page);
                
                for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno)) {
                    ItemId itemId = PageGetItemId(page, offno);
                    if (!ItemIdIsUsed(itemId)) continue;
                    
                    IndexTuple itup = (IndexTuple) PageGetItem(page, itemId);
                    
                    /* 注意：TID 记录必须在 cacheCtx 下进行吗？不需要，global_tids 数组本身已分配 */
                    /* 这里直接赋值即可 */
                    
                    bool isnull = false;
                    Datum vector_datum = index_getattr(itup, 1, tupdesc, &isnull);
                    
                    if (!isnull) {
                        /* 安全检查：防止估算偏小导致数组越界 */
                        if (current_global_offset >= n_estimated_vectors) {
                            /* 生产环境可能需要 realloc，这里简化为截断并警告 */
                            static bool warned = false;
                            if (!warned) {
                                elog(WARNING, "GPU Index Loading: 实际向量超过估算值，部分数据被截断");
                                warned = true;
                            }
                            continue;
                        }
                        
                        /* 如果不是复用 Build 阶段的 Handle，则需要拷贝向量数据 */
                        if (!use_registered_handle) {
                            Vector *vec = DatumGetVector(vector_datum);
                            
                            /* 缓冲区满，触发上传 */
                            if (cluster_vec_count >= tmp_buf_cap) {
                                ivf_append_cluster_data_wrapper(
                                    idx_handle, i, pinned_cluster_buffer, cluster_vec_count, cluster_start_offset
                                );
                                cluster_start_offset += cluster_vec_count;
                                cluster_vec_count = 0;
                            }
                            
                            /* 拷贝到 Pinned Memory */
                            memcpy(pinned_cluster_buffer + (cluster_vec_count * dimensions), 
                                   vec->x, dimensions * sizeof(float));
                        }
                        
                        /* 记录 TID */
                        global_tids[current_global_offset] = itup->t_tid;
                        
                        current_global_offset++;
                        if (!use_registered_handle) cluster_vec_count++;
                    }
                }
                
                searchPage = IvfflatPageGetOpaque(page)->nextblkno;
                UnlockReleaseBuffer(buf);
            }
            
            /* 上传当前 Cluster 尾部数据 */
            if (!use_registered_handle && cluster_vec_count > 0) {
                ivf_append_cluster_data_wrapper(
                    idx_handle, i, pinned_cluster_buffer, cluster_vec_count, cluster_start_offset
                );
            }
        }
    }
    PG_CATCH();
    {
        /* 异常处理：释放 Pinned Memory，避免内存泄漏 */
        cuda_free_pinned(pinned_cluster_buffer);
        if (bas) FreeAccessStrategy(bas);
        MemoryContextDelete(loopTmpCtx);
        PG_RE_THROW();
    }
    PG_END_TRY();

    /* 正常释放 */
    cuda_free_pinned(pinned_cluster_buffer);
    if (bas) FreeAccessStrategy(bas);
    MemoryContextDelete(loopTmpCtx);
    
    /* 5. 完成上传 (计算 Norms 等) */
    if (!use_registered_handle) {
        /* 注意：传入实际读取到的 current_global_offset */
        ivf_finalize_streaming_upload_wrapper(idx_handle, cluster_centers_flat, current_global_offset);
    }
    pfree(cluster_centers_flat);
    pfree(cluster_start_pages); /* predicted_cluster_sizes 未使用 */

    /* 6. 更新缓存条目 */
    entry->idx_handle = idx_handle;
    entry->global_tids = global_tids;
    /* 重要：n_total_vectors 不能超过 global_tids 数组的实际大小 */
    /* 如果发生截断，current_global_offset 可能等于 n_estimated_vectors */
    entry->n_total_vectors = Min(current_global_offset, n_estimated_vectors);
    entry->n_total_clusters = n_total_clusters;
    entry->dimensions = dimensions;
    
    if (current_global_offset > n_estimated_vectors) {
        elog(WARNING, "GPU Index Loading: 实际向量数 %d 超过估算值 %d，已截断", 
             current_global_offset, n_estimated_vectors);
    }
    elog(INFO, "GPU Index Loading: 完成，加载向量 %d 个", entry->n_total_vectors);
    
    return entry;
}
#endif /* USE_CUDA */
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
ProcessBatchQueriesGPU(IndexScanDesc scan, ScanKeyBatch batch_keys, int k, int distance_mode)
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
    
    /* 安全检查：确保 global_tids 已初始化 */
    if (!index_entry->global_tids) {
        elog(ERROR, "ProcessBatchQueriesGPU: index_entry->global_tids is NULL");
        return;
    }
    
    /* 安全检查：验证索引条目字段 */
    if (index_entry->n_total_clusters <= 0) {
        elog(ERROR, "ProcessBatchQueriesGPU: index_entry->n_total_clusters (%d) <= 0", 
             index_entry->n_total_clusters);
        return;
    }
    if (index_entry->dimensions <= 0) {
        elog(ERROR, "ProcessBatchQueriesGPU: index_entry->dimensions (%d) <= 0", 
             index_entry->dimensions);
        return;
    }
    if (dimensions != index_entry->dimensions) {
        elog(ERROR, "ProcessBatchQueriesGPU: 维度不匹配 - scan->dimensions=%d, index_entry->dimensions=%d",
             dimensions, index_entry->dimensions);
        return;
    }
    if (so->probes > index_entry->n_total_clusters) {
        elog(ERROR, "ProcessBatchQueriesGPU: probes (%d) > n_total_clusters (%d)",
             so->probes, index_entry->n_total_clusters);
        return;
    }
    
    elog(LOG, "ProcessBatchQueriesGPU: 使用索引 - n_total_clusters=%d, n_total_vectors=%d, dimensions=%d, probes=%d, k=%d",
         index_entry->n_total_clusters, index_entry->n_total_vectors, index_entry->dimensions, so->probes, k);
    
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
    /* 安全检查：在调用前验证 handle */
    if (!batch_handle) {
        elog(ERROR, "ProcessBatchQueriesGPU: batch_handle is NULL before stage1");
        pfree(query_batch_flat);
        return;
    }
    if (!index_entry->idx_handle) {
        elog(ERROR, "ProcessBatchQueriesGPU: index_entry->idx_handle is NULL before stage2");
        pfree(query_batch_flat);
        ivf_destroy_batch_context_wrapper(batch_handle);
        return;
    }
    
    elog(LOG, "ProcessBatchQueriesGPU: 调用 stage1_prepare - nbatch=%d, dimensions=%d", nbatch, dimensions);
    ivf_pipeline_stage1_prepare_wrapper(batch_handle, query_batch_flat, nbatch);
    
    /* 检查 CUDA 错误 */
    if (cuda_check_last_error()) {
        elog(ERROR, "ProcessBatchQueriesGPU: CUDA 错误在 stage1 之后: %s", cuda_get_last_error_string());
        pfree(query_batch_flat);
        ivf_destroy_batch_context_wrapper(batch_handle);
        return;
    }
    
    elog(LOG, "ProcessBatchQueriesGPU: 调用 stage2_compute - nbatch=%d, n_probes=%d, k=%d, distance_mode=%d",
         nbatch, so->probes, k, distance_mode);
    ivf_pipeline_stage2_compute_wrapper(batch_handle, index_entry->idx_handle, 
                                        nbatch, so->probes, k, distance_mode);
    
    /* 检查 CUDA 错误 */
    if (cuda_check_last_error()) {
        elog(ERROR, "ProcessBatchQueriesGPU: CUDA 错误在 stage2 之后: %s", cuda_get_last_error_string());
        pfree(query_batch_flat);
        ivf_destroy_batch_context_wrapper(batch_handle);
        return;
    }
    
    /* 5. 获取结果 */
    float* topk_dist_flat = (float*)palloc(nbatch * k * sizeof(float));
    int* topk_index_flat = (int*)palloc(nbatch * k * sizeof(int));
    
    ivf_pipeline_get_results_wrapper(batch_handle, topk_dist_flat, topk_index_flat, nbatch, k);
    
    /* 关键修复：强制同步，确保 GPU 完全写完 CPU 内存 */
    /* 虽然 wrapper 内部已经有 cudaStreamSynchronize，但为了安全起见，这里再次同步 */
    cuda_device_synchronize();
    
    elog(LOG, "[1] get_results返回");
    
    /* 检查 CUDA 错误 */
    if (cuda_check_last_error()) {
        elog(ERROR, "ProcessBatchQueriesGPU: CUDA 错误在 get_results 之后: %s", cuda_get_last_error_string());
        pfree(query_batch_flat);
        pfree(topk_dist_flat);
        pfree(topk_index_flat);
        ivf_destroy_batch_context_wrapper(batch_handle);
        return;
    }
    
    /* 验证 GPU 返回的数据合理性 */
    for (int i = 0; i < nbatch * k && i < 100; i++) {
        if (topk_index_flat[i] < -1 || topk_index_flat[i] >= index_entry->n_total_vectors) {
            elog(WARNING, "ProcessBatchQueriesGPU: GPU返回异常索引[%d]=%d (范围: [0, %d))", 
                 i, topk_index_flat[i], index_entry->n_total_vectors);
        }
    }
    
    /* 安全检查：确保结果数组不为 NULL */
    if (!topk_dist_flat || !topk_index_flat) {
        elog(ERROR, "ProcessBatchQueriesGPU: GPU results are NULL");
        pfree(query_batch_flat);
        ivf_destroy_batch_context_wrapper(batch_handle);
        return;
    }
    
    /* 指针数组转换 */
    float** topk_dist = (float**)palloc(nbatch * sizeof(float*));
    int** topk_index = (int**)palloc(nbatch * sizeof(int*));
    if (!topk_dist || !topk_index) {
        elog(ERROR, "ProcessBatchQueriesGPU: Failed to allocate pointer arrays");
        pfree(query_batch_flat);
        pfree(topk_dist_flat);
        pfree(topk_index_flat);
        ivf_destroy_batch_context_wrapper(batch_handle);
        return;
    }
    for (int i = 0; i < nbatch; i++) {
        topk_dist[i] = topk_dist_flat + i * k;
        topk_index[i] = topk_index_flat + i * k;
    }
    elog(LOG, "[2] 指针数组转换完成");
    
    /* 6. 结果转换 (ID Mapping) */
    so->result_buffer = CreateBatchBuffer(nbatch, k, dimensions, CurrentMemoryContext);
    if (!so->result_buffer) {
        elog(ERROR, "ProcessBatchQueriesGPU: Failed to create result buffer");
        pfree(query_batch_flat);
        pfree(topk_dist_flat);
        pfree(topk_index_flat);
        pfree(topk_dist);
        pfree(topk_index);
        ivf_destroy_batch_context_wrapper(batch_handle);
        return;
    }
    elog(LOG, "[3] CreateBatchBuffer完成");
    
    /* 安全检查：确保 global_tids 有效 */
    if (!index_entry->global_tids) {
        elog(ERROR, "ProcessBatchQueriesGPU: index_entry->global_tids is NULL before ConvertBatchPipelineResults");
        pfree(query_batch_flat);
        pfree(topk_dist_flat);
        pfree(topk_index_flat);
        pfree(topk_dist);
        pfree(topk_index);
        ivf_destroy_batch_context_wrapper(batch_handle);
        return;
    }
    
    elog(LOG, "[4] 准备调用ConvertBatchPipelineResults");
    ConvertBatchPipelineResults(scan, topk_dist, topk_index, nbatch, k,
                                so->result_buffer, 
                                index_entry->global_tids, /* 使用缓存的 TID 表 */
                                index_entry->n_total_vectors, 
                                scan->indexRelation);
    elog(LOG, "[5] ConvertBatchPipelineResults返回");
    
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

/* ========================================================================= */
/*                              元数据准备函数 (优化版)                        */
/* ========================================================================= */

/* 
 * 优化 1: 消除双重扫描
 * 只读取 Meta Pages 获取聚类中心和起始页，绝对不遍历数据页链表。
 */
static int
PrepareMetaInfoOnly(IndexScanDesc scan,
                    int** cluster_size_out,
                    BlockNumber** cluster_pages_out,
                    float** cluster_centers_flat_out,
                    int* n_total_clusters_out, 
                    int* n_total_vectors_out)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    /* 不需要 tupdesc，因为我们直接读取结构体 */
    int totalLists = 0;
    int dimensions = so->dimensions;

    /* 获取总列表数 */
    IvfflatGetMetaPageInfo(scan->indexRelation, &totalLists, NULL);
    
    if (totalLists <= 0) return -1;
    
    /* 分配输出数组 */
    BlockNumber* cluster_start_pages = (BlockNumber*)palloc(totalLists * sizeof(BlockNumber));
    float* cluster_centers_flat = (float*)palloc(totalLists * dimensions * sizeof(float));
    
    /* 遍历 Meta Pages (存储聚类中心列表的页面) */
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
            
            ItemId itemId = PageGetItemId(cpage, offno);
            if (!ItemIdIsUsed(itemId)) continue;
            
            /* 【修复点】: 这里是 IvfflatList 结构体，不是 IndexTuple */
            IvfflatList list = (IvfflatList) PageGetItem(cpage, itemId);
            
            /* 直接从结构体获取起始页 */
            cluster_start_pages[center_idx] = list->startPage;
            
            /* 直接从结构体获取聚类中心向量 */
            /* IvfflatList 结构体定义中包含 Vector center; 它是变长的，但在内存中是连续的 */
            Vector *center_vec = &list->center;
            
            /* 注意：这里不需要 index_getattr，因为不是 Tuple */
            memcpy(cluster_centers_flat + center_idx * dimensions, 
                   center_vec->x, dimensions * sizeof(float));
            
            center_idx++;
        }
        nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;
        UnlockReleaseBuffer(cbuf);
    }
    
    *cluster_size_out = NULL;
    *cluster_pages_out = cluster_start_pages;
    *cluster_centers_flat_out = cluster_centers_flat;
    *n_total_clusters_out = center_idx;
    *n_total_vectors_out = 0;
    
    return 0;
}

/* ========================================================================= */
/*                              结果转换函数 (优化版：TID 排序回表)            */
/* ========================================================================= */

#ifdef USE_CUDA

/* 用于回表排序的辅助结构 */
typedef struct HeapFetchRequest {
    ItemPointerData tid;
    int32   query_idx;
    int32   k_idx;
    float   distance;
    int32   original_index; /* 保持稳定排序或调试用 */
} HeapFetchRequest;

/* qsort 比较函数：按 BlockNumber 升序，OffNumber 升序 */
static int
compare_fetch_requests(const void *a, const void *b)
{
    const HeapFetchRequest *ra = (const HeapFetchRequest *)a;
    const HeapFetchRequest *rb = (const HeapFetchRequest *)b;
    
    BlockNumber ba = ItemPointerGetBlockNumber(&ra->tid);
    BlockNumber bb = ItemPointerGetBlockNumber(&rb->tid);
    
    if (ba < bb) return -1;
    if (ba > bb) return 1;
    
    OffsetNumber oa = ItemPointerGetOffsetNumber(&ra->tid);
    OffsetNumber ob = ItemPointerGetOffsetNumber(&rb->tid);
    
    if (oa < ob) return -1;
    if (oa > ob) return 1;
    
    return 0;
}

#endif /* USE_CUDA */

static void
ConvertBatchPipelineResults(IndexScanDesc scan, float** topk_dist, int** topk_index,
                            int n_query, int k, BatchBuffer* result_buffer,
                            ItemPointer global_tids, int n_total_vectors, Relation indexRelation)
{
    /* 1. 安全检查 */
    if (!result_buffer || !global_tids || !indexRelation) {
        elog(ERROR, "ConvertBatchPipelineResults: 关键参数为 NULL");
        return;
    }
    if (!topk_dist || !topk_index) {
        elog(ERROR, "ConvertBatchPipelineResults: topk_dist 或 topk_index 为 NULL");
        return;
    }

    /* 2. 获取快照并校验 */
    Snapshot snapshot = GetActiveSnapshot();
    if (snapshot == NULL) {
        elog(ERROR, "ConvertBatchPipelineResults: 无法获取活跃快照");
        return;
    }

    /* 3. 打开堆表 */
    Oid heapRelid = IndexGetRelation(RelationGetRelid(indexRelation), false);
    Relation heapRelation = table_open(heapRelid, AccessShareLock);
    TupleDesc heapDesc = RelationGetDescr(heapRelation);
    
    /* 4. 查找 payload 列 */
    int id_attnum = -1;
    for (int i = 1; i <= heapDesc->natts; i++) {
        Form_pg_attribute attr = TupleDescAttr(heapDesc, i - 1);
        if (strcmp(NameStr(attr->attname), "id") == 0) {
            id_attnum = i;
            break;
        }
    }

#ifdef USE_CUDA
    int total_items = n_query * k;
    int valid_items_count = 0;
    
    /* 5. 分配请求数组 (使用当前上下文) */
    HeapFetchRequest *requests = (HeapFetchRequest *)palloc(total_items * sizeof(HeapFetchRequest));

    /* 6. 收集并预校验 TID */
    /* 关键修复：添加对齐检查 */
    if (n_total_vectors > 1) {
        size_t actual_offset = (char*)&global_tids[1] - (char*)&global_tids[0];
        if (actual_offset != sizeof(ItemPointerData)) {
            elog(ERROR, "ConvertBatchPipelineResults: ItemPointerData 对齐异常 - sizeof=%zu, actual_offset=%zu", 
                 sizeof(ItemPointerData), actual_offset);
            pfree(requests);
            table_close(heapRelation, AccessShareLock);
            return;
        }
    }
    
    for (int q = 0; q < n_query; q++) {
        for (int i = 0; i < k; i++) {
            int vec_idx = topk_index[q][i];
            int buffer_idx = q * k + i;
            
            /* 设置默认值 */
            result_buffer->query_ids[buffer_idx] = q;
            result_buffer->global_vector_indices[buffer_idx] = -1; // 默认值
            result_buffer->distances[buffer_idx] = (vec_idx < 0) ? INFINITY : topk_dist[q][i];
            ItemPointerSetInvalid(&result_buffer->vector_ids[buffer_idx]);
            
            /* 关键修复：更严格的边界检查 */
            if (vec_idx >= 0 && vec_idx < n_total_vectors) {
                /* 验证 vec_idx 不会导致指针越界 */
                ItemPointer tid = &global_tids[vec_idx];
                
                /* 验证 TID 指针在合法范围内 */
                ItemPointer first_tid = &global_tids[0];
                ItemPointer last_tid = &global_tids[n_total_vectors - 1];
                if (tid < first_tid || tid > last_tid) {
                    elog(WARNING, "ConvertBatchPipelineResults: TID 指针越界 - vec_idx=%d, tid=%p, range=[%p, %p]", 
                         vec_idx, (void*)tid, (void*)first_tid, (void*)last_tid);
                    continue;
                }
                
                /* 物理合法性检查 */
                if (ItemPointerIsValid(tid)) {
                    BlockNumber block = ItemPointerGetBlockNumber(tid);
                    OffsetNumber offset = ItemPointerGetOffsetNumber(tid);
                    
                    /* 验证 BlockNumber 和 OffsetNumber 的合理性 */
                    if (block == InvalidBlockNumber || offset == InvalidOffsetNumber || 
                        offset == 0 || offset > MaxHeapTuplesPerPage) {
                        elog(DEBUG2, "ConvertBatchPipelineResults: TID 物理值无效 - vec_idx=%d, block=%u, offset=%u", 
                             vec_idx, block, offset);
                        continue;
                    }
                    
                    requests[valid_items_count].tid = *tid;
                    requests[valid_items_count].query_idx = q;
                    requests[valid_items_count].k_idx = i;
                    requests[valid_items_count].original_index = buffer_idx;
                    valid_items_count++;
                }
            } else if (vec_idx != -1) {
                /* 记录异常索引值 */
                elog(DEBUG2, "ConvertBatchPipelineResults: GPU返回异常索引 - vec_idx=%d, n_total_vectors=%d", 
                     vec_idx, n_total_vectors);
            }
        }
    }

    /* 7. TID 排序：极大地提高磁盘 I/O 效率 */
    if (valid_items_count > 0) {
        qsort(requests, valid_items_count, sizeof(HeapFetchRequest), compare_fetch_requests);
    }

    /* 8. 批量回表 - 核心修复点 */
    BufferAccessStrategy bas = GetAccessStrategy(BAS_BULKREAD); // 使用批量读取策略
    
    /* 关键修复：验证 k 的一致性，防止 buffer_dest_idx 计算错误 */
    if (result_buffer->k != k) {
        elog(ERROR, "ConvertBatchPipelineResults: k 不一致 - result_buffer->k=%d, 参数k=%d", 
             result_buffer->k, k);
        FreeAccessStrategy(bas);
        pfree(requests);
        table_close(heapRelation, AccessShareLock);
        return;
    }
    
    for (int i = 0; i < valid_items_count; i++) {
        HeapFetchRequest *req = &requests[i];
        
        /* 关键修复：严格验证所有索引 */
        if (req->query_idx < 0 || req->query_idx >= n_query) {
            elog(WARNING, "ConvertBatchPipelineResults: 无效的query_idx=%d (n_query=%d)", 
                 req->query_idx, n_query);
            continue;
        }
        if (req->k_idx < 0 || req->k_idx >= k) {
            elog(WARNING, "ConvertBatchPipelineResults: 无效的k_idx=%d (k=%d)", 
                 req->k_idx, k);
            continue;
        }
        
        int buffer_dest_idx = req->query_idx * k + req->k_idx;
        
        /* 再次验证索引安全性 */
        if (buffer_dest_idx < 0 || buffer_dest_idx >= total_items) {
            elog(WARNING, "ConvertBatchPipelineResults: buffer_dest_idx越界 - idx=%d, total=%d, query_idx=%d, k_idx=%d", 
                 buffer_dest_idx, total_items, req->query_idx, req->k_idx);
            continue;
        }

        /* 关键修复：验证 TID 的物理合法性 */
        BlockNumber block = ItemPointerGetBlockNumber(&req->tid);
        OffsetNumber offset = ItemPointerGetOffsetNumber(&req->tid);
        if (block == InvalidBlockNumber || offset == InvalidOffsetNumber || 
            offset == 0 || offset > MaxHeapTuplesPerPage) {
            elog(DEBUG2, "ConvertBatchPipelineResults: TID 物理值无效 - block=%u, offset=%u", 
                 block, offset);
            result_buffer->global_vector_indices[buffer_dest_idx] = -1;
            continue;
        }

        /* 设置 TID */
        result_buffer->vector_ids[buffer_dest_idx] = req->tid;

        if (id_attnum > 0) {
            /* 关键修复：确保 snapshot 仍然有效 */
            if (snapshot == NULL) {
                elog(ERROR, "ConvertBatchPipelineResults: snapshot 在循环中变为 NULL");
                FreeAccessStrategy(bas);
                pfree(requests);
                table_close(heapRelation, AccessShareLock);
                return;
            }
            
            HeapTupleData tuple;
            tuple.t_self = req->tid;
            Buffer buf = InvalidBuffer;

            /* 使用更安全的 TAM 接口 */
            if (heap_fetch(heapRelation, snapshot, &tuple, &buf, false)) {
                bool isnull;
                Datum d = heap_getattr(&tuple, id_attnum, heapDesc, &isnull);
                if (!isnull) {
                    result_buffer->global_vector_indices[buffer_dest_idx] = DatumGetInt32(d);
                } else {
                    result_buffer->global_vector_indices[buffer_dest_idx] = -1;
                }
                if (BufferIsValid(buf)) {
                    ReleaseBuffer(buf);
                }
            } else {
                /* 如果 TID 在当前快照不可见或已失效 */
                result_buffer->global_vector_indices[buffer_dest_idx] = -1;
            }
        }
    }

    FreeAccessStrategy(bas);
    pfree(requests);
#else
    elog(ERROR, "ConvertBatchPipelineResults: CUDA support required");
#endif
    table_close(heapRelation, AccessShareLock);
}
