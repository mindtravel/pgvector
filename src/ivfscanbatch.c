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

/* C++ Wrapper 声明 */
extern int batch_search_pipeline_wrapper(float* d_query_batch,
                                         int* d_cluster_size,
                                         float* d_cluster_vectors,
                                         float* d_cluster_centers,
                                         int* d_initial_indices,
                                         float* d_topk_dist,
                                         int* d_topk_index,
                                         int n_query, int n_dim, int n_total_cluster,
                                         int n_total_vectors, int n_probes, int k);


/* ========================================================================= */
/*                              核心处理逻辑                                  */
/* ========================================================================= */

/*
 * ProcessBatchQueriesGPU
 * 
 * 流程优化：
 * 1. PrepareMetaInfoOnly: 快速扫描元数据，计算需要的显存大小。
 * 2. cuda_malloc: 一次性分配 GPU 显存。
 * 3. Pipeline Loop: 再次扫描索引，将向量读入 Pinned Buffer，满了就 Async 发送。
 * 4. Kernel: GPU 计算 TopK。
 * 5. Convert: 将 GPU 返回的 index 映射回 TID，并延迟回表 (Heap Fetch)。
 */
void
ProcessBatchQueriesGPU(IndexScanDesc scan, ScanKeyBatch batch_keys, int k)
{
    IvfflatBatchScanOpaque so;
    int nbatch;
    int dimensions;

    int* predicted_cluster_sizes;
    BlockNumber* cluster_start_pages;
    float* cluster_centers_flat;
    int n_total_clusters;
    int n_total_vectors;

    float* query_batch_flat;
    ItemPointer global_tids;

    float *d_query_batch, *d_cluster_vectors, *d_cluster_centers, *d_topk_dist;
    int *d_cluster_size, *d_initial_indices, *d_topk_index;

    so = (IvfflatBatchScanOpaque)scan->opaque;
    nbatch = batch_keys->nkeys;
    dimensions = so->dimensions;
    so->batch_processing_complete = false;
    
    /* 重新创建 result_buffer */
    if (so->result_buffer != NULL) {
        if (so->result_buffer->n_queries != nbatch || so->result_buffer->k != k) {
            so->result_buffer = NULL;
        }
    }
    
    /* ========== 步骤1: 获取预估元数据 (用于分配内存) ========== */
    predicted_cluster_sizes = NULL;  /* 预估值，用于内存分配 */
    cluster_start_pages = NULL;
    cluster_centers_flat = NULL;
    n_total_clusters = 0;
    n_total_vectors = 0;
    
    if (PrepareMetaInfoOnly(scan, &predicted_cluster_sizes, &cluster_start_pages, 
                            &cluster_centers_flat, &n_total_clusters, &n_total_vectors) != 0) {
        elog(ERROR, "ProcessBatchQueriesGPU: 元数据准备失败");
        return;
    }
    
    if (n_total_clusters <= 0 || n_total_vectors <= 0) {
        elog(ERROR, "ProcessBatchQueriesGPU: 无效的元数据 (n_total_clusters=%d, n_total_vectors=%d)",
             n_total_clusters, n_total_vectors);
        return;
    }
    
    /* ========== 步骤2: 准备 query_batch (扁平化) ========== */
    query_batch_flat = (float*)palloc(nbatch * dimensions * sizeof(float));
    /* 正确提取向量数据：VectorBatch 中的向量是连续存储的，但每个向量包含 Vector 结构头 */
    /* 需要逐个提取每个向量的 x[] 数组部分 */
    for (int i = 0; i < nbatch; i++) {
        Datum vec_datum = ScanKeyBatchGetVector(batch_keys, i);
        Vector *vec = DatumGetVector(vec_datum);
        memcpy(query_batch_flat + i * dimensions, vec->x, dimensions * sizeof(float));
    }
    
    
    /* ========== 步骤3: CPU 端分配扁平的 TID 映射表 (不传 GPU) ========== */
    /* 
     * 注意：分配稍微多一点的内存以防万一 Pass 2 读到的比 Pass 1 多
     * (虽然在没有并发插入的情况下不太可能，但为了安全)
     */
    int max_vectors = (int)(n_total_vectors * 1.1) + 1000;
    global_tids = (ItemPointer)palloc0(max_vectors * sizeof(ItemPointerData));
    
    /* ========== 步骤3.5: 准备实际大小统计数组 (用于修正) ========== */
    int* actual_cluster_sizes = (int*)palloc0(n_total_clusters * sizeof(int));
    
    /* GPU 内存指针 */
    d_query_batch = NULL;
    d_cluster_vectors = NULL;
    d_cluster_centers = NULL;
    d_topk_dist = NULL;
    d_cluster_size = NULL;
    d_initial_indices = NULL;
    d_topk_index = NULL;
    
    /* 计算分配大小 */
    size_t query_batch_size = (size_t)nbatch * dimensions * sizeof(float);
    size_t cluster_size_size = (size_t)n_total_clusters * sizeof(int);
    size_t cluster_vectors_size = (size_t)max_vectors * dimensions * sizeof(float);  /* 使用 max_vectors 安全 */
    size_t cluster_centers_size = (size_t)n_total_clusters * dimensions * sizeof(float);
    size_t initial_indices_size = (size_t)nbatch * n_total_clusters * sizeof(int);
    size_t topk_dist_size = (size_t)nbatch * k * sizeof(float);
    size_t topk_index_size = (size_t)nbatch * k * sizeof(int);
    
    /* ========== 步骤4: GPU 内存分配 ========== */
    cuda_malloc((void**)&d_query_batch, query_batch_size);
    cuda_malloc((void**)&d_cluster_size, cluster_size_size);
    cuda_malloc((void**)&d_cluster_vectors, cluster_vectors_size);
    cuda_malloc((void**)&d_cluster_centers, cluster_centers_size);
    cuda_malloc((void**)&d_topk_dist, topk_dist_size);
    cuda_malloc((void**)&d_initial_indices, initial_indices_size);
    cuda_malloc((void**)&d_topk_index, topk_index_size);
    
    /* ========== 步骤5: CPU -> GPU 静态数据复制 ========== */
    cuda_memcpy_h2d(d_query_batch, query_batch_flat, query_batch_size);
    /* 注意：这里先不要复制 d_cluster_size，等读取完实际数据后再复制 */
    cuda_memcpy_h2d(d_cluster_centers, cluster_centers_flat, cluster_centers_size);
    
    pfree(query_batch_flat);
    pfree(cluster_centers_flat);
    
    /* ========== 步骤6: 流式读取向量数据并传输 (Gather Loop) ========== */
    GpuPipelineContext pipeline_ctx;
    /* 初始化双缓冲流水线 */
    cuda_pipeline_init(&pipeline_ctx, dimensions, max_vectors, d_cluster_vectors);
    
    TupleDesc tupdesc = RelationGetDescr(scan->indexRelation);
    int global_vec_idx = 0;
    
    for (int i = 0; i < n_total_clusters; i++) {
        BlockNumber searchPage = cluster_start_pages[i];
        int cluster_vec_count = 0;  /* 当前 Cluster 的实际计数器 */
        
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
                
                bool isnull = false;
                Datum vector_datum = index_getattr(itup, 1, tupdesc, &isnull);
                
                if (!isnull) {
                    /* 安全检查：防止溢出缓冲区 */
                    if (global_vec_idx >= max_vectors) {
                        /* 这种情况极少见，除非有大量并发插入 */
                        elog(WARNING, "ProcessBatchQueriesGPU: Scan exceeded allocated buffer size, truncating.");
                        break;
                    }
                    
                    Vector *vec = DatumGetVector(vector_datum);
                    
                    /* === 双缓冲逻辑修改开始 === */
                    
                    /* 1. 获取当前活跃的 Buffer 指针 */
                    int active_idx = pipeline_ctx.active_buf_idx;
                    float* current_buffer = pipeline_ctx.h_vec_buffers[active_idx];
                    
                    /* 2. 计算在当前 Buffer 中的偏移量 */
                    size_t buf_offset = pipeline_ctx.current_counts[active_idx] * dimensions;
                    
                    /* 3. 拷贝数据 */
                    memcpy(current_buffer + buf_offset, vec->x, dimensions * sizeof(float));
                    
                    /* 4. 更新计数 */
                    pipeline_ctx.current_counts[active_idx]++;
                    
                    /* 5. 缓冲区满？Flush! (这会自动切换 buffer 并等待) */
                    if (pipeline_ctx.current_counts[active_idx] >= pipeline_ctx.chunk_capacity) {
                        cuda_pipeline_flush_vectors_only(&pipeline_ctx);
                    }
                    /* === 双缓冲逻辑修改结束 === */
                    
                    /* 记录 TID */
                    global_tids[global_vec_idx] = itup->t_tid;
                    
                    global_vec_idx++;
                    cluster_vec_count++;
                }
            }
            searchPage = IvfflatPageGetOpaque(page)->nextblkno;
            UnlockReleaseBuffer(buf);
        }
        
        /* 记录该 Cluster 的实际大小 */
        actual_cluster_sizes[i] = cluster_vec_count;
    }
    
    /* 刷新剩余数据 */
    cuda_pipeline_flush_vectors_only(&pipeline_ctx);
    /* 清理 */
    cuda_pipeline_free(&pipeline_ctx);
    
    /* ========== 关键步骤6.5: 将实际统计的 Cluster Size 传给 GPU ========== */
    /* 
     * 如果这里不更新，GPU 依然使用 Pass 1 的 predicted_sizes 来计算偏移，
     * 一旦有偏差，后续所有 Cluster 的读取都会错位。
     */
    cuda_memcpy_h2d(d_cluster_size, actual_cluster_sizes, cluster_size_size);

    /* ========== 步骤7: 初始化 d_initial_indices (Coarse-Quantization Optimization) ========== */
    int* initial_indices_host = (int*)palloc(nbatch * n_total_clusters * sizeof(int));
    for (int query_idx = 0; query_idx < nbatch; query_idx++) {
        for (int cluster_idx = 0; cluster_idx < n_total_clusters; cluster_idx++) {
            initial_indices_host[query_idx * n_total_clusters + cluster_idx] = cluster_idx;
        }
    }
    cuda_memcpy_h2d(d_initial_indices, initial_indices_host, nbatch * n_total_clusters * sizeof(int));
    pfree(initial_indices_host);

    /* ========== 步骤8: 调用 GPU Kernel ========== */

    /* 传递 total_vectors 应为实际读取的数量 */
    int result = batch_search_pipeline_wrapper(d_query_batch, d_cluster_size, d_cluster_vectors,
                                                d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index,
                                                nbatch, dimensions, n_total_clusters,
                                                global_vec_idx /* 使用实际总数 */, so->probes, k);
    
    if (result != 0) {
        elog(ERROR, "ProcessBatchQueriesGPU: Kernel 执行失败 (Error: %d)", result);
        cuda_cleanup_memory(d_query_batch, d_cluster_size, d_cluster_vectors,
                        d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
        return;
    }

    /* ========== 步骤9: GPU -> CPU 结果回传 ========== */
    float* topk_dist_flat = (float*)palloc(nbatch * k * sizeof(float));
    int* topk_index_flat = (int*)palloc(nbatch * k * sizeof(int));
    
    cuda_memcpy_d2h(topk_dist_flat, d_topk_dist, nbatch * k * sizeof(float));
    cuda_memcpy_d2h(topk_index_flat, d_topk_index, nbatch * k * sizeof(int)); 

    /* 组织为二维指针数组 (兼容旧接口) */
    float** topk_dist = (float**)palloc(nbatch * sizeof(float*));
    int** topk_index = (int**)palloc(nbatch * sizeof(int*));
    for (int i = 0; i < nbatch; i++) {
        topk_dist[i] = topk_dist_flat + i * k;
        topk_index[i] = topk_index_flat + i * k;
    }
    
    /* ========== 步骤10: 转换结果 (延迟回表) ========== */
    so->result_buffer = CreateBatchBuffer(nbatch, k, dimensions, CurrentMemoryContext);
    
    ConvertBatchPipelineResults(scan, topk_dist, topk_index, nbatch, k,
                                so->result_buffer, global_tids, global_vec_idx, scan->indexRelation);
    
    /* ========== 清理资源 ========== */
    pfree(global_tids);
    pfree(actual_cluster_sizes);
    pfree(predicted_cluster_sizes);
    pfree(cluster_start_pages);
    
    cuda_cleanup_memory(d_query_batch, d_cluster_size, d_cluster_vectors,
                    d_cluster_centers, d_initial_indices, d_topk_dist, d_topk_index);
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