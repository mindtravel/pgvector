/* scanbatch.c */
#include "postgres.h"
#include "access/genam.h"
#include "access/skey.h"
#include "utils/datum.h"
#include "access/relscan.h"
#include "executor/tuptable.h"
#include "nodes/execnodes.h"
#include "funcapi.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "catalog/index.h"
#include "catalog/pg_attribute.h"
#include "utils/lsyscache.h"
#include "utils/syscache.h"
#include "fmgr.h"
#include "utils/rel.h"
#include "access/htup_details.h"
#include "access/htup.h"
#include "access/heapam.h"
#include "access/tableam.h"
#include "access/table.h"
#include "storage/bufmgr.h"
#include "utils/snapmgr.h"
#include "utils/tuplestore.h"
#include "executor/executor.h"
#include "miscadmin.h"
#include "scanbatch.h"
#include "vector.h"
#include "vector_batch.h"
#include "ivfscanbatch.h"


ScanKeyBatch
ScanKeyBatchCreate(int nkeys, int vec_dim)
{
    ScanKeyBatch batch;

    /* 分配ScankeyBatch结构 */
    batch = (ScanKeyBatch)palloc0(sizeof(ScanKeyBatchData));
    batch->nkeys = nkeys;
    batch->vec_dim = vec_dim;  /* 存储实际维度，不是字节大小 */
    batch->first_vector_ptr = NULL;  /* 将在ScanKeyBatchAddData中设置 */
    batch->vec_size = VECTOR_SIZE(vec_dim);  /* 存储字节大小 */

    return batch;
}


PG_FUNCTION_INFO_V1(batch_vector_search_cos);
Datum
batch_vector_search_cos(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    TupleDesc     tupdesc;
    Tuplestorestate *tupstore;
    MemoryContext oldcontext;
    
    Datum *elems;
    bool *nulls;
    int nelems;
    
    Oid index_oid;
    VectorBatch *query_vectors_batch;
    int k;
    int n_querys;
    int vec_dim;
    
    ScanKeyBatch batch_keys;
    Relation index;
    IndexScanDesc scan;
    
    /* 1. 检查调用上下文，确保支持 Materialize 模式 */
    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
        ereport(ERROR, 
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("set-valued function called in context that cannot accept a set")));
    
    if (!(rsinfo->allowedModes & SFRM_Materialize))
        ereport(ERROR, 
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("materialize mode required, but it is not allowed in this context")));
    
    /* 2. 获取参数 */
    index_oid = PG_GETARG_OID(0);
    k = PG_GETARG_INT32(2);
    
    elog(LOG, "batch_vector_search_cos: 接收到的参数 - index_oid=%u, k=%d", index_oid, k);
    
    /* 检查参数是否为 NULL */
    if (PG_ARGISNULL(1))
        elog(ERROR, "batch_vector_search_cos: query_vectors cannot be NULL");
    
    /* 获取 vector_batch 参数 */
    query_vectors_batch = PG_GETARG_VECTOR_BATCH_P(1);
    n_querys = query_vectors_batch->count;
    vec_dim = query_vectors_batch->dim;
    
    elog(LOG, "batch_vector_search_cos: vector_batch - count=%d, dim=%d, size=%u", 
         n_querys, vec_dim, VARSIZE(query_vectors_batch));
    
    /* 3. 初始化 TupleDesc */
    tupdesc = rsinfo->expectedDesc;
    if (tupdesc == NULL)
    {
        /* 如果没有提供 expectedDesc，尝试从函数返回类型获取 */
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            elog(ERROR, "return type must be a row type");
        tupdesc = BlessTupleDesc(tupdesc);
    }
    else
    {
        /* 如果提供了 expectedDesc，需要确保它是 blessed */
        tupdesc = BlessTupleDesc(tupdesc);
    }
    
    /* 4. 创建 Tuplestore - 必须在 per-query context 中创建 */
    oldcontext = MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
    
    /* work_mem 是允许使用的最大内存，超过会溢出到磁盘 */
    tupstore = tuplestore_begin_heap(true, false, work_mem);
    
    MemoryContextSwitchTo(oldcontext);
    
    /* 5. 为 ScanKeyBatchAddData 准备数据 */
    elems = (Datum *) palloc(n_querys * sizeof(Datum));
    nulls = (bool *) palloc(n_querys * sizeof(bool));
    nelems = n_querys;
    
    for (int i = 0; i < n_querys; i++)
    {
        Vector *vec = VectorBatchGetVector(query_vectors_batch, i);
        elems[i] = PointerGetDatum(vec);
        nulls[i] = false;
    }
    
    /* 6. 创建批量键 */
    batch_keys = ScanKeyBatchCreate(n_querys, vec_dim);
    if (!batch_keys) {
        elog(ERROR, "batch_vector_search_cos: ScanKeyBatchCreate 失败");
    }
    
    /* 设置批量键指向的数据区域 */ 
    ScanKeyBatchAddData(batch_keys, elems, nulls, nelems);
    
    /* 7. 创建索引扫描 */
    index = index_open(index_oid, AccessShareLock);
    scan = ivfflatbatchbeginscan(index, 1, batch_keys);
    if (!scan) {
        index_close(index, AccessShareLock);
        elog(ERROR, "batch_vector_search_cos: ivfflatbatchbeginscan 失败");
    }
    
    /* 8. GPU 批处理：使用零拷贝版本，直接将结果写入 tuplestore */
    /* 这样可以减少内存分配和数据复制，提高性能 */
    ivfflatbatchgettuple(scan, ForwardScanDirection, tupstore, tupdesc, k, COSINE_DISTANCE);
    
    /* 9. 标记 tuplestore 完成 */
    tuplestore_donestoring(tupstore);
    
    /* 10. 清理资源 */
    index_close(index, AccessShareLock);
    ivfflatbatchendscan(scan);
    
    /* 11. 设置返回信息 */
    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tupstore;
    rsinfo->setDesc = tupdesc;
    
    return (Datum) 0;
}


PG_FUNCTION_INFO_V1(batch_vector_search_l2);
Datum
batch_vector_search_l2(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    TupleDesc     tupdesc;
    Tuplestorestate *tupstore;
    MemoryContext oldcontext;
    
    Datum *elems;
    bool *nulls;
    int nelems;
    
    Oid index_oid;
    VectorBatch *query_vectors_batch;
    int k;
    int n_querys;
    int vec_dim;
    
    ScanKeyBatch batch_keys;
    Relation index;
    IndexScanDesc scan;
    
    /* 1. 检查调用上下文，确保支持 Materialize 模式 */
    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
        ereport(ERROR, 
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("set-valued function called in context that cannot accept a set")));
    
    if (!(rsinfo->allowedModes & SFRM_Materialize))
        ereport(ERROR, 
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("materialize mode required, but it is not allowed in this context")));
    
    /* 2. 获取参数 */
    index_oid = PG_GETARG_OID(0);
    k = PG_GETARG_INT32(2);
    
    elog(LOG, "batch_vector_search_l2: 接收到的参数 - index_oid=%u, k=%d", index_oid, k);
    
    /* 检查参数是否为 NULL */
    if (PG_ARGISNULL(1))
        elog(ERROR, "batch_vector_search_l2: query_vectors cannot be NULL");
    
    /* 获取 vector_batch 参数 */
    query_vectors_batch = PG_GETARG_VECTOR_BATCH_P(1);
    n_querys = query_vectors_batch->count;
    vec_dim = query_vectors_batch->dim;
    
    elog(LOG, "batch_vector_search_l2: vector_batch - count=%d, dim=%d, size=%u", 
         n_querys, vec_dim, VARSIZE(query_vectors_batch));
    
    /* 3. 初始化 TupleDesc */
    tupdesc = rsinfo->expectedDesc;
    if (tupdesc == NULL)
    {
        /* 如果没有提供 expectedDesc，尝试从函数返回类型获取 */
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            elog(ERROR, "return type must be a row type");
        tupdesc = BlessTupleDesc(tupdesc);
    }
    else
    {
        /* 如果提供了 expectedDesc，需要确保它是 blessed */
        tupdesc = BlessTupleDesc(tupdesc);
    }
    
    /* 4. 创建 Tuplestore - 必须在 per-query context 中创建 */
    oldcontext = MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
    
    /* work_mem 是允许使用的最大内存，超过会溢出到磁盘 */
    tupstore = tuplestore_begin_heap(true, false, work_mem);
    
    MemoryContextSwitchTo(oldcontext);
    
    /* 5. 为 ScanKeyBatchAddData 准备数据 */
    elems = (Datum *) palloc(n_querys * sizeof(Datum));
    nulls = (bool *) palloc(n_querys * sizeof(bool));
    nelems = n_querys;
    
    for (int i = 0; i < n_querys; i++)
    {
        Vector *vec = VectorBatchGetVector(query_vectors_batch, i);
        elems[i] = PointerGetDatum(vec);
        nulls[i] = false;
    }
    
    /* 6. 创建批量键 */
    batch_keys = ScanKeyBatchCreate(n_querys, vec_dim);
    if (!batch_keys) {
        elog(ERROR, "batch_vector_search_l2: ScanKeyBatchCreate 失败");
    }
    
    /* 设置批量键指向的数据区域 */ 
    ScanKeyBatchAddData(batch_keys, elems, nulls, nelems);
    
    /* 7. 创建索引扫描 */
    index = index_open(index_oid, AccessShareLock);
    scan = ivfflatbatchbeginscan(index, 1, batch_keys);
    if (!scan) {
        index_close(index, AccessShareLock);
        elog(ERROR, "batch_vector_search_l2: ivfflatbatchbeginscan 失败");
    }
    
    /* 8. GPU 批处理：使用零拷贝版本，直接将结果写入 tuplestore */
    /* 这样可以减少内存分配和数据复制，提高性能 */
    ivfflatbatchgettuple(scan, ForwardScanDirection, tupstore, tupdesc, k, L2_DISTANCE);
    
    /* 9. 标记 tuplestore 完成 */
    tuplestore_donestoring(tupstore);
    
    /* 10. 清理资源 */
    index_close(index, AccessShareLock);
    ivfflatbatchendscan(scan);
    
    /* 11. 设置返回信息 */
    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tupstore;
    rsinfo->setDesc = tupdesc;
    
    return (Datum) 0;
}

void
ScanKeyBatchAddData(ScanKeyBatch batch, Datum *vectors, bool *nulls, int n_vectors)
{
    Vector* first_vec;

    /* 检查向量数量是否匹配 */
    if (n_vectors != batch->nkeys)
        elog(ERROR, "vector count mismatch: expected %d, got %d", batch->nkeys, n_vectors);
    
    /* 直接指向第一个向量，不需要循环检查 */
    /* 注意：vectors 数组中的每个元素都是通过 PointerGetDatum(vec) 创建的指针，
     * 所以应该使用 DatumGetPointer 而不是 DatumGetVector（后者会调用 PG_DETOAST_DATUM） */
    first_vec = (Vector *) DatumGetPointer(vectors[0]);
    
    if (first_vec == NULL)
        elog(ERROR, "ScanKeyBatchAddData: first vector pointer is NULL");

    /* 检查维度是否匹配 */
    if (first_vec->dim != batch->vec_dim)
        elog(ERROR, "vector dimension mismatch: expected %d, got %d", 
             batch->vec_dim, first_vec->dim);
    
    /* 设置第一个向量指针 */
    batch->first_vector_ptr = (void*)first_vec;
}

/*
 * 从ScankeyBatch获取向量
 */
Datum
ScanKeyBatchGetVector(ScanKeyBatch batch, int index)
{
    if (index < 0 || index >= batch->nkeys)
        elog(ERROR, "index out of bounds in ScanKeyBatchGetVector");

    if (batch->first_vector_ptr == NULL)
        elog(ERROR, "ScanKeyBatchGetVector: first_vector_ptr is NULL");

    /* 通过指针算术获取指定索引的向量 */
    return PointerGetDatum((Vector *)((char*)batch->first_vector_ptr + index * batch->vec_size));
}

/*
 * 获取连续存储的数据指针
 */
void*
ScanKeyBatchGetContinuousData(ScanKeyBatch batch)
{
    Vector *first_vec;
    
    if (batch->first_vector_ptr == NULL)
        elog(ERROR, "ScanKeyBatchGetContinuousData: first_vector_ptr is NULL");
    
    /* 返回第一个向量的数据部分（跳过Vector结构头） */
    first_vec = (Vector*)batch->first_vector_ptr;
    return &first_vec->x[0];
}