/* scanbatch.c */
#include "postgres.h"
#include "access/genam.h"
#include "access/skey.h"
#include "utils/datum.h"
#include "access/relscan.h"
#include "executor/tuptable.h"
#include "funcapi.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "catalog/index.h"
#include "catalog/pg_attribute.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "access/htup_details.h"
#include "access/htup.h"
#include "access/heapam.h"
#include "access/tableam.h"
#include "access/table.h"
#include "storage/bufmgr.h"
#include "utils/snapmgr.h"
#include "scanbatch.h"
#include "vector.h"
#include "vector_batch.h"
#include "ivfscanbatch.h"


// 函数声明 - 与ivfscanbatch.h保持一致
// extern IndexScanDesc ivfflatbatchbeginscan(Relation index, int norderbys, ScanKeyBatch batch_keys);
// extern void ivfflatbatchendscan(IndexScanDesc scan);


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


PG_FUNCTION_INFO_V1(batch_vector_search_c);
Datum
batch_vector_search_c(PG_FUNCTION_ARGS)
{
    Datum *elems;
    bool *nulls;
    int nelems;
    
    Oid index_oid;
    ArrayType *query_vectors_array;
    int k;
    int n_querys;
    int16 typlen;
    int vec_dim;

    FuncCallContext *funcctx;
    BatchSearchState *state;

    bool typbyval;
    char typalign;

    Datum values[3];
    bool tuple_nulls[3];
    HeapTuple tuple;
    
    int base_idx;
    int total_results;

    // elog(LOG, "batch_vector_search_c: 开始批量向量搜索");
    
    /* 获取参数 */
    index_oid           = PG_GETARG_OID(0);
    query_vectors_array = PG_GETARG_ARRAYTYPE_P(1);
    k                   = PG_GETARG_INT32(2);

    // elog(LOG, "batch_vector_search_c: 参数获取完成 - index_oid=%u, k=%d", index_oid, k);
    
    /* 解析查询向量数组结构，获取向量数量 */
    n_querys = ArrayGetNItems(
        ARR_NDIM(query_vectors_array),
        ARR_DIMS(query_vectors_array)
    );
    // elog(LOG, "batch_vector_search_c: 向量数量 = %d", n_querys);
    
    /* 解析向量内容，获取向量维度 */
    get_typlenbyvalalign(
        ARR_ELEMTYPE(query_vectors_array), 
        &typlen, &typbyval, &typalign
    );
    deconstruct_array(
        query_vectors_array, ARR_ELEMTYPE(query_vectors_array),
        typlen, typbyval, typalign, 
        &elems, &nulls, &nelems
    );

    if (nelems > 0 && !nulls[0]) {
        vec_dim = DatumGetVector(elems[0])->dim;
        // elog(LOG, "batch_vector_search_c: 向量维度 = %d", vec_dim);
    } else {
        elog(ERROR, "batch_vector_search_c: 无法获取向量维度，数组为空或第一个向量为NULL");
    }
    
    // 由于函数返回TABLE类型，我们需要使用SRF机制
    if (SRF_IS_FIRSTCALL()) {
        ScanKeyBatch batch_keys;
        Relation index;
        IndexScanDesc scan;
        int max_results;
        
        // 初始化函数上下文
        funcctx = SRF_FIRSTCALL_INIT();
        
        // 分配状态结构
        state = (BatchSearchState *)MemoryContextAllocZero(funcctx->multi_call_memory_ctx, sizeof(BatchSearchState));
        if (!state) {
            elog(ERROR, "batch_vector_search_c: 内存分配失败");
        }
        funcctx->user_fctx = state;
        
        /* 创建批量键 */
        batch_keys = ScanKeyBatchCreate(n_querys, vec_dim);
        if (!batch_keys) {
            elog(ERROR, "batch_vector_search_c: ScanKeyBatchCreate 失败");
        }
        
        elog(LOG, "batch_vector_search_c: 创建 batch_keys, nkeys=%d, vec_dim=%d, batch_keys=%p", 
             n_querys, vec_dim, (void*)batch_keys);
        
        /* 设置批量键指向的数据区域 */ 
        ScanKeyBatchAddData(batch_keys, elems, nulls, nelems);
        elog(LOG, "batch_vector_search_c: batch_keys 数据设置完成, nkeys=%d", batch_keys->nkeys);
        
        // 创建索引扫描
        // 注意：IndexScanDesc是通过RelationGetIndexScan在当前内存上下文（SRF内存上下文）中分配的
        // IndexScanDesc的生命周期由PostgreSQL管理，我们只需要清理opaque数据
        // 但是，tmpCtx也是在CurrentMemoryContext（SRF内存上下文）中创建的
        // 所以，tmpCtx会在SRF清理时自动删除，我们需要在SRF清理之前手动删除它
        index = index_open(index_oid, AccessShareLock);
        scan = ivfflatbatchbeginscan(index, 1, batch_keys);
        if (!scan) {
            index_close(index, AccessShareLock);
            /* batch_keys 由 SRF 内存上下文自动清理 */
            elog(ERROR, "batch_vector_search_c: ivfflatbatchbeginscan 失败");
        }
        // elog(LOG, "batch_vector_search_c: 索引扫描创建完成");
        
        // 一次性获取所有查询的所有结果
        // 直接在 state 中分配最终数据缓冲区，避免中间复制
        max_results = k * n_querys * 3; // 每个查询k个结果 × 查询数量 × 3个字段
        state->result_values = palloc(max_results * sizeof(Datum));
        state->result_nulls = palloc(max_results * sizeof(bool));
        state->current_result = 0;
        state->k = k;
        state->n_querys = n_querys;
        
        // 调用批量扫描函数，直接填充 state 中的数据缓冲区
        ivfflatbatchgettuple(scan, ForwardScanDirection, 
                            state->result_values, state->result_nulls, 
                             k);
        
        // 保存资源以便后续清理
        state->scan = scan;
        state->index = index;  // 保存index引用以便后续关闭
        state->batch_keys = batch_keys;
        state->heap_rel = NULL;  // 已经关闭，不需要保存
        state->heap_rel_opened_by_us = false;
        
        // 设置返回元组描述符
        if (get_call_result_type(fcinfo, NULL, &funcctx->tuple_desc) != TYPEFUNC_COMPOSITE)
            elog(ERROR, "return type must be a row type");
        funcctx->tuple_desc = BlessTupleDesc(funcctx->tuple_desc);
        
    }
    
    // 每次调用返回一个结果
    funcctx = SRF_PERCALL_SETUP();
    state = (BatchSearchState *)funcctx->user_fctx;
    
    // 计算总结果数：每个查询k个结果，共n_querys个查询
    total_results = state->k * state->n_querys;
    
    if (state->current_result >= total_results) {
        // 清理 PostgreSQL 资源
        // 注意：result_values、result_nulls、batch_keys 都在 SRF 内存上下文中分配，
        // 会在 SRF 结束时自动释放，不需要手动释放
        Relation index_to_close = state->index;
        
        // 先关闭 index，再清理 scan（避免访问已删除的内存）
        if (index_to_close) {
            index_close(index_to_close, AccessShareLock);
        }
        
        if (state->scan) {
            ivfflatbatchendscan(state->scan);
        }
        
        // 清空指针，避免后续访问（但不要手动释放，SRF 会自动清理）
        // 注意：不要在这里清空 batch_keys，因为它可能还在被 scan 使用
        state->index = NULL;
        state->scan = NULL;
        state->result_values = NULL;
        state->result_nulls = NULL;
        state->batch_keys = NULL;
        
        SRF_RETURN_DONE(funcctx);
    }
    
    // 直接从 state 中读取结果并返回（GetBatchResults 已经填充好了）
    base_idx = state->current_result * 3;
    values[0] = state->result_values[base_idx + 0];
    values[1] = state->result_values[base_idx + 1];
    values[2] = state->result_values[base_idx + 2];
    tuple_nulls[0] = state->result_nulls[base_idx + 0];
    tuple_nulls[1] = state->result_nulls[base_idx + 1];
    tuple_nulls[2] = state->result_nulls[base_idx + 2];
    
    state->current_result++;
    
    tuple = heap_form_tuple(funcctx->tuple_desc, values, tuple_nulls);
    SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
}

/*
 * 向ScankeyBatch添加vectorbatch
 */

void
ScanKeyBatchAddData(ScanKeyBatch batch, Datum *vectors, bool *nulls, int n_vectors)
{
    Vector* first_vec;

    /* 检查向量数量是否匹配 */
    if (n_vectors != batch->nkeys)
        elog(ERROR, "vector count mismatch: expected %d, got %d", batch->nkeys, n_vectors);
    
    /* 直接指向第一个向量，不需要循环检查 */
    first_vec = DatumGetVector(vectors[0]);

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