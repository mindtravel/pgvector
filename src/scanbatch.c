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

/* scanbatch_free 已删除 - ScanKeyBatch 由 SRF 内存上下文自动管理 */

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
    

    elog(LOG, "batch_vector_search_c: 开始批量向量搜索");
    
    /* 获取参数 */
    index_oid           = PG_GETARG_OID(0);
    query_vectors_array = PG_GETARG_ARRAYTYPE_P(1);
    k                   = PG_GETARG_INT32(2);

    elog(LOG, "batch_vector_search_c: 参数获取完成 - index_oid=%u, k=%d", index_oid, k);
    
    /* 解析查询向量数组结构，获取向量数量 */
    n_querys = ArrayGetNItems(
        ARR_NDIM(query_vectors_array),
        ARR_DIMS(query_vectors_array)
    );
    elog(LOG, "batch_vector_search_c: 向量数量 = %d", n_querys);
    
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
        elog(LOG, "batch_vector_search_c: 向量维度 = %d", vec_dim);
    } else {
        elog(ERROR, "batch_vector_search_c: 无法获取向量维度，数组为空或第一个向量为NULL");
    }
    
    // 由于函数返回TABLE类型，我们需要使用SRF机制
    if (SRF_IS_FIRSTCALL()) {
        ScanKeyBatch batch_keys;
        Relation index;
        IndexScanDesc scan;
        int max_results;
        int max_values;
        Datum *result_values;
        bool *result_nulls;
        int returned_tuples;
        
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
        
        /* 设置批量键指向的数据区域 */ 
        ScanKeyBatchAddData(batch_keys, elems, nulls, nelems);
        elog(LOG, "batch_vector_search_c: 所有向量添加完成");
        
        // 创建索引扫描
        index = index_open(index_oid, AccessShareLock);
        scan = ivfflatbatchbeginscan(index, 1, batch_keys);
        if (!scan) {
            index_close(index, AccessShareLock);
            /* batch_keys 由 SRF 内存上下文自动清理 */
            elog(ERROR, "batch_vector_search_c: ivfflatbatchbeginscan 失败");
        }
        elog(LOG, "batch_vector_search_c: 索引扫描创建完成");
        
        // 一次性获取所有查询的所有结果
        max_results = k * n_querys; // 每个查询k个结果 × 查询数量
        max_values = max_results * 3; // 每个结果有3个字段
        result_values = palloc(max_values * sizeof(Datum));
        result_nulls = palloc(max_values * sizeof(bool));
        returned_tuples = 0;
        
        elog(LOG, "batch_vector_search_c: 开始批量处理，预期最大结果数: %d", max_results);
        
        // 调用批量扫描函数，一次性处理所有查询
        ivfflatbatchgettuple(scan, ForwardScanDirection, 
                            result_values, result_nulls, 
                            max_values, &returned_tuples, k);
        
        elog(LOG, "batch_vector_search_c: 批量处理完成，返回结果数: %d", returned_tuples);
        
        // 保存结果数据到状态中
        state->result_values = result_values;
        state->result_nulls = result_nulls;
        state->returned_tuples = returned_tuples;
        state->current_result = 0;
        state->k = k;
        state->n_querys = n_querys;
        
        // 保存资源以便后续清理
        state->scan = scan;
        state->index = index;
        state->batch_keys = batch_keys;
        
        // 获取堆表 Relation（从索引中获取）
        // 标记是否由我们打开（需要关闭）
        state->heap_rel_opened_by_us = false;
        if (scan->heapRelation) {
            // 使用索引扫描中的 heapRelation（不需要我们关闭）
            state->heap_rel = scan->heapRelation;
        } else {
            // 如果 scan 中没有 heapRelation，从索引元数据中获取
            Oid heap_oid = IndexGetRelation(index_oid, false);
            if (!OidIsValid(heap_oid)) {
                elog(ERROR, "batch_vector_search_c: 无法获取堆表 OID");
            }
            state->heap_rel = table_open(heap_oid, AccessShareLock);
            state->heap_rel_opened_by_us = true;  // 标记为我们打开的
        }
        
        // 获取 vector_id 字段的属性编号（假设是第1列，通常是主键或id字段）
        {
            int attnum;
            
            // 使用 get_attnum 通过关系ID和属性名获取属性编号
            attnum = get_attnum(RelationGetRelid(state->heap_rel), "id");
            if (attnum == InvalidAttrNumber) {
                // 如果找不到 "id" 字段，尝试使用第1列
                state->vector_id_attnum = 1;
                elog(LOG, "batch_vector_search_c: 未找到 'id' 字段，使用第1列作为 vector_id");
            } else {
                state->vector_id_attnum = attnum;
                elog(LOG, "batch_vector_search_c: 找到 'id' 字段，属性编号 = %d", state->vector_id_attnum);
            }
        }
        
        // 设置返回元组描述符
        if (get_call_result_type(fcinfo, NULL, &funcctx->tuple_desc) != TYPEFUNC_COMPOSITE)
            elog(ERROR, "return type must be a row type");
        funcctx->tuple_desc = BlessTupleDesc(funcctx->tuple_desc);
        
        elog(LOG, "batch_vector_search_c: 初始化完成，准备返回 %d 个结果", returned_tuples);
    }
    
    // 每次调用返回一个结果
    funcctx = SRF_PERCALL_SETUP();
    state = (BatchSearchState *)funcctx->user_fctx;
    
    if (state->current_result >= state->returned_tuples) {
        // 清理 PostgreSQL 资源
        if (state->scan) {
            ivfflatbatchendscan(state->scan);  // 清理扫描资源和 tmpCtx
        }
        if (state->index) {
            index_close(state->index, AccessShareLock);
        }
        if (state->heap_rel && state->heap_rel_opened_by_us) {
            // 如果 heap_rel 是我们自己打开的，需要关闭它
            table_close(state->heap_rel, AccessShareLock);
            state->heap_rel = NULL;  // 避免重复关闭
        }
        /* batch_keys 由 SRF 内存上下文自动清理 */
        
        // result_values 和 result_nulls 由 SRF 内存上下文自动清理
        SRF_RETURN_DONE(funcctx);
    }
    
    // 返回当前结果
    if (!state->result_nulls[state->current_result * 3]) {
        // 直接从result_values数组中获取结果
        int base_idx = state->current_result * 3;
        
        values[0] = state->result_values[base_idx + 0]; // query_id
        values[1] = state->result_values[base_idx + 1]; // vector_id  
        values[2] = state->result_values[base_idx + 2]; // distance
        
        tuple_nulls[0] = state->result_nulls[base_idx + 0];
        tuple_nulls[1] = state->result_nulls[base_idx + 1];
        tuple_nulls[2] = state->result_nulls[base_idx + 2];
        
        {
            // values[1] 是 ItemPointer (TID)，需要从堆表中获取实际的 vector_id 值
            ItemPointer tid = (ItemPointer)DatumGetPointer(values[1]);
            
            // 调试：打印TID值
            if (tid && ItemPointerIsValid(tid)) {
                elog(LOG, "batch_vector_search_c: 调试 - TID值: (%u,%u)", 
                     ItemPointerGetBlockNumber(tid),
                     ItemPointerGetOffsetNumber(tid));
            }
            
            if (tid && ItemPointerIsValid(tid) && state->heap_rel) {
                HeapTupleData heap_tuple_data;
                HeapTuple     heap_tuple = &heap_tuple_data;
                bool          isnull;
                Datum         vector_id_datum;
                Snapshot      snapshot;
                Buffer        buffer = InvalidBuffer;
                bool          found;
                
                // 获取当前快照
                snapshot = GetActiveSnapshot();
                
                // 初始化 heap_tuple 的 t_self 为 TID
                heap_tuple->t_self = *tid;
                
                // 从堆表中获取元组（类似单次查询的方式）
                // heap_fetch 的签名：bool heap_fetch(Relation relation, Snapshot snapshot,
                //                                   HeapTuple tuple, Buffer *userbuf, bool keep_buf)
                found = heap_fetch(state->heap_rel, snapshot, heap_tuple, &buffer, false);
                
                if (found && HeapTupleIsValid(heap_tuple)) {
                    // 从元组中获取 vector_id 字段的值
                    vector_id_datum = heap_getattr(heap_tuple, 
                                                   state->vector_id_attnum,
                                                   RelationGetDescr(state->heap_rel),
                                                   &isnull);
                    
                    if (!isnull) {
                        // 将 Datum 转换为 int32
                        values[1] = vector_id_datum;
                        tuple_nulls[1] = false;
                        
                        elog(LOG, "batch_vector_search_c: 返回结果 %d - query_id=%d, vector_id=%d (tid: %u,%u), distance=%.6f", 
                             state->current_result, 
                             DatumGetInt32(values[0]), 
                             DatumGetInt32(values[1]),
                             ItemPointerGetBlockNumber(tid),
                             ItemPointerGetOffsetNumber(tid),
                             DatumGetFloat8(values[2]));
                    } else {
                        // vector_id 列为 NULL
                        tuple_nulls[1] = true;
                        elog(LOG, "batch_vector_search_c: 返回结果 %d - query_id=%d, vector_id=null (tid: %u,%u), distance=%.6f", 
                             state->current_result, DatumGetInt32(values[0]),
                             ItemPointerGetBlockNumber(tid),
                             ItemPointerGetOffsetNumber(tid),
                             DatumGetFloat8(values[2]));
                    }
                    
                    // 释放 buffer（如果分配了）
                    if (BufferIsValid(buffer)) {
                        ReleaseBuffer(buffer);
                    }
                } else {
                    // 元组不存在或不可见
                    tuple_nulls[1] = true;
                    elog(LOG, "batch_vector_search_c: 返回结果 %d - query_id=%d, vector_id=null (tid: %u,%u 元组不存在), distance=%.6f", 
                         state->current_result, DatumGetInt32(values[0]),
                         ItemPointerGetBlockNumber(tid),
                         ItemPointerGetOffsetNumber(tid),
                         DatumGetFloat8(values[2]));
                }
            } else {
                // TID 无效
                tuple_nulls[1] = true;
                elog(LOG, "batch_vector_search_c: 返回结果 %d - query_id=%d, vector_id=null (TID无效), distance=%.6f", 
                     state->current_result, DatumGetInt32(values[0]), DatumGetFloat8(values[2]));
            }
        }
    } else {
        // 处理null结果
        values[0] = (Datum)0;
        values[1] = (Datum)0;
        values[2] = (Datum)0;
        tuple_nulls[0] = true;
        tuple_nulls[1] = true;
        tuple_nulls[2] = true;
    }
    
    state->current_result++;
    
    tuple = heap_form_tuple(funcctx->tuple_desc, values, tuple_nulls);
    SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
}

/* ScanKeyBatchFree 已删除 - 原因如下：
 * 1. ScanKeyBatch 结构体在 SRF 内存上下文中分配，SRF 结束时自动清理
 * 2. ScanKeyBatch 只包含指针，不分配额外内存：
 *    - first_vector_ptr 指向 PostgreSQL 数组数据，不需要释放
 *    - 其他字段都是基本类型
 * 3. 手动 pfree 是多余的，PostgreSQL 的内存管理机制会自动处理
 */

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