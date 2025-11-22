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
        // elog(LOG, "batch_vector_search_c: 所有向量添加完成");
        
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
        // elog(LOG, "batch_vector_search_c: 索引扫描创建完成 - scan=%p, scan->opaque=%p, scan->indexRelation=%p", 
        //      scan, scan ? scan->opaque : NULL, scan ? scan->indexRelation : NULL);
        
        // 一次性获取所有查询的所有结果
        max_results = k * n_querys; // 每个查询k个结果 × 查询数量
        max_values = max_results * 3; // 每个结果有3个字段
        result_values = palloc(max_values * sizeof(Datum));
        result_nulls = palloc(max_values * sizeof(bool));
        returned_tuples = 0;
        
        // elog(LOG, "batch_vector_search_c: 开始批量处理，预期最大结果数: %d", max_results);
        // elog(LOG, "batch_vector_search_c: 准备调用ivfflatbatchgettuple - scan=%p, result_values=%p, result_nulls=%p, max_values=%d, k=%d", 
        //      scan, result_values, result_nulls, max_values, k);
        
        // 调用批量扫描函数，一次性处理所有查询
        bool gettuple_result = ivfflatbatchgettuple(scan, ForwardScanDirection, 
                            result_values, result_nulls, 
                            max_values, &returned_tuples, k);
        
        // elog(LOG, "batch_vector_search_c: ivfflatbatchgettuple返回 - result=%d, returned_tuples=%d", gettuple_result, returned_tuples);
        // elog(LOG, "batch_vector_search_c: 批量处理完成，返回结果数: %d", returned_tuples);
        
        // 检查返回的结果是否有效
        if (returned_tuples == 0) {
            elog(WARNING, "batch_vector_search_c: 没有返回任何结果");
        }
        
        // 获取堆表 Relation（从索引中获取）
        Relation heap_rel;
        bool heap_rel_opened_by_us = false;
        if (scan->heapRelation) {
            // 使用索引扫描中的 heapRelation（不需要我们关闭）
            heap_rel = scan->heapRelation;
        } else {
            // 如果 scan 中没有 heapRelation，从索引元数据中获取
            Oid heap_oid = IndexGetRelation(index_oid, false);
            if (!OidIsValid(heap_oid)) {
                elog(ERROR, "batch_vector_search_c: 无法获取堆表 OID");
            }
            heap_rel = table_open(heap_oid, AccessShareLock);
            heap_rel_opened_by_us = true;  // 标记为我们打开的
        }
        
        // 获取 vector_id 字段的属性编号（假设是第1列，通常是主键或id字段）
        int vector_id_attnum;
        {
            int attnum;
            
            // 使用 get_attnum 通过关系ID和属性名获取属性编号
            attnum = get_attnum(RelationGetRelid(heap_rel), "id");
            if (attnum == InvalidAttrNumber) {
                // 如果找不到 "id" 字段，尝试使用第1列
                vector_id_attnum = 1;
                // elog(LOG, "batch_vector_search_c: 未找到 'id' 字段，使用第1列作为 vector_id");
            } else {
                vector_id_attnum = attnum;
                // elog(LOG, "batch_vector_search_c: 找到 'id' 字段，属性编号 = %d", vector_id_attnum);
            }
        }
        
        // 在第一次调用时，将所有TID转换为实际的vector_id值
        // 这样在SRF的后续调用中，我们只需要从result_values中读取数据
        // elog(LOG, "batch_vector_search_c: 开始转换TID到vector_id，共 %d 个结果", returned_tuples);
        Snapshot snapshot = GetActiveSnapshot();
        int converted_count = 0;
        for (int i = 0; i < returned_tuples; i++) {
            int base_idx = i * 3;
            if (!result_nulls[base_idx + 1]) {
                // values[1] 是 ItemPointer (TID)，需要从堆表中获取实际的 vector_id 值
                // 注意：GetBatchResults返回的是PointerGetDatum(&buffer->vector_ids[i])
                // 所以result_values[base_idx + 1]是一个指向ItemPointerData的指针
                ItemPointer tid = (ItemPointer)DatumGetPointer(result_values[base_idx + 1]);
                
                // 检查TID指针是否有效
                if (tid != NULL && ItemPointerIsValid(tid)) {
                    converted_count++;
                    HeapTupleData heap_tuple_data;
                    HeapTuple     heap_tuple = &heap_tuple_data;
                    bool          isnull;
                    Datum         vector_id_datum;
                    Buffer        buffer = InvalidBuffer;
                    bool          found;
                    ItemPointerData tid_copy;
                    
                    // 复制TID数据（因为tid可能指向tmpCtx中的数据，可能被释放）
                    tid_copy = *tid;
                    
                    // 初始化 heap_tuple 的 t_self 为 TID
                    heap_tuple->t_self = tid_copy;
                    
                    // 从堆表中获取元组
                    found = heap_fetch(heap_rel, snapshot, heap_tuple, &buffer, false);
                    
                    if (found && HeapTupleIsValid(heap_tuple)) {
                        // 从元组中获取 vector_id 字段的值
                        vector_id_datum = heap_getattr(heap_tuple, 
                                                       vector_id_attnum,
                                                       RelationGetDescr(heap_rel),
                                                       &isnull);
                        
                        if (!isnull) {
                            // 将 Datum 转换为 int32 并存储（在SRF内存上下文中）
                            // 注意：result_values在SRF内存上下文中，所以这里存储的值是安全的
                            result_values[base_idx + 1] = vector_id_datum;
                            result_nulls[base_idx + 1] = false;
                        } else {
                            // vector_id 列为 NULL
                            result_nulls[base_idx + 1] = true;
                        }
                        
                        // 释放 buffer（如果分配了）
                        if (BufferIsValid(buffer)) {
                            ReleaseBuffer(buffer);
                        }
                    } else {
                        // 元组不存在或不可见
                        result_nulls[base_idx + 1] = true;
                    }
                } else {
                    // TID 无效
                    result_nulls[base_idx + 1] = true;
                }
            }
        }
        // elog(LOG, "batch_vector_search_c: TID转换完成，成功转换 %d 个TID", converted_count);
        
        // 关闭堆表（如果是我们打开的）
        if (heap_rel_opened_by_us) {
            table_close(heap_rel, AccessShareLock);
        }
        
        // 保存结果数据到状态中
        state->result_values = result_values;
        state->result_nulls = result_nulls;
        state->returned_tuples = returned_tuples;
        state->current_result = 0;
        state->k = k;
        state->n_querys = n_querys;
        
        // 保存资源以便后续清理
        state->scan = scan;
        state->index = NULL;  // 不保存index引用，让PostgreSQL通过scan管理
        state->batch_keys = batch_keys;
        state->heap_rel = NULL;  // 已经关闭，不需要保存
        state->heap_rel_opened_by_us = false;
        
        // 设置返回元组描述符
        if (get_call_result_type(fcinfo, NULL, &funcctx->tuple_desc) != TYPEFUNC_COMPOSITE)
            elog(ERROR, "return type must be a row type");
        funcctx->tuple_desc = BlessTupleDesc(funcctx->tuple_desc);
        
        // elog(LOG, "batch_vector_search_c: 初始化完成，准备返回 %d 个结果", returned_tuples);
    }
    
    // 每次调用返回一个结果
    funcctx = SRF_PERCALL_SETUP();
    state = (BatchSearchState *)funcctx->user_fctx;
    
    // elog(LOG, "batch_vector_search_c: SRF后续调用 - current_result=%d, returned_tuples=%d", 
    //      state ? state->current_result : -1, state ? state->returned_tuples : -1);
    
    if (!state) {
        elog(ERROR, "batch_vector_search_c: state为NULL");
        SRF_RETURN_DONE(funcctx);
    }
    
    if (state->current_result >= state->returned_tuples) {
        // elog(LOG, "batch_vector_search_c: 所有结果已返回，开始清理资源");
        
        // 清理 PostgreSQL 资源
        // 最简化策略：让PostgreSQL的SRF机制自动处理所有资源清理
        
        // 重要发现：
        // 1. scan结构和scan->opaque都在SRF内存上下文中
        // 2. PostgreSQL会在SRF结束时自动调用清理函数
        // 3. 手动调用清理函数可能导致双重清理问题
        // 4. 最安全的方式是完全依赖PostgreSQL的自动清理机制
        
        // elog(LOG, "batch_vector_search_c: 依赖PostgreSQL自动清理资源 - scan=%p, index=%p", 
        //      state->scan, state->index);
        
        // heap_rel 已经在第一次调用时关闭，不需要再次关闭
        /* batch_keys 由 SRF 内存上下文自动清理 */
        
        // 在返回DONE之前，确保所有指针都已清空
        // 这样即使PostgreSQL在清理SRF内存上下文时访问state，也不会访问已删除的内存
        state->result_values = NULL;
        state->result_nulls = NULL;
        state->batch_keys = NULL;
        
        // 重要：确保所有可能被PostgreSQL访问的字段都已清空
        // scan和index已经设置为NULL，heap_rel已经在第一次调用时关闭
        // 但是，我们需要确保state结构本身是安全的
        
        // elog(LOG, "batch_vector_search_c: 资源清理完成，返回DONE - scan=%p, index=%p", 
        //      state->scan, state->index);
        
        // 在返回DONE之前，确认所有指针状态
        // 这可以防止PostgreSQL在清理SRF内存上下文时访问已删除的内存
        
        SRF_RETURN_DONE(funcctx);
    }
    
    // 返回当前结果
    // 注意：TID已经在第一次调用时转换为实际的vector_id值，所以这里直接读取即可
    int base_idx = state->current_result * 3;
    
    if (base_idx + 2 >= state->returned_tuples * 3) {
        elog(ERROR, "batch_vector_search_c: 索引越界 - base_idx=%d, returned_tuples=%d", 
             base_idx, state->returned_tuples);
    }
    
    values[0] = state->result_values[base_idx + 0]; // query_id
    values[1] = state->result_values[base_idx + 1]; // vector_id (已经是实际的int32值)
    values[2] = state->result_values[base_idx + 2]; // distance
    
    tuple_nulls[0] = state->result_nulls[base_idx + 0];
    tuple_nulls[1] = state->result_nulls[base_idx + 1];
    tuple_nulls[2] = state->result_nulls[base_idx + 2];
    
    // elog(LOG, "batch_vector_search_c: 返回结果 %d - query_id=%d, vector_id=%d, distance=%.6f", 
    //      state->current_result,
    //      DatumGetInt32(values[0]),
    //      DatumGetInt32(values[1]),
    //      DatumGetFloat8(values[2]));
    
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