/* scanbatch.h */
#ifndef SCANBATCH_H
#define SCANBATCH_H

#include "postgres.h"
#include "access/skey.h"
#include "utils/array.h"
#include "access/relscan.h"
#include "executor/tuptable.h"
#include "vector.h"

#define COSINE_DISTANCE 0
#define L2_DISTANCE 1

// 前向声明
struct IndexScanDescData;
typedef struct IndexScanDescData *IndexScanDesc;

/* 批量扫描键数据结构 */
typedef struct ScanKeyBatchData
{
    int         nkeys;          /* 批量中的向量数量 */
    int         vec_dim;        /* 向量维度 */
    Size        vec_size;       /* 每个向量的大小 */
    void*       first_vector_ptr; /* 指向第一个向量的指针 */
} ScanKeyBatchData;

typedef ScanKeyBatchData* ScanKeyBatch;

/* 函数声明 */
extern ScanKeyBatch ScanKeyBatchCreate(int nkeys, int vec_dim);
/* ScanKeyBatchFree 已删除 - 由 SRF 内存上下文自动管理 */
extern void ScanKeyBatchAddData(ScanKeyBatch batch, Datum *vectors, bool *nulls, int n_vectors);
extern Datum ScanKeyBatchGetVector(ScanKeyBatch batch, int index);
extern void* ScanKeyBatchGetContinuousData(ScanKeyBatch batch);

// 批量搜索状态结构
typedef struct BatchSearchState {
    Oid index_oid;
    ArrayType *query_vectors;
    int k;
    int n_querys;
    Datum *result_values;
    bool *result_nulls;
    int current_result;
    IndexScanDesc scan;
    Relation index;
    Relation heap_rel;          /* 堆表 Relation，用于从 TID 获取 vector_id */
    bool heap_rel_opened_by_us; /* 标记 heap_rel 是否由我们打开（需要关闭） */
    int vector_id_attnum;       /* vector_id 字段的属性编号 */
    ScanKeyBatch batch_keys;
} BatchSearchState;

#endif /* SCANBATCH_H */