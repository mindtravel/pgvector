/* scanbatch.c */
#include "postgres.h"
#include "access/skey.h"
#include "utils/datum.h"
#include "scanbatch.h"
#include "vector.h"

/*
 * 创建新的ScankeyBatch
 */
ScanKeyBatch
ScanKeyBatchCreate(int nkeys, int vec_dim)
{
    ScanKeyBatch batch;
    Size        size;
    Size        vec_size;

    /* 计算向量大小 */
    vec_size = VECTOR_SIZE(vec_dim);

    /* 分配ScankeyBatch结构 */
    batch = (ScanKeyBatch)palloc0(sizeof(ScanKeyBatchData));
    batch->nkeys = nkeys;
    batch->vec_dim = vec_dim;
    batch->vec_size = vec_size;

    /* 分配ScanKey数组 */
    batch->keys = (ScanKey)palloc(nkeys * sizeof(ScanKeyData));

    /* 分配连续内存存储所有向量数据 */
    size = nkeys * vec_size;
    batch->batch_data = palloc(size);
    batch->data_contiguous = true;
    batch->keySize = sizeof(ScanKeyData);

    /* 初始化每个ScanKey */
    for (int i = 0; i < nkeys; i++)
    {
        ScanKey     key = &batch->keys[i];
        Pointer     vec_data = (Pointer)batch->batch_data + i * vec_size;

        /* 初始化ScanKey */
        key->sk_flags = 0;
        key->sk_attno = 1;
        key->sk_strategy = 0;
        key->sk_subtype = 0;
        key->sk_collation = 0;
        key->sk_func.fn_oid = InvalidOid;
        key->sk_argument = PointerGetDatum(vec_data);
    }

    return batch;
}

/*
 * 释放ScankeyBatch
 */
void
ScanKeyBatchFree(ScanKeyBatch batch)
{
    if (batch == NULL)
        return;

    if (batch->keys != NULL)
        pfree(batch->keys);

    if (batch->batch_data != NULL)
        pfree(batch->batch_data);

    pfree(batch);
}

/*
 * 向ScankeyBatch添加向量
 */
void
ScanKeyBatchAddVector(ScanKeyBatch batch, int index, Datum vector)
{
    if (index < 0 || index >= batch->nkeys)
        elog(ERROR, "index out of bounds in ScanKeyBatchAddVector");

    /* 获取目标内存位置 */
    Pointer     target = (Pointer)batch->batch_data + index * batch->vec_size;
    Vector* src_vec = DatumGetVector(vector);

    /* 检查维度是否匹配 */
    if (src_vec->dim != batch->vec_dim)
        elog(ERROR, "vector dimension mismatch in ScanKeyBatchAddVector");

    /* 复制向量数据到连续内存 */
    memcpy(target, src_vec, batch->vec_size);
}

/*
 * 从ScankeyBatch获取向量
 */
Datum
ScanKeyBatchGetVector(ScanKeyBatch batch, int index)
{
    if (index < 0 || index >= batch->nkeys)
        elog(ERROR, "index out of bounds in ScanKeyBatchGetVector");

    Pointer     vec_data = (Pointer)batch->batch_data + index * batch->vec_size;
    return PointerGetDatum(vec_data);
}

/*
 * 检查数据是否连续存储
 */
bool
ScanKeyBatchIsContinuous(ScanKeyBatch batch)
{
    return batch->data_contiguous;
}

/*
 * 获取连续存储的数据指针
 */
void*
ScanKeyBatchGetContinuousData(ScanKeyBatch batch)
{
    return batch->batch_data;
}