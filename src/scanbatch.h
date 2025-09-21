/* scanbatch.h */
#ifndef SCANBATCH_H
#define SCANBATCH_H

#include "postgres.h"
#include "access/skey.h"
#include "vector.h"

/* 批量扫描键数据结构 */
typedef struct ScanKeyBatchData
{
    int         nkeys;          /* 批量中的键数量 */
    int         keySize;        /* 每个键的大小（字节）*/
    ScanKey     keys;           /* 指向第一个ScanKey的指针 */
    void* batch_data;     /* 连续存储的批量查询向量数据 */
    int         vec_dim;        /* 向量维度 */
    Size        vec_size;       /* 每个向量的大小 */
    bool        data_continuous;
} ScanKeyBatchData;

typedef ScanKeyBatchData* ScanKeyBatch;

/* 函数声明 */
extern ScanKeyBatch ScanKeyBatchCreate(int nkeys, int vec_dim);
extern void ScanKeyBatchFree(ScanKeyBatch batch);
extern void ScanKeyBatchAddVector(ScanKeyBatch batch, int index, Datum vector);
extern Datum ScanKeyBatchGetVector(ScanKeyBatch batch, int index);
extern bool ScanKeyBatchIsContinuous(ScanKeyBatch batch);
extern void* ScanKeyBatchGetContinuousData(ScanKeyBatch batch);

#endif /* SCANBATCH_H */