/* test_batch_scan.c */
#include "postgres.h"
#include "scanbatch.h"
#include "ivfflat.h"

void
test_batch_scan(void)
{
    /* 创建批量键 */
    ScanKeyBatch batch = ScanKeyBatchCreate(10, 128);

    /* 添加一些测试向量 */
    for (int i = 0; i < 10; i++)
    {
        Vector* vec = InitVector(128);
        for (int j = 0; j < 128; j++)
        {
            vec->x[j] = (float)rand() / RAND_MAX;
        }
        ScanKeyBatchAddVector(batch, i, PointerGetDatum(vec));
        pfree(vec);
    }

    /* 检查数据是否连续 */
    if (ScanKeyBatchIsContinuous(batch))
    {
        elog(INFO, "Batch data is continuous and ready for GPU processing");
    }

    /* 清理 */
    ScanKeyBatchFree(batch);
}