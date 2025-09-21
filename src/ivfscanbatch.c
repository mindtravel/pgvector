/* ivfscanbatch.c */
#include "postgres.h"
#include "access/relscan.h"
#include "ivfflat.h"
#include "scanbatch.h"
#include "utils/memutils.h"

/*
 * 批量扫描状态结构
 */
typedef struct IvfflatBatchScanOpaqueData
{
    IvfflatScanOpaqueData base;  /* 基础扫描状态 */
    ScanKeyBatch    batch_keys;   /* 批量查询键 */
    int             current_key;  /* 当前处理的键索引 */
    Tuplesortstate** sortstates;  /* 每个查询的排序状态数组 */
    TupleTableSlot** vslots;      /* 每个查询的虚拟槽位数组 */
    TupleTableSlot** mslots;      /* 每个查询的最小元组槽位数组 */
} IvfflatBatchScanOpaqueData;

typedef IvfflatBatchScanOpaqueData* IvfflatBatchScanOpaque;

/*
 * 初始化批量扫描
 */
IndexScanDesc
ivfflatbatchbeginscan(Relation index, int nkeys, int norderbys, ScanKeyBatch batch_keys)
{
    IndexScanDesc scan;
    IvfflatBatchScanOpaque so;
    MemoryContext oldCtx;
    int         lists, dimensions;

    /* 创建扫描描述符 */
    scan = RelationGetIndexScan(index, nkeys, norderbys);

    /* 获取列表数和维度数 */
    IvfflatGetMetaPageInfo(index, &lists, &dimensions);

    /* 分配扫描状态 */
    so = (IvfflatBatchScanOpaque)palloc0(sizeof(IvfflatBatchScanOpaqueData));

    /* 初始化基础扫描状态 */
    so->base.typeInfo = IvfflatGetTypeInfo(index);
    so->base.first = true;
    so->base.probes = ivfflat_probes;
    so->base.maxProbes = (ivfflat_iterative_scan != IVFFLAT_ITERATIVE_SCAN_OFF) ?
        Max(ivfflat_max_probes, ivfflat_probes) : ivfflat_probes;
    so->base.dimensions = dimensions;

    /* 设置支持函数 */
    so->base.procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
    so->base.normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
    so->base.collation = index->rd_indcollation[0];

    /* 创建内存上下文 */
    so->base.tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
        "Ivfflat batch scan temporary context",
        ALLOCSET_DEFAULT_SIZES);

    oldCtx = MemoryContextSwitchTo(so->base.tmpCtx);

    /* 存储批量键 */
    so->batch_keys = batch_keys;
    so->current_key = 0;

    /* 为每个查询创建排序状态和槽位 */
    int nbatch = batch_keys->nkeys;
    so->sortstates = palloc(nbatch * sizeof(Tuplesortstate*));
    so->vslots = palloc(nbatch * sizeof(TupleTableSlot*));
    so->mslots = palloc(nbatch * sizeof(TupleTableSlot*));

    /* 创建元组描述符 */
    so->base.tupdesc = CreateTemplateTupleDesc(2);
    TupleDescInitEntry(so->base.tupdesc, (AttrNumber)1, "distance", FLOAT8OID, -1, 0);
    TupleDescInitEntry(so->base.tupdesc, (AttrNumber)2, "heaptid", TIDOID, -1, 0);

    /* 初始化每个查询的状态 */
    for (int i = 0; i < nbatch; i++)
    {
        so->sortstates[i] = InitScanSortState(so->base.tupdesc);
        so->vslots[i] = MakeSingleTupleTableSlot(so->base.tupdesc, &TTSOpsVirtual);
        so->mslots[i] = MakeSingleTupleTableSlot(so->base.tupdesc, &TTSOpsMinimalTuple);
    }

    /* 设置缓冲区访问策略 */
    // so->base.bas = GetAccessStrategy(BAS_BULKREAD);

    /* 初始化列表队列 */
    so->base.listQueue = pairingheap_allocate(CompareLists, scan);
    so->base.listPages = palloc(so->base.maxProbes * sizeof(BlockNumber));
    so->base.listIndex = 0;
    so->base.lists = palloc(so->base.maxProbes * sizeof(IvfflatScanList));

    MemoryContextSwitchTo(oldCtx);

    scan->opaque = so;

    return scan;
}

/*
 * 批量扫描获取多个元组
 */
bool
ivfflatbatchgettuple(IndexScanDesc scan, ScanDirection dir, Datum* values, bool* isnull, int max_tuples, int* returned_tuples)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    int nbatch = so->batch_keys->nkeys;
    int count = 0;

    /* 确保向前扫描 */
    Assert(ScanDirectionIsForward(dir));

    /* 初始化返回数组 */
    for (int i = 0; i < max_tuples; i++)
    {
        values[i] = (Datum)0;
        isnull[i] = true;
    }

    /* 处理每个查询 */
    for (int i = 0; i < nbatch && count < max_tuples; i++)
    {
        /* 获取当前查询的排序状态 */
        Tuplesortstate* sortstate = so->sortstates[i];
        TupleTableSlot* mslot = so->mslots[i];
        ItemPointer heaptid;
        bool        isnull_local;

        /* 如果还没有处理这个查询，先进行处理 */
        if (so->current_key <= i)
        {
            Datum value = ScanKeyBatchGetVector(so->batch_keys, i);

            /* 处理当前查询 */
            GetScanLists(scan, value);
            GetScanItems(scan, value);

            so->current_key = i + 1;
        }

        /* 从排序状态获取元组 */
        if (tuplesort_gettupleslot(sortstate, true, false, mslot, NULL))
        {
            heaptid = (ItemPointer)DatumGetPointer(slot_getattr(mslot, 2, &isnull_local));

            /* 存储结果 */
            values[count] = PointerGetDatum(heaptid);
            isnull[count] = false;
            count++;
        }
    }

    *returned_tuples = count;
    return (count > 0);
}

/*
 * 结束批量扫描
 */
void
ivfflatbatchendscan(IndexScanDesc scan)
{
    IvfflatBatchScanOpaque so = (IvfflatBatchScanOpaque)scan->opaque;
    int nbatch = so->batch_keys->nkeys;

    /* 释放每个查询的排序状态和槽位 */
    for (int i = 0; i < nbatch; i++)
    {
        tuplesort_end(so->sortstates[i]);
        ExecDropSingleTupleTableSlot(so->vslots[i]);
        ExecDropSingleTupleTableSlot(so->mslots[i]);
    }

    /* 释放数组 */
    pfree(so->sortstates);
    pfree(so->vslots);
    pfree(so->mslots);

    /* 释放基础资源 */
    pairingheap_free(so->base.listQueue);
    pfree(so->base.listPages);
    pfree(so->base.lists);

    MemoryContextDelete(so->base.tmpCtx);

    pfree(so);
    scan->opaque = NULL;
}