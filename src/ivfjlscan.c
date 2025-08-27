#include "postgres.h"

#include <float.h>

#include "access/relscan.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "lib/pairingheap.h"
#include "ivfflat.h"
#include "ivfjl.h"
#include "ivfscan.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"

/*
 * Prepare for an index scan
 */
IndexScanDesc
ivfflatbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	IvfflatScanOpaque so;
	int			lists;
	int			dimensions;
	int			probes = ivfflat_probes;
	int			maxProbes;
	MemoryContext oldCtx;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	/* Get lists and dimensions from metapage */
	IvfflatGetMetaPageInfo(index, &lists, &dimensions);

	if (ivfflat_iterative_scan != IVFFLAT_ITERATIVE_SCAN_OFF)
		maxProbes = Max(ivfflat_max_probes, probes);
	else
		maxProbes = probes;

	if (probes > lists)
		probes = lists;

	if (maxProbes > lists)
		maxProbes = lists;

	so = (IvfflatScanOpaque) palloc(sizeof(IvfflatScanOpaqueData));
	so->typeInfo = IvfflatGetTypeInfo(index);
	so->first = true;
	so->probes = probes;
	so->maxProbes = maxProbes;
	so->dimensions = dimensions;

	/* Set support functions */
	so->procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
	so->normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
	so->collation = index->rd_indcollation[0];

	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Ivfflat scan temporary context",
									   ALLOCSET_DEFAULT_SIZES);

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/* Create tuple description for sorting */
	so->tupdesc = CreateTemplateTupleDesc(2);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 1, "distance", FLOAT8OID, -1, 0);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 2, "heaptid", TIDOID, -1, 0);

	/* Prep sort */
	so->sortstate = InitScanSortState(so->tupdesc);

	/* Need separate slots for puttuple and gettuple */
	so->vslot = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsVirtual);
	so->mslot = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsMinimalTuple);

	/*
	 * Reuse same set of shared buffers for scan
	 *
	 * See postgres/src/backend/storage/buffer/README for description
	 */
	so->bas = GetAccessStrategy(BAS_BULKREAD);

	so->listQueue = pairingheap_allocate(CompareLists, scan);
	so->listPages = palloc(maxProbes * sizeof(BlockNumber));
	so->listIndex = 0;
	so->lists = palloc(maxProbes * sizeof(IvfflatScanList));

	MemoryContextSwitchTo(oldCtx);

	scan->opaque = so;

	return scan;
}

/*
 * Start or restart an index scan
 */
void
ivfflatrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;

	so->first = true;
	pairingheap_reset(so->listQueue);
	so->listIndex = 0;

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
bool
ivfflatgettuple(IndexScanDesc scan, ScanDirection dir)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	ItemPointer heaptid;
	bool		isnull;

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan ivfflat index without order");

		/* Requires MVCC-compliant snapshot as not able to pin during sorting */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with ivfflat");

		value = GetScanValue(scan);
		IvfflatBench("GetScanLists", GetScanLists(scan, value));
		IvfflatBench("GetScanItems", GetScanItems(scan, value));
		so->first = false;
		so->value = value;
	}

	while (!tuplesort_gettupleslot(so->sortstate, true, false, so->mslot, NULL))
	{
		if (so->listIndex == so->maxProbes)
			return false;

		IvfflatBench("GetScanItems", GetScanItems(scan, so->value));
	}

	heaptid = (ItemPointer) DatumGetPointer(slot_getattr(so->mslot, 2, &isnull));

	scan->xs_heaptid = *heaptid;
	scan->xs_recheck = false;
	scan->xs_recheckorderby = false;
	return true;
}

/*
 * End a scan and release resources
 */
void
ivfflatendscan(IndexScanDesc scan)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;

	/* Free any temporary files */
	tuplesort_end(so->sortstate);

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}

IndexScanDesc ivfjlbeginscan(Relation index, int nkeys, int norderbys) {
    IndexScanDesc scan;
    IvfjlScanOpaque so;
    int lists, dimensions;
    int probes = ivfflat_probes;
    int maxProbes;
    MemoryContext oldCtx;
	Buffer metaBuf;
	Page metaPage;

    scan = RelationGetIndexScan(index, nkeys, norderbys);
    
    /*获取元数据信息*/ 
    IvfflatGetMetaPageInfo(index, &lists, &dimensions);
    
    /*计算扫描参数*/ 
    if (ivfflat_iterative_scan != IVFFLAT_ITERATIVE_SCAN_OFF)
        maxProbes = Max(ivfflat_max_probes, probes);
    else
        maxProbes = probes;
    
    if (probes > lists) probes = lists;
    if (maxProbes > lists) maxProbes = lists;
    
    /*分配ivfjl scan opaque*/ 
    so = (IvfjlScanOpaque) palloc0(sizeof(IvfjlScanOpaqueData));
    
    /*初始化基础字段*/ 
    so->base.typeInfo = IvfflatGetTypeInfo(index);
    so->base.first = true;
    so->base.probes = probes;
    so->base.maxProbes = maxProbes;
    so->base.dimensions = dimensions;
    
    /*设置支持函数*/ 
    so->base.procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
    so->base.normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
    so->base.collation = index->rd_indcollation[0];
    
    /*创建临时内存上下文*/ 
    so->base.tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
                                           "Ivfjl scan temporary context",
                                           ALLOCSET_DEFAULT_SIZES);
    
    oldCtx = MemoryContextSwitchTo(so->base.tmpCtx);
    
    /*创建元组描述符*/ 
    so->base.tupdesc = CreateTemplateTupleDesc(2);
    TupleDescInitEntry(so->base.tupdesc, (AttrNumber) 1, "distance", FLOAT8OID, -1, 0);
    TupleDescInitEntry(so->base.tupdesc, (AttrNumber) 2, "heaptid", TIDOID, -1, 0);
    
    /* 初始化排序状态 */
    so->base.sortstate = InitScanSortState(so->base.tupdesc);
    
    /* 创建元组槽 */
    so->base.vslot = MakeSingleTupleTableSlot(so->base.tupdesc, &TTSOpsVirtual);
    so->base.mslot = MakeSingleTupleTableSlot(so->base.tupdesc, &TTSOpsMinimalTuple);
    
    /* 设置缓冲区访问策略 */
    so->base.bas = GetAccessStrategy(BAS_BULKREAD);
    
    /* 初始化列表管理 */
    so->base.listQueue = pairingheap_allocate(CompareLists, scan);
    so->base.listPages = palloc(maxProbes * sizeof(BlockNumber));
    so->base.listIndex = 0;
    so->base.lists = palloc(maxProbes * sizeof(IvfflatScanList));
    
    MemoryContextSwitchTo(oldCtx);
    
    /* 读取JL投影矩阵 */
    metaBuf = ReadBuffer(index, IVFFLAT_METAPAGE_BLKNO);
    if (!BufferIsValid(metaBuf)) {
        elog(ERROR, "failed to read metapage");
    }
    
    LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
    metaPage = BufferGetPage(metaBuf);
    ReadJLFromMetaPage(metaPage, &so->jlProj, CurrentMemoryContext);
    UnlockReleaseBuffer(metaBuf);
    
    /* 设置JL相关参数 */
    so->jlDimensions = so->jlProj.reduced_dim;
    so->reorder = ivfjl_enable_reorder;
    so->reorderCandidates = ivfjl_reorder_candidates;
    
    scan->opaque = so;
    return scan;
}

/*
 * Start or restart an IVFJL index scan
 */
void
ivfjlrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
    IvfjlScanOpaque so = (IvfjlScanOpaque) scan->opaque;
    
    IvfflatScanOpaque temp_so = (IvfflatScanOpaque) &so->base;
    scan->opaque = temp_so;
    ivfflatrescan(scan, keys, nkeys, orderbys, norderbys);
    
    scan->opaque = so;
    
    /*TODO: 可能需要重置JL相关的状态*/

    /*TODO: 可能需要重置重排序相关的状态*/
}

/*
 * End an IVFJL scan and release resources
 */
void
ivfjlendscan(IndexScanDesc scan)
{
    IvfjlScanOpaque so = (IvfjlScanOpaque) scan->opaque;
    
    /* 释放排序状态 */
    if (so->base.sortstate != NULL)
    {
        tuplesort_end(so->base.sortstate);
        so->base.sortstate = NULL;
    }
    
    /* 释放内存上下文 */
    if (so->base.tmpCtx != NULL)
    {
        MemoryContextDelete(so->base.tmpCtx);
        so->base.tmpCtx = NULL;
    }
    
    /* 释放JL投影矩阵 */
    if (so->jlProj.matrix != NULL)
    {
        FreeJLProjection(&so->jlProj);
    }
    
    /* 释放ivfjl的scan opaque */
    pfree(so);
    scan->opaque = NULL;
}

/*
 * Get IVFJL scan lists and sort by distance using projected vectors
 */
static void
IvfjlGetScanLists(IndexScanDesc scan, Datum projectedValue)
{
    IvfjlScanOpaque so = (IvfjlScanOpaque) scan->opaque;
    BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
    int         listCount = 0;
    double      maxDistance = DBL_MAX;

    /* Search all list pages */
    while (BlockNumberIsValid(nextblkno))
    {
        Buffer      cbuf;
        Page        cpage;
        OffsetNumber maxoffno;

		/*读取聚类列表页面*/
        cbuf = ReadBuffer(scan->indexRelation, nextblkno);
        LockBuffer(cbuf, BUFFER_LOCK_SHARE);
        cpage = BufferGetPage(cbuf);

        maxoffno = PageGetMaxOffsetNumber(cpage);

		/*遍历页面中的每个聚类*/
        for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
        {
            IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
            double      distance;

			/*获取聚类中心*/ 
            /* Use original vector for distance calculation */
            distance = DatumGetFloat8(so->base.distfunc(so->base.procinfo, so->base.collation, 
                                                       PointerGetDatum(&list->center), projectedValue));

            if (listCount < so->base.maxProbes)
            {
                IvfflatScanList *scanlist;

                scanlist = &so->base.lists[listCount];
                scanlist->startPage = list->startPage;
                scanlist->distance = distance;
                listCount++;

                /* Add to heap */
                pairingheap_add(so->base.listQueue, &scanlist->ph_node);

                /* Calculate max distance */
                if (listCount == so->base.maxProbes)
                    maxDistance = GetScanList(pairingheap_first(so->base.listQueue))->distance;
            }
            else if (distance < maxDistance)
            {
                IvfflatScanList *scanlist;

                /* Remove */
                scanlist = GetScanList(pairingheap_remove_first(so->base.listQueue));

                /* Reuse */
                scanlist->startPage = list->startPage;
                scanlist->distance = distance;
                pairingheap_add(so->base.listQueue, &scanlist->ph_node);

                /* Update max distance */
                maxDistance = GetScanList(pairingheap_first(so->base.listQueue))->distance;
            }
        }

        nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;

        UnlockReleaseBuffer(cbuf);
    }

    for (int i = listCount - 1; i >= 0; i--)
        so->base.listPages[i] = GetScanList(pairingheap_remove_first(so->base.listQueue))->startPage;

    Assert(pairingheap_is_empty(so->base.listQueue));
}

/*
 * Get IVFJL scan items using projected vectors with optional reordering
 */
static void
IvfjlGetScanItems(IndexScanDesc scan, Datum projectedValue, Datum originalValue)
{
    IvfjlScanOpaque so = (IvfjlScanOpaque) scan->opaque;
    TupleDesc   tupdesc = RelationGetDescr(scan->indexRelation);
    TupleTableSlot *slot = so->base.vslot;
    int         batchProbes = 0;

    tuplesort_reset(so->base.sortstate);

    /* Search closest probes lists */
    while (so->base.listIndex < so->base.maxProbes && (++batchProbes) <= so->base.probes)
    {
        BlockNumber searchPage = so->base.listPages[so->base.listIndex++];

        /* Search all entry pages for list */
        while (BlockNumberIsValid(searchPage))
        {
            Buffer      buf;
            Page        page;
            OffsetNumber maxoffno;

            buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, searchPage, RBM_NORMAL, so->base.bas);
            LockBuffer(buf, BUFFER_LOCK_SHARE);
            page = BufferGetPage(buf);
            maxoffno = PageGetMaxOffsetNumber(page);

            for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
            {
                IndexTuple  itup;
                Datum       datum;
                bool        isnull;
                ItemId      itemid = PageGetItemId(page, offno);
                double      distance;

                itup = (IndexTuple) PageGetItem(page, itemid);
                datum = index_getattr(itup, 1, tupdesc, &isnull);

				/* Use projected vector for approximate distance calculation */
				distance = DatumGetFloat8(so->base.distfunc(so->base.procinfo, so->base.collation, datum, projectedValue));

                /*
                 * Add virtual tuple
                 */
                ExecClearTuple(slot);
                slot->tts_values[0] = Float8GetDatum(distance);
                slot->tts_isnull[0] = false;
                slot->tts_values[1] = PointerGetDatum(&itup->t_tid);
                slot->tts_isnull[1] = false;
                ExecStoreVirtualTuple(slot);

                tuplesort_puttupleslot(so->base.sortstate, slot);
            }

            searchPage = IvfflatPageGetOpaque(page)->nextblkno;

            UnlockReleaseBuffer(buf);
        }
    }

    tuplesort_performsort(so->base.sortstate);
}
/*
 * Fetch the next tuple in the given IVFJL scan
 */
bool
ivfjlgettuple(IndexScanDesc scan, ScanDirection dir)
{
    IvfjlScanOpaque so = (IvfjlScanOpaque)scan->opaque;
    ItemPointer heaptid;
    bool        isnull;

    /*
     * Index can be used to scan backward, but Postgres doesn't support
     * backward scan on operators
     */
    Assert(ScanDirectionIsForward(dir));

    if (so->base.first)/*是第一次调用*/
    {
        Datum       value;
		float		*originalX;
		float		*projectedX;

        /* Count index scan for stats */
        pgstat_count_index_scan(scan->indexRelation);

        /* Safety check */
        if (scan->orderByData == NULL)
            elog(ERROR, "cannot scan ivfjl index without order");

		/* Requires MVCC-compliant snapshot as not able to pin during sorting */
		/* https://www.postgresql.org/docs/current/index-locking.html */
        if (!IsMVCCSnapshot(scan->xs_snapshot))
            elog(ERROR, "non-MVCC snapshots are not supported with ivfjl");

        /* Get original query vector */
        value = GetScanValue(scan);
        originalX = (float *) DatumGetPointer(value);
        projectedX = (float *) palloc(so->jlDimensions * sizeof(float));
        
        /* Perform JL projection */
        JLProjectVector(&so->jlProj, originalX, projectedX);
        
        /* Store projected vector for later use */
        so->jlValue = PointerGetDatum(&projectedX);

		/*这两行是用来测试代码执行时间的*/ 
		IvfflatBench("IvfjlGetScanLists", IvfjlGetScanLists(scan, so->jlValue));
		IvfflatBench("IvfjlGetScanItems", IvfjlGetScanItems(scan, so->jlValue, value));/* 传入原始向量用于重排序 */
        
        so->base.first = false;
        so->base.value = value;
    }

    while (!tuplesort_gettupleslot(so->base.sortstate, true, false, so->base.mslot, NULL))
    {
        if (so->base.listIndex == so->base.maxProbes)
            return false;

        IvfjlGetScanItems(scan, so->jlValue, so->base.value);
    }

    heaptid = (ItemPointer) DatumGetPointer(slot_getattr(so->base.mslot, 2, &isnull));

    scan->xs_heaptid = *heaptid;
    scan->xs_recheck = false;
    scan->xs_recheckorderby = false;
    return true;
}
