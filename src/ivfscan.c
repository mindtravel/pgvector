#include "postgres.h"

#include <float.h>

#include "access/relscan.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "lib/pairingheap.h"
#include "ivfflat.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
// #include "ivfjl.h"

#ifdef USE_CUDA
#include "cuda/cuda_wrapper.h"
#endif

#define GetScanList(ptr) pairingheap_container(IvfflatScanList, ph_node, ptr)
#define GetScanListConst(ptr) pairingheap_const_container(IvfflatScanList, ph_node, ptr)


#ifdef USE_CUDA
static void GetScanLists_GPU(IndexScanDesc scan, Datum value) __attribute__((unused));
static int UploadCentersToGPU(IndexScanDesc scan);
#endif

/*
 * Compare list distances
 */
static int
CompareLists(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (GetScanListConst(a)->distance > GetScanListConst(b)->distance)
		return 1;

	if (GetScanListConst(a)->distance < GetScanListConst(b)->distance)
		return -1;

	return 0;
}

/*
 * Get lists and sort by distance
 */
static void
GetScanLists(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
	int			listCount = 0;
	double		maxDistance = DBL_MAX;

	/* 
	 * 单次查询使用CPU版本的probe选择以确保与批量查询的一致性
	 * 批量查询使用GPU加速，单次查询使用CPU计算，两者结果一致
	 */
#ifdef USE_CUDA
	// elog(LOG, "GetScanLists: 单次查询使用CPU版本的probe选择以确保结果一致性");
#endif

	/* Search all list pages */
	while (BlockNumberIsValid(nextblkno))
	{
		Buffer		cbuf;
		Page		cpage;
		OffsetNumber maxoffno;

		cbuf = ReadBuffer(scan->indexRelation, nextblkno);
		LockBuffer(cbuf, BUFFER_LOCK_SHARE);
		cpage = BufferGetPage(cbuf);

		maxoffno = PageGetMaxOffsetNumber(cpage);

		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
			double		distance;

			/* Use procinfo from the index instead of scan key for performance */
			distance = DatumGetFloat8(so->distfunc(so->procinfo, so->collation, PointerGetDatum(&list->center), value));

			if (listCount < so->maxProbes)
			{
				IvfflatScanList *scanlist;

				scanlist = &so->lists[listCount];
				scanlist->startPage = list->startPage;
				scanlist->distance = distance;
				listCount++;

				/* Add to heap */
				pairingheap_add(so->listQueue, &scanlist->ph_node);

				/* Calculate max distance */
				if (listCount == so->maxProbes)
					maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
			}
			else if (distance < maxDistance)
			{
				IvfflatScanList *scanlist;

				/* Remove */
				scanlist = GetScanList(pairingheap_remove_first(so->listQueue));

				/* Reuse */
				scanlist->startPage = list->startPage;
				scanlist->distance = distance;
				pairingheap_add(so->listQueue, &scanlist->ph_node);

				/* Update max distance */
				maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
			}
		}

		nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;

		UnlockReleaseBuffer(cbuf);
	}

	for (int i = listCount - 1; i >= 0; i--)
		so->listPages[i] = GetScanList(pairingheap_remove_first(so->listQueue))->startPage;

	Assert(pairingheap_is_empty(so->listQueue));
}

/*
 * GPU加速版本的聚类中心距离计算
 */
#ifdef USE_CUDA
static void GetScanLists_GPU(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so;
	BlockNumber nextblkno;
	int			totalLists;
	int			listCount;
	int			center_idx;
	BlockNumber *list_pages;
	float *query_data;
	
	so = (IvfflatScanOpaque) scan->opaque;
	nextblkno = IVFFLAT_HEAD_BLKNO;
	totalLists = 0;
	listCount = 0;
	center_idx = 0;
	
	/* 检查聚类中心是否已上传到GPU */
	if (!so->centers_uploaded) {
		elog(INFO, "聚类中心数据未上传到GPU，回退到CPU计算");
		GetScanLists(scan, value);
		return;
	}
	
	/* 计算总列表数量并收集列表页面信息 */
	
	list_pages = NULL;
	
	while (BlockNumberIsValid(nextblkno))
	{
		Buffer		cbuf;
		Page		cpage;
		OffsetNumber maxoffno;

		cbuf = ReadBuffer(scan->indexRelation, nextblkno);
		LockBuffer(cbuf, BUFFER_LOCK_SHARE);
		cpage = BufferGetPage(cbuf);
		maxoffno = PageGetMaxOffsetNumber(cpage);
		totalLists += maxoffno;
		nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;
		UnlockReleaseBuffer(cbuf);
	}
	
	/* 分配内存存储列表页面信息 */
	list_pages = palloc(totalLists * sizeof(BlockNumber));
	
	/* 收集列表页面信息 */
	nextblkno = IVFFLAT_HEAD_BLKNO;
	
	while (BlockNumberIsValid(nextblkno))
	{
		Buffer		cbuf;
		Page		cpage;
		OffsetNumber maxoffno;

		cbuf = ReadBuffer(scan->indexRelation, nextblkno);
		LockBuffer(cbuf, BUFFER_LOCK_SHARE);
		cpage = BufferGetPage(cbuf);
		maxoffno = PageGetMaxOffsetNumber(cpage);

		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
			list_pages[center_idx] = list->startPage;
			center_idx++;
		}

		nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;
		UnlockReleaseBuffer(cbuf);
	}
	
	/* 准备查询向量数据 */
	query_data = (float *) VARDATA(DatumGetPointer(value));
	
	/* 使用GPU计算距离 */
	if (cuda_compute_center_distances((CudaCenterSearchContext*)so->cuda_ctx, query_data, so->gpu_distances) == 0) {
		/* 使用GPU计算结果进行排序 */
		double maxDistance = DBL_MAX;
		
		for (int i = 0; i < totalLists; i++) {
			double distance = so->gpu_distances[i];
			
			if (listCount < so->maxProbes) {
				IvfflatScanList *scanlist = &so->lists[listCount];
				scanlist->startPage = list_pages[i];
				scanlist->distance = distance;
				listCount++;
				
				/* Add to heap */
				pairingheap_add(so->listQueue, &scanlist->ph_node);
				
				/* Calculate max distance */
				if (listCount == so->maxProbes)
					maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
			}
			else if (distance < maxDistance) {
				IvfflatScanList *scanlist = GetScanList(pairingheap_remove_first(so->listQueue));
				
				/* Reuse */
				scanlist->startPage = list_pages[i];
				scanlist->distance = distance;
				pairingheap_add(so->listQueue, &scanlist->ph_node);
				
				/* Update max distance */
				maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
			}
		}
	} else {
		/* GPU计算失败，回退到CPU计算 */
		elog(INFO, "GPU distance computation failed, falling back to CPU");
		GetScanLists(scan, value);
		pfree(list_pages);
		return;
	}
	
	/* 输出排序结果 */
	elog(LOG, "GetScanLists_GPU: 单次查询选择的probe列表:");
	for (int i = listCount - 1; i >= 0; i--) {
		IvfflatScanList *scanlist = GetScanList(pairingheap_remove_first(so->listQueue));
		so->listPages[i] = scanlist->startPage;
		elog(LOG, "  probe %d: 页面=%u, 距离=%.6f", i, scanlist->startPage, scanlist->distance);
	}

	Assert(pairingheap_is_empty(so->listQueue));
	
	/* 清理临时内存 */
	pfree(list_pages);
}
#endif

/*
 * 上传聚类中心数据到GPU（在初始化时调用）
 */
#ifdef USE_CUDA
static int UploadCentersToGPU(IndexScanDesc scan)
{
	IvfflatScanOpaque so;
	BlockNumber nextblkno;
	int			totalLists;
	int			center_idx;
	int			dimensions;
	float *centers_data;
	int upload_result;
	CudaCenterSearchContext* ctx;
	
	// elog(LOG, "UploadCentersToGPU: 函数开始执行");
	
	so = (IvfflatScanOpaque) scan->opaque;
	// elog(LOG, "UploadCentersToGPU: 获取扫描状态成功, so=%p", so);
	
	if (!so) {
		elog(ERROR, "UploadCentersToGPU: 扫描状态为空");
		return -1;
	}
	
	// elog(LOG, "UploadCentersToGPU: 维度=%d, cuda_ctx=%p, use_gpu=%s", 
	// 	 so->dimensions, so->cuda_ctx, so->use_gpu ? "是" : "否");
	
	totalLists = 0;
	center_idx = 0;
	dimensions = so->dimensions;
	nextblkno = IVFFLAT_HEAD_BLKNO;
	
	/* 检查CUDA上下文是否有效 */
	if (!so->cuda_ctx) {
		elog(ERROR, "CUDA上下文为空，无法上传聚类中心数据");
		return -1;
	}
	
	/* 如果已经上传过，直接返回成功 */
	if (so->centers_uploaded) {
		// elog(LOG, "聚类中心数据已上传，跳过重复上传");
		return 0;
	}
	
	// elog(LOG, "开始收集聚类中心数据 (维度: %d)", dimensions);
	
	/* 计算总列表数量 */
	while (BlockNumberIsValid(nextblkno))
	{
		Buffer		cbuf;
		Page		cpage;
		OffsetNumber maxoffno;

		cbuf = ReadBuffer(scan->indexRelation, nextblkno);
		LockBuffer(cbuf, BUFFER_LOCK_SHARE);
		cpage = BufferGetPage(cbuf);
		maxoffno = PageGetMaxOffsetNumber(cpage);
		totalLists += maxoffno;
		nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;
		UnlockReleaseBuffer(cbuf);
	}
	
	if (totalLists <= 0) {
		elog(ERROR, "没有找到聚类中心数据");
		return -1;
	}
	
	// elog(LOG, "找到 %d 个聚类中心，开始分配内存", totalLists);
	
	/* 分配内存存储聚类中心数据 */
	centers_data = palloc(totalLists * dimensions * sizeof(float));
	if (!centers_data) {
		elog(ERROR, "无法分配聚类中心数据内存 (%d个中心, %d维)", totalLists, dimensions);
		return -1;
	}
	
	/* 收集聚类中心数据 */
	nextblkno = IVFFLAT_HEAD_BLKNO;
	center_idx = 0;
	
	while (BlockNumberIsValid(nextblkno))
	{
		Buffer		cbuf;
		Page		cpage;
		OffsetNumber maxoffno;

		cbuf = ReadBuffer(scan->indexRelation, nextblkno);
		LockBuffer(cbuf, BUFFER_LOCK_SHARE);
		cpage = BufferGetPage(cbuf);
		maxoffno = PageGetMaxOffsetNumber(cpage);

		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			IvfflatList list;
			Vector *center_vector;
			float *center_data;
			
			if (center_idx >= totalLists) {
				elog(ERROR, "聚类中心索引超出范围: %d >= %d", center_idx, totalLists);
				UnlockReleaseBuffer(cbuf);
				pfree(centers_data);
				return -1;
			}
			
			list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
			
			/* 提取聚类中心向量数据 */
			center_vector = &list->center;
			center_data = (float *) VARDATA(center_vector);
			
			/* 复制到centers_data数组 */
			memcpy(&centers_data[center_idx * dimensions], center_data, dimensions * sizeof(float));
			center_idx++;
		}

		nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;
		UnlockReleaseBuffer(cbuf);
	}
	
	if (center_idx != totalLists) {
		elog(ERROR, "聚类中心数量不匹配: 预期 %d, 实际 %d", totalLists, center_idx);
		pfree(centers_data);
		return -1;
	}
	
	// elog(LOG, "聚类中心数据收集完成，开始上传到GPU");
	
	/* 上传聚类中心数据到GPU */
	ctx = (CudaCenterSearchContext*)so->cuda_ctx;
	
	/* 添加类型安全检查 */
	if (!ctx) {
		elog(ERROR, "CUDA上下文指针为空");
		pfree(centers_data);
		return -1;
	}
	
	elog(LOG, "CUDA上下文信息 - 聚类中心数: %d, 维度: %d, 零拷贝: %s, 已初始化: %s", 
		 ctx->num_centers, ctx->dimensions, 
		 ctx->use_zero_copy ? "是" : "否",
		 ctx->initialized ? "是" : "否");
	
	if (!ctx->initialized) {
		elog(ERROR, "CUDA上下文未初始化");
		pfree(centers_data);
		return -1;
	}
	
	if (ctx->use_zero_copy) {
		// elog(LOG, "使用零拷贝模式上传数据");
		upload_result = cuda_upload_centers_zero_copy(ctx, centers_data);
	} else {
		// elog(LOG, "使用标准模式上传数据");
		upload_result = cuda_upload_centers(ctx, centers_data);
	}
	
	/* 清理临时内存 */
	pfree(centers_data);
	
	if (upload_result == 0) {
		so->centers_uploaded = true;
		// elog(LOG, "聚类中心数据已成功上传到GPU (%d个中心)", totalLists);
	} else {
		elog(ERROR, "聚类中心数据上传到GPU失败，错误代码: %d", upload_result);
	}
	
	return upload_result;
}
#endif

/*
 * Get items - 未使用的函数，已注释
 */
/* 未使用的函数已删除 */

/*
 * Get items
 */
static void
GetScanItems_GPU(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	TupleDesc	tupdesc = RelationGetDescr(scan->indexRelation);
	TupleTableSlot *slot = so->vslot;
	int			batchProbes = 0;

	tuplesort_reset(so->sortstate);

	/* Search closest probes lists */
	while (so->listIndex < so->maxProbes && (++batchProbes) <= so->probes)
	{
		BlockNumber searchPage = so->listPages[so->listIndex++];

		/* Search all entry pages for list */
		while (BlockNumberIsValid(searchPage))
		{
			Buffer		buf;
			Page		page;
			OffsetNumber maxoffno;

			buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, searchPage, RBM_NORMAL, so->bas);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			maxoffno = PageGetMaxOffsetNumber(page);

			for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
			{
				IndexTuple	itup;
				Datum		datum;
				bool		isnull;
				ItemId		itemid = PageGetItemId(page, offno);

				itup = (IndexTuple) PageGetItem(page, itemid);
				datum = index_getattr(itup, 1, tupdesc, &isnull);

				/*
				 * Add virtual tuple
				 *
				 * Use procinfo from the index instead of scan key for
				 * performance
				 */
				ExecClearTuple(slot);
				slot->tts_values[0] = so->distfunc(so->procinfo, so->collation, datum, value);
				slot->tts_isnull[0] = false;
				slot->tts_values[1] = PointerGetDatum(&itup->t_tid);
				slot->tts_isnull[1] = false;
				ExecStoreVirtualTuple(slot);

				tuplesort_puttupleslot(so->sortstate, slot);
			}

			searchPage = IvfflatPageGetOpaque(page)->nextblkno;

			UnlockReleaseBuffer(buf);
		}
	}

	tuplesort_performsort(so->sortstate);

#if defined(IVFFLAT_MEMORY)
	elog(INFO, "memory: %zu MB", MemoryContextMemAllocated(CurrentMemoryContext, true) / (1024 * 1024));
#endif
}

/*
 * Get items (non-GPU version)
 */
static void
GetScanItems(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	TupleDesc	tupdesc = RelationGetDescr(scan->indexRelation);
	TupleTableSlot *slot = so->vslot;
	int			batchProbes = 0;

	tuplesort_reset(so->sortstate);

	/* Search closest probes lists */
	while (so->listIndex < so->maxProbes && (++batchProbes) <= so->probes)
	{
		BlockNumber searchPage = so->listPages[so->listIndex++];

		/* Search all entry pages for list */
		while (BlockNumberIsValid(searchPage))
		{
			Buffer		buf;
			Page		page;
			OffsetNumber maxoffno;

			buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, searchPage, RBM_NORMAL, so->bas);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			maxoffno = PageGetMaxOffsetNumber(page);

			for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
			{
				IndexTuple	itup;
				Datum		datum;
				bool		isnull;
				ItemId		itemid = PageGetItemId(page, offno);

				itup = (IndexTuple) PageGetItem(page, itemid);
				datum = index_getattr(itup, 1, tupdesc, &isnull);

				/*
				 * Add virtual tuple
				 *
				 * Use procinfo from the index instead of scan key for
				 * performance
				 */
				ExecClearTuple(slot);
				slot->tts_values[0] = so->distfunc(so->procinfo, so->collation, datum, value);
				slot->tts_isnull[0] = false;
				slot->tts_values[1] = PointerGetDatum(&itup->t_tid);
				slot->tts_isnull[1] = false;
				ExecStoreVirtualTuple(slot);

				tuplesort_puttupleslot(so->sortstate, slot);
			}

			searchPage = IvfflatPageGetOpaque(page)->nextblkno;

			UnlockReleaseBuffer(buf);
		}
	}

	tuplesort_performsort(so->sortstate);

#if defined(IVFFLAT_MEMORY)
	elog(INFO, "memory: %zu MB", MemoryContextMemAllocated(CurrentMemoryContext, true) / (1024 * 1024));
#endif
}

/*
 * Zero distance
 */
static Datum
ZeroDistance(FmgrInfo *flinfo, Oid collation, Datum arg1, Datum arg2)
{
	return Float8GetDatum(0.0);
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
	{
		value = PointerGetDatum(NULL);
		so->distfunc = ZeroDistance;
	}
	else
	{
		value = scan->orderByData->sk_argument;
		so->distfunc = FunctionCall2Coll;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->normprocinfo != NULL)
		{
			MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

			value = IvfflatNormValue(so->typeInfo, so->collation, value);

			MemoryContextSwitchTo(oldCtx);
		}
	}

	return value;
}

/*
 * Initialize scan sort state
 */
static Tuplesortstate *
InitScanSortState(TupleDesc tupdesc)
{
	AttrNumber	attNums[] = {1};
	Oid			sortOperators[] = {Float8LessOperator};
	Oid			sortCollations[] = {InvalidOid};
	bool		nullsFirstFlags[] = {false};

	return tuplesort_begin_heap(tupdesc, 1, attNums, sortOperators, sortCollations, nullsFirstFlags, work_mem, NULL, false);
}

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

	// elog("LOG", "ivfflatbeginscan");
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

#ifdef USE_CUDA
	/* 初始化GPU支持 */
	so->use_gpu = false;
	so->centers_uploaded = false;
	so->cuda_ctx = NULL;
	so->gpu_distances = NULL;
	
	/* 先设置 scan->opaque，这样后续的 GPU 测试才能正常工作 */
	scan->opaque = so;
	
	/* 逐步测试 GPU 功能 */
	// elog(LOG, "开始测试 GPU 功能");
	
	/* 测试 CUDA 可用性检查 */
	if (cuda_is_available()) {
		// elog(LOG, "CUDA 可用性检查通过");
		
		/* 测试 CUDA 基本功能 */
		if (cuda_basic_test()) {
			// elog(LOG, "CUDA 基本功能测试通过");
			
			/* 测试 CUDA 上下文初始化 */
			// elog(LOG, "开始测试 CUDA 上下文初始化");
			so->cuda_ctx = cuda_center_search_init(lists, dimensions, false);
			if (so->cuda_ctx) {
				// elog(LOG, "CUDA 上下文初始化成功");
				
				/* 测试数据上传 */
				// elog(LOG, "开始测试数据上传功能");
				if (UploadCentersToGPU(scan) == 0) {
					// elog(LOG, "数据上传成功");
				} else {
					// elog(LOG, "数据上传失败");
					/* 如果数据上传失败，清理CUDA上下文 */
					cuda_center_search_cleanup(so->cuda_ctx);
					so->cuda_ctx = NULL;
				}
				
				/* 注意：不要在这里清理CUDA上下文，因为后续还需要使用 */
				// elog(LOG, "CUDA 上下文保留用于后续使用");
			} else {
				// elog(LOG, "CUDA 上下文初始化失败");
			}
		} else {
			elog(ERROR, "CUDA 基本功能测试失败");
		}
	} else {
		elog(LOG, "CUDA 不可用");
	}
	
	/* 启用 GPU 支持 */
	if (so->cuda_ctx) {
		so->use_gpu = true;
		so->gpu_distances = palloc(lists * sizeof(float));
		if (!so->gpu_distances) {
			elog(ERROR, "无法分配GPU距离结果内存，将使用CPU计算");
			so->use_gpu = false;
			cuda_center_search_cleanup(so->cuda_ctx);
			so->cuda_ctx = NULL;
		} else {
			elog(LOG, "GPU聚类中心搜索已启用（标准模式）");
		}
	} else {
		so->use_gpu = false;
		so->gpu_distances = NULL;
	}
#endif

	MemoryContextSwitchTo(oldCtx);
	
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
#ifdef USE_CUDA
		IvfflatBench("GetScanItems", GetScanItems_GPU(scan, value));
#else
		IvfflatBench("GetScanItems", GetScanItems(scan, value));
#endif
		so->first = false;
		so->value = value;
	}

	while (!tuplesort_gettupleslot(so->sortstate, true, false, so->mslot, NULL))
	{
		if (so->listIndex == so->maxProbes)
			return false;

#ifdef USE_CUDA
		IvfflatBench("GetScanItems", GetScanItems_GPU(scan, so->value));
#else
		IvfflatBench("GetScanItems", GetScanItems(scan, so->value));
#endif
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

#ifdef USE_CUDA
	/* 清理CUDA资源 */
	if (so->cuda_ctx) {
		cuda_center_search_cleanup((CudaCenterSearchContext*)so->cuda_ctx);
		so->cuda_ctx = NULL;
	}
	
	if (so->gpu_distances) {
		pfree(so->gpu_distances);
		so->gpu_distances = NULL;
	}
#endif

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}

// typedef struct IvfjlScanOpaqueData {
//     JLProjection jlproj;
//     // ... 其它 ivfflat scan opaque 字段 ...
// } IvfjlScanOpaqueData;
// typedef IvfjlScanOpaqueData *IvfjlScanOpaque;

// IndexScanDesc ivfjlbeginscan(Relation index, int nkeys, int norderbys) {
//     IndexScanDesc scan = RelationGetIndexScan(index, nkeys, norderbys);
//     IvfjlScanOpaque so = (IvfjlScanOpaque) palloc0(sizeof(IvfjlScanOpaqueData));
//     // 读取 JL 投影矩阵
//     Buffer metaBuf = ReadBuffer(index, IVFFLAT_METAPAGE_BLKNO);
//     LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
//     Page metaPage = BufferGetPage(metaBuf);
//     ReadJLFromMetaPage(metaPage, &so->jlproj, CurrentMemoryContext);
//     UnlockReleaseBuffer(metaBuf);
//     scan->opaque = so;
//     return scan;
// }