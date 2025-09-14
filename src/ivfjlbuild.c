#include "postgres.h"
#include "access/reloptions.h"
#include "catalog/index.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/guc.h"
#include "ivfjl.h"
#include "ivfflat.h"
#include "ivfjlbuild.h"

/*
 * Similar with ivfflatbuild
 */
IndexBuildResult *ivfjlbuild(Relation heap, Relation index, IndexInfo *indexInfo) {
    IndexBuildResult *result;
    IvfflatBuildState buildstate;

    IvfjlBuildIndex(heap, index, indexInfo, &buildstate, MAIN_FORKNUM);

    result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
    result->heap_tuples = buildstate.reltuples;
    result->index_tuples = buildstate.indtuples;

    return result;
}

/*
 * Build the index for an unlogged table with IVFJL
 */
void
ivfjlbuildempty(Relation index)
{
    IndexInfo  *indexInfo = BuildIndexInfo(index);
    IvfflatBuildState buildstate;

    IvfjlBuildIndex(NULL, index, indexInfo, &buildstate, INIT_FORKNUM);
}

/*
 * Build the IVFJL index
 */
void
IvfjlBuildIndex(Relation heap, Relation index, IndexInfo *indexInfo,
               IvfflatBuildState * buildState, ForkNumber forkNum)
{
    /*设置BuildState*/
    InitBuildState(buildState, heap, index, indexInfo); /*设置ivfflat相关字段*/
    IvfjlInitBuildState(buildState, index, buildState->jlProj); /*设置ivfjl相关字段*/

    ComputeCenters(buildState);

    /* Generate JL projection matrix */
    GenerateJLProjection(buildState->jlProj, buildState->dimensions, 
		IVFJL_DEFAULT_REDUCED_DIM, CurrentMemoryContext);

    /* Create pages */
    CreateListPages(index, buildState->centers, buildState->dimensions, buildState->lists, forkNum, &(buildState->listInfo));
    IvfjlCreateListPages(index, buildState->centers, buildState->jlCenters, buildState->dimensions, IVFJL_DEFAULT_REDUCED_DIM,
        buildState->lists, forkNum, &(buildState->listInfo), buildState->jlProj);

    CreateEntryPages(buildState, forkNum);
    IvfjlCreateMetaPage(index, buildState->dimensions, IVFJL_DEFAULT_REDUCED_DIM, buildState->lists, forkNum, buildState->jlProj);

    /* Write WAL for initialization fork since GenericXLog functions do not */
    if (forkNum == INIT_FORKNUM)
        log_newpage_range(index, forkNum, 0, RelationGetNumberOfBlocksInFork(index, forkNum), true);

    IvfjlFreeBuildState(buildState);
}

/*
 * Initialize the IVFJL build state
 */
void
IvfjlInitBuildState(IvfflatBuildState * buildstate, Relation index, JLProjection* jlproj)
{
	IvfflatOptions* opts;
	/* Initialize the base ivfflat build state */
    
    /* Initialize JL projection - set to zero initially */
	memset(buildstate->jlProj, 0, sizeof(JLProjection));

	/* Initialize JL centers array */
	buildstate->jlCenters = VectorArrayInit(buildstate->lists, IVFJL_DEFAULT_REDUCED_DIM, 
		buildstate->typeInfo->itemSize(IVFJL_DEFAULT_REDUCED_DIM));

	/* Initialize reorder parameters from index options */
	opts = (IvfflatOptions *) index->rd_options;
	if (opts) {
		buildstate->reorder = opts->reorder;
		buildstate->reorderCandidates = opts->reorderCandidates;
	} else {
		buildstate->reorder = true;  /* 默认启用重排序 */
		buildstate->reorderCandidates = IVFJL_DEFAULT_CANDIDATE_MULTIPLES;
	}
}

/*
 * Create meta page for IVFJL with JL projection data
 */
void
IvfjlCreateMetaPage(
	Relation index,
	int dimensions,
	int jlDimensions,
	int lists,
	ForkNumber forkNum,
	JLProjection* jlproj)
{
    Buffer 			buf;
    Page 			page;
    GenericXLogState *state;
    IvfflatMetaPage metap;

    buf = IvfflatNewBuffer(index, forkNum);
    IvfflatInitRegisterPage(index, &buf, &page, &state);

    /* Initialize meta page */
    metap = IvfflatPageGetMeta(page);
    metap->magicNumber = IVFFLAT_MAGIC_NUMBER;
    metap->version = IVFFLAT_VERSION;
    metap->dimensions = dimensions;
	metap->lists = lists;
	metap->jlDimensions = jlDimensions;

    /* Write JL projection data to meta page */
    // WriteJLToMetaPage(page, jlproj);

    IvfflatCommitBuffer(buf, state);
}

/*
 * Free resources for IVFJL build state
 */
void
IvfjlFreeBuildState(IvfflatBuildState * buildstate)
{
    /* Free JL projection */
    FreeJLProjection(buildstate->jlProj);

    /* Free JL centers array */
    if (buildstate->jlCenters) {
        VectorArrayFree(buildstate->jlCenters);
        buildstate->jlCenters = NULL;
    }

    /* Free base build state */
    FreeBuildState(buildstate);
}

// void
// IvfjlCreateListPages(Relation index, VectorArray centers,
//                      int original_dim, int jl_dim, int lists, ForkNumber forkNum, 
//                      ListInfo** listInfo, JLProjection* jlproj)
// {
// 	Buffer		buf;
// 	Page		page;
// 	GenericXLogState *state;
// 	Size		listSize;
//     IvfjlList list;
// 	Size		jlCenterSize;

// 	/* Calculate jlCenter size based on jl_dim instead of original_dim */
	// jlCenterSize = VECTOR_SIZE(jl_dim);
	// listSize = MAXALIGN(IVFJL_LIST_SIZE(jlCenterSize));
	// list = palloc0(listSize);

// 	buf = IvfflatNewBuffer(index, forkNum);
// 	IvfflatInitRegisterPage(index, &buf, &page, &state);

// 	for (int i = 0; i < lists; i++)
// 	{
// 		OffsetNumber offno;

// 		/* Zero memory for each list */
// 		MemSet(list, 0, listSize);

// 		/* Load list */
// 		list->startPage = InvalidBlockNumber;
// 		list->insertPage = InvalidBlockNumber;
		
// 		/* Initialize jlCenter with proper size and dimension */
// 		SET_VARSIZE(&list->jlCenter, VECTOR_SIZE(jl_dim));
// 		list->jlCenter.dim = jl_dim;
// 		list->jlCenter.unused = 0;
		
// 		/* Apply JL projection */
// 		JLProjectVector(jlproj, (Vector*)VectorArrayGet(centers, i), &(list->jlCenter));

// 		/* Ensure free space */
// 		if (PageGetFreeSpace(page) < listSize)
// 			IvfflatAppendPage(index, &buf, &page, &state, forkNum);

// 		/* Add the item */
// 		offno = PageAddItem(page, (Item) list, listSize, InvalidOffsetNumber, false, false);
// 		if (offno == InvalidOffsetNumber)
// 			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

// 		/* Save location info */
// 		(*listInfo)[i].blkno = BufferGetBlockNumber(buf);
// 		(*listInfo)[i].offno = offno;
// 	}

// 	IvfflatCommitBuffer(buf, state);/*提交缓冲区到磁盘*/
// 	pfree(list);/*刷盘*/
// }

/*
 * Create list pages for IVFJL with both original and projected centers
 */
void
IvfjlCreateListPages(Relation index, VectorArray centers, VectorArray jlCenters,
                     int original_dim, int jl_dim, int lists, ForkNumber forkNum, 
                     ListInfo** listInfo, JLProjection* jlproj)
{
    Buffer 		buf;
    Page 		page;
    GenericXLogState *state;
    Size 		listSize;
    IvfjlList list;

    /* Calculate size needed for list with both centers */
	listSize = MAXALIGN(IVFJL_LIST_SIZE(VECTOR_SIZE(jl_dim)));
	list = palloc0(listSize);

    buf = IvfflatNewBuffer(index, forkNum);
    IvfflatInitRegisterPage(index, &buf, &page, &state);

    /* Apply JL projection to centers first */
    for (int i = 0; i < lists; i++) {
        Vector *origin_center = (Vector *)VectorArrayGet(centers, i);
        Vector *jl_center = (Vector *)VectorArrayGet(jlCenters, i);
        
        /* Project the center vector */
        JLProjectVector(jlproj, origin_center->x, jl_center->x);
        jl_center->dim = jl_dim;
    }

    for (int i = 0; i < lists; i++) {
        OffsetNumber offno;

        /* Zero memory for each list */
        MemSet(list, 0, listSize);

        /* Load list */
        list->startPage = InvalidBlockNumber;
        list->insertPage = InvalidBlockNumber;
        
        /* Store projected center */
        memcpy(&list->jlCenter, VectorArrayGet(jlCenters, i), VARSIZE_ANY(VectorArrayGet(jlCenters, i)));

        /* Ensure free space - reuse existing logic */
        if (PageGetFreeSpace(page) < listSize)
            IvfflatAppendPage(index, &buf, &page, &state, forkNum);

        /* Add the item */
        offno = PageAddItem(page, (Item) list, listSize, InvalidOffsetNumber, false, false);
        if (offno == InvalidOffsetNumber)
            elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

        /* Save location info */
        (*listInfo)[i].blkno = BufferGetBlockNumber(buf);
        (*listInfo)[i].offno = offno;
    }

    IvfflatCommitBuffer(buf, state);
    pfree(list);
}