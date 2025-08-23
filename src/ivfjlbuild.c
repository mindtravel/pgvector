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
 * Build the index for an unlogged table
 */
// void
// ivfjlbuildempty(Relation index)
// {
// 	IndexInfo  *indexInfo = BuildIndexInfo(index);
// 	IvfjlBuildState buildstate;

// 	BuildIndex(NULL, index, indexInfo, &buildstate, INIT_FORKNUM);
// }

/*
 * Similar with ivfflatbuild
 */
IndexBuildResult *ivfjlbuild(Relation heap, Relation index, IndexInfo *indexInfo) {
    IndexBuildResult *result;
    IvfjlBuildState buildstate;

    IvfjlBuildIndex(heap, index, indexInfo, &buildstate, MAIN_FORKNUM);

    result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
    result->heap_tuples = buildstate.base.reltuples;
    result->index_tuples = buildstate.base.indtuples;

    return result;
}

/*
 * Build the index for an unlogged table with IVFJL
 */
void
ivfjlbuildempty(Relation index)
{
    IndexInfo  *indexInfo = BuildIndexInfo(index);
    IvfjlBuildState buildstate;

    IvfjlBuildIndex(NULL, index, indexInfo, &buildstate, INIT_FORKNUM);
}

/*
 * Build the IVFJL index
 */
void
IvfjlBuildIndex(Relation heap, Relation index, IndexInfo *indexInfo,
               IvfjlBuildState * buildState, ForkNumber forkNum)
{
	IvfflatBuildState* baseBuildState;
	baseBuildState = &(buildState->base);

    InitBuildState(&buildState->base, heap, index, indexInfo);
    IvfjlInitBuildState(buildState, heap, index, indexInfo, &buildState->jlProj);

    ComputeCenters(&(buildState->base));

    /* Generate JL projection matrix */
    GenerateJLProjection(&buildState->jlProj, baseBuildState->dimensions, 
		IVFJL_DEFAULT_REDUCED_DIM, CurrentMemoryContext);

    /* Create pages */
    CreateListPages(index, baseBuildState->centers, baseBuildState->dimensions, baseBuildState->lists, forkNum, &baseBuildState->listInfo);
    IvfjlCreateListPages(index, baseBuildState->centers, buildState->jlCenters, baseBuildState->dimensions, IVFJL_DEFAULT_REDUCED_DIM,
        baseBuildState->lists, forkNum, &baseBuildState->listInfo, &buildState->jlProj);

    IvfjlCreateMetaPage(index, baseBuildState->dimensions, IVFJL_DEFAULT_REDUCED_DIM, baseBuildState->lists, forkNum, &buildState->jlProj);

    CreateEntryPages(baseBuildState, forkNum);

    /* Write WAL for initialization fork since GenericXLog functions do not */
    if (forkNum == INIT_FORKNUM)
        log_newpage_range(index, forkNum, 0, RelationGetNumberOfBlocksInFork(index, forkNum), true);

    IvfjlFreeBuildState(buildState);
}

/*
 * Initialize the IVFJL build state
 */
void
IvfjlInitBuildState(IvfjlBuildState * buildstate, Relation heap, Relation index, IndexInfo *indexInfo, JLProjection* jlproj)
{
	IvfjlOptions* opts;
	/* Initialize the base ivfflat build state */
    
    /* Initialize JL projection - set to zero initially */
	memset(&buildstate->jlProj, 0, sizeof(JLProjection));

	/* Initialize JL centers array */
	buildstate->jlCenters = VectorArrayInit(buildstate->base.lists, IVFJL_DEFAULT_REDUCED_DIM, 
		buildstate->base.typeInfo->itemSize(IVFJL_DEFAULT_REDUCED_DIM));

	/* Initialize reorder parameters from index options */
	opts = (IvfjlOptions *) index->rd_options;
	if (opts) {
		buildstate->reorder = opts->reorder;
		buildstate->reorderCandidates = opts->reorderCandidates;
	} else {
		buildstate->reorder = true;  /* 默认启用重排序 */
		buildstate->reorderCandidates = IVFJL_DEFAULT_CANDIDATE_MULTIPLES;
	}
}

// /*
//  * Compute centers for IVFJL with JL projection
//  */
// void
// IvfjlComputeCenters(IvfjlBuildState * buildstate)
// {
//     int numSamples;

//     pgstat_progress_update_param(PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_IVFFLAT_PHASE_KMEANS);

//     /* Target 50 samples per list, with at least 10000 samples */
//     numSamples = buildstate->base.lists * 50;
//     if (numSamples < 10000)
//         numSamples = 10000;

//     /* Skip samples for unlogged table */
//     if (buildstate->base.heap == NULL)
//         numSamples = 1;

//     /* Sample rows with original dimensions first */
//     buildstate->base.samples = VectorArrayInit(numSamples, buildstate->base.dimensions, buildstate->base.centers->itemsize);
//     if (buildstate->base.heap != NULL)
//     {
//         SampleRows(&buildstate->base);

//         if (buildstate->base.samples->length < buildstate->base.lists)
//         {
//             ereport(NOTICE,
//                     (errmsg("ivfjl index created with little data"),
//                      errdetail("This will result in poor performance."),
//                      errhint("Consider increasing the number of sample rows.")));
//         }
//     }

//     /* Generate JL projection matrix with original dimensions */
//     GenerateJLProjection(&buildstate->jlProj, buildstate->base.dimensions, IVFJL_DEFAULT_REDUCED_DIM, CurrentMemoryContext);

//     /* Apply JL projection to samples */
//     for (int i = 0; i < buildstate->base.samples->length; i++)
//     {
//         Vector *vec = (Vector *)VectorArrayGet(buildstate->base.samples, i);
//         float *dst = (float *)palloc(sizeof(float) * IVFJL_DEFAULT_REDUCED_DIM);
        
//         /* Project the vector */
//         JLProjectVector(&buildstate->jlProj, vec->x, dst);
        
//         /* Replace original vector data with projected data */
//         memcpy(vec->x, dst, sizeof(float) * IVFJL_DEFAULT_REDUCED_DIM);
//         vec->dim = IVFJL_DEFAULT_REDUCED_DIM;
        
//         pfree(dst);
//     }

//     /* Update dimensions to reduced dimensions */
//     buildstate->base.dimensions = IVFJL_DEFAULT_REDUCED_DIM;
    
//     /* Reinitialize centers array with reduced dimensions */
//     VectorArrayFree(buildstate->base.centers);
//     buildstate->base.centers = VectorArrayInit(buildstate->base.lists, IVFJL_DEFAULT_REDUCED_DIM, 
//                                               buildstate->base.typeInfo->itemSize(IVFJL_DEFAULT_REDUCED_DIM));

//     /* Perform k-means clustering on projected samples */
//     IvfflatBench("k-means", IvfflatKmeans(buildstate->base.index, buildstate->base.samples, buildstate->base.centers, buildstate->base.typeInfo));

//     /* Free samples */
//     VectorArrayFree(buildstate->base.samples);
//     buildstate->base.samples = NULL;
// }

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
    IvfjlMetaPage metap;

    buf = IvfflatNewBuffer(index, forkNum);
    IvfflatInitRegisterPage(index, &buf, &page, &state);

    /* Initialize meta page */
    metap = IvfjlPageGetMeta(page);
    metap->base.magicNumber = IVFFLAT_MAGIC_NUMBER;
    metap->base.version = IVFFLAT_VERSION;
    metap->base.dimensions = dimensions;
	metap->base.lists = lists;
	metap->base.jlDimensions = jlDimensions;
	
	// metap->lastUsedOffset = 
    //     ((char *) metap + sizeof(IvfjlMetaPageData)) - (char *) page;

    /* Write JL projection data to meta page */
    WriteJLToMetaPage(page, jlproj);

    IvfflatCommitBuffer(buf, state);
}

/*
 * Free resources for IVFJL build state
 */
void
IvfjlFreeBuildState(IvfjlBuildState * buildstate)
{
    /* Free JL projection */
    FreeJLProjection(&buildstate->jlProj);

    /* Free JL centers array */
    if (buildstate->jlCenters) {
        VectorArrayFree(buildstate->jlCenters);
        buildstate->jlCenters = NULL;
    }

    /* Free base build state */
    FreeBuildState(&buildstate->base);
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
        JLProjectVector(jlproj, origin_center, jl_center);
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