#include "postgres.h"
#include "access/reloptions.h"
#include "catalog/index.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/guc.h"
#include "ivfjl.h"
#include "ivfflat.h"
#include "ivfjlbuild.h"

/* Forward declarations for IVFJL functions */
static void IvfjlBuildIndex(Relation heap, Relation index, IndexInfo *indexInfo,
    IvfjlBuildState * buildstate, ForkNumber forkNum);
static void IvfjlInitBuildState(IvfjlBuildState * buildstate, Relation heap, Relation index, IndexInfo *indexInfo);
// static void IvfjlComputeCenters(IvfjlBuildState * buildstate);
static void IvfjlCreateMetaPage(Relation index, int dimensions, int JLDimensions, int lists, ForkNumber forkNum, JLProjection *jlproj);
static void IvfjlFreeBuildState(IvfjlBuildState* buildstate);
static void IvfjlCreateListPages(Relation index, VectorArray centers, VectorArray jlCenters,
                                    int original_dim, int jl_dim, int lists, ForkNumber forkNum, ListInfo** listInfo);

/*
 * Build the index for an unlogged table
 */
void
ivfflatbuildempty(Relation index)
{
	IndexInfo  *indexInfo = BuildIndexInfo(index);
	IvfflatBuildState buildstate;

	BuildIndex(NULL, index, indexInfo, &buildstate, INIT_FORKNUM);
}

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
static void
IvfjlBuildIndex(Relation heap, Relation index, IndexInfo *indexInfo,
               IvfjlBuildState * buildstate, ForkNumber forkNum)
{
    IvfjlInitBuildState(buildstate, heap, index, indexInfo);

    ComputeCenters(&(buildstate->base));

    /* Create pages */
    CreateListPages(index, buildstate->base.centers, buildstate->base.dimensions, buildstate->base.lists, forkNum, &buildstate->base.listInfo);
    IvfjlCreateListPages(index, buildstate->base.centers, buildstate->base.dimensions, IVFJL_DEFAULT_REDUCED_DIM,
        buildstate->base.lists, forkNum, &buildstate->base.listInfo, &buildstate->jlProj);

    IvfjlCreateMetaPage(index, buildstate->base.dimensions, IVFJL_DEFAULT_REDUCED_DIM, buildstate->base.lists, forkNum, &buildstate->jlProj);

    CreateEntryPages(&buildstate->base, forkNum);

    /* Write WAL for initialization fork since GenericXLog functions do not */
    if (forkNum == INIT_FORKNUM)
        log_newpage_range(index, forkNum, 0, RelationGetNumberOfBlocksInFork(index, forkNum), true);

    IvfjlFreeBuildState(buildstate);
}

/*
 * Initialize the IVFJL build state
 */
static void
IvfjlInitBuildState(IvfjlBuildState * buildstate, Relation heap, Relation index, IndexInfo *indexInfo, JLProjection* jlproj)
{
	IvfjlOptions* opts;
	/* Initialize the base ivfflat build state */
    InitBuildState(&buildstate->base, heap, index, indexInfo);
    
    /* Initialize JL projection - set to zero initially */
	memset(&buildstate->jlProj, 0, sizeof(JLProjection));

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
// static void
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
static void
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
static void
IvfjlFreeBuildState(IvfjlBuildState * buildstate)
{
    /* Free JL projection */
    FreeJLProjection(&buildstate->jlProj);
    
    /* Free base build state */
    FreeBuildState(&buildstate->base);
}

static void
IvfjlCreateListPages(Relation index, VectorArray centers,
                     int original_dim, int jl_dim, int lists, ForkNumber forkNum, 
                     ListInfo** listInfo, JLProjection* jlproj)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	Size		listSize;
    IvfjlList list;

	listSize = MAXALIGN(IVFJL_LIST_SIZE(jlCenter->itemsize));
	list = palloc0(listSize);

	buf = IvfflatNewBuffer(index, forkNum);
	IvfflatInitRegisterPage(index, &buf, &page, &state);

	for (int i = 0; i < lists; i++)
	{
		OffsetNumber offno;

		/* Zero memory for each list */
		MemSet(list, 0, listSize);

		/* Load list */
		list->startPage = InvalidBlockNumber;
		list->insertPage = InvalidBlockNumber;
		memcpy(&list->jlCenter, VectorArrayGet(centers, i), VARSIZE_ANY(VectorArrayGet(centers, i)));

		/* Ensure free space */
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

	IvfflatCommitBuffer(buf, state);/*提交缓冲区到磁盘*/
	pfree(list);/*刷盘*/
}