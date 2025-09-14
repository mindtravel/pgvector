#include "postgres.h"

#include <float.h>

#include "access/amapi.h"
#include "access/reloptions.h"
#include "commands/progress.h"
#include "commands/vacuum.h"
#include "ivfflat.h"
#include "utils/float.h"
#include "utils/guc.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"
#include "ivfjl.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(x) EmitWarningsOnPlaceholders(x)
#endif

/*复用ivfflat的变量，不需要再Init函数中再次注册*/
extern int			ivfflat_probes;
extern int			ivfflat_iterative_scan;
extern int			ivfflat_max_probes;
bool				ivfjl_enable_reorder;
int					ivfjl_reorder_candidates;
static relopt_kind ivfjl_relopt_kind;

/*
 * Initialize IVFJL index options and variables
 */
void
IvfjlInit(void)
{
	ivfjl_relopt_kind = add_reloption_kind();
	add_int_reloption(ivfjl_relopt_kind, "lists", "Number of inverted lists",
		IVFFLAT_DEFAULT_LISTS, IVFFLAT_MIN_LISTS, IVFFLAT_MAX_LISTS, AccessExclusiveLock);
	DefineCustomBoolVariable("ivfjl.enable_reorder", "Enable original vector reordering",
		NULL, &ivfjl_enable_reorder,
		true, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("ivfjl.reorder_candidates", "Default number of candidates for reordering",
		NULL, &ivfjl_reorder_candidates,
		IVFJL_DEFAULT_CANDIDATE_MULTIPLES, IVFJL_MIN_CANDIDATE_MULTIPLES, IVFJL_MAX_CANDIDATE_MULTIPLES,
		PGC_USERSET, 0, NULL, NULL, NULL);
}

/*
 * Get the name of index build phase
 */
static char *
ivfjlbuildphasename(int64 phasenum)
{
	switch (phasenum)
	{
		case PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE:
			return "initializing";
		case PROGRESS_IVFFLAT_PHASE_KMEANS:
			return "performing k-means";
		case PROGRESS_IVFFLAT_PHASE_ASSIGN:
			return "assigning tuples";
		case PROGRESS_IVFFLAT_PHASE_LOAD:
			return "loading tuples";
		default:
			return NULL;
	}
}

/*
 * Estimate the cost of an index scan
 */
static void
ivfjlcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
					Cost *indexStartupCost, Cost *indexTotalCost,
					Selectivity *indexSelectivity, double *indexCorrelation,
					double *indexPages)
{
	GenericCosts costs;
	int			lists;
	double		ratio;
	double		sequentialRatio = 0.5;
	double		startupPages;
	double		spc_seq_page_cost;
	Relation	index;

	/* Never use index without order */
	if (path->indexorderbys == NULL)
	{
		*indexStartupCost = get_float8_infinity();
		*indexTotalCost = get_float8_infinity();
		*indexSelectivity = 0;
		*indexCorrelation = 0;
		*indexPages = 0;
#if PG_VERSION_NUM >= 180000
		/* See "On disable_cost" thread on pgsql-hackers */
		path->path.disabled_nodes = 2;
#endif
		return;
	}

	MemSet(&costs, 0, sizeof(costs));

	genericcostestimate(root, path, loop_count, &costs);

	index = index_open(path->indexinfo->indexoid, NoLock);
	IvfflatGetMetaPageInfo(index, &lists, NULL);
	index_close(index, NoLock);

	/* Get the ratio of lists that we need to visit */
	ratio = ((double) ivfflat_probes) / lists;
	if (ratio > 1.0)
		ratio = 1.0;

	get_tablespace_page_costs(path->indexinfo->reltablespace, NULL, &spc_seq_page_cost);

	/* Change some page cost from random to sequential */
	costs.indexTotalCost -= sequentialRatio * costs.numIndexPages * (costs.spc_random_page_cost - spc_seq_page_cost);

	/* Startup cost is cost before returning the first row */
	costs.indexStartupCost = costs.indexTotalCost * ratio;

	/* Adjust cost if needed since TOAST not included in seq scan cost */
	startupPages = costs.numIndexPages * ratio;
	if (startupPages > path->indexinfo->rel->pages && ratio < 0.5)
	{
		/* Change rest of page cost from random to sequential */
		costs.indexStartupCost -= (1 - sequentialRatio) * startupPages * (costs.spc_random_page_cost - spc_seq_page_cost);

		/* Remove cost of extra pages */
		costs.indexStartupCost -= (startupPages - path->indexinfo->rel->pages) * spc_seq_page_cost;
	}

	*indexStartupCost = costs.indexStartupCost;
	*indexTotalCost = costs.indexTotalCost;
	*indexSelectivity = costs.indexSelectivity;
	*indexCorrelation = costs.indexCorrelation;
	*indexPages = costs.numIndexPages;
}

/*
 * Parse and validate the reloptions
 */
static bytea*
ivfjloptions(Datum reloptions, bool validate)
{
    static const relopt_parse_elt tab[] = {
		{"lists", RELOPT_TYPE_INT, offsetof(IvfflatOptions, lists)},
		{"reorder", RELOPT_TYPE_BOOL, offsetof(IvfflatOptions, reorder)},
		{"reorder_candidates", RELOPT_TYPE_INT, offsetof(IvfflatOptions, reorderCandidates)},
	};

	return (bytea *) build_reloptions(reloptions, validate,
									  ivfjl_relopt_kind,
									  sizeof(IvfflatOptions),
									  tab, lengthof(tab));
}

/*
 * Validate catalog entries for the specified operator class
 */
static bool
ivfjlvalidate(Oid opclassoid)
{
	return true;
}

/*
 * IVFJL handler function
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ivfjlhandler);
Datum
ivfjlhandler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	amroutine->amstrategies = 0;
	amroutine->amsupport = 5;
	amroutine->amoptsprocnum = 0;
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;
	amroutine->amcanbackward = false;	/* can change direction mid-scan */
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = false;
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = false;
	amroutine->amsearchnulls = false;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = false;
#if PG_VERSION_NUM >= 170000
	amroutine->amcanbuildparallel = true;
#endif
	amroutine->amcaninclude = false;
	amroutine->amusemaintenanceworkmem = false; /* not used during VACUUM */
#if PG_VERSION_NUM >= 160000
	amroutine->amsummarizing = false;
#endif
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype = InvalidOid;

	/* Interface functions */
	amroutine->ambuild = ivfflatbuild;
	amroutine->ambuildempty = ivfjlbuildempty; /*暂时测着没问题*/
	amroutine->aminsert = ivfflatinsert;
#if PG_VERSION_NUM >= 170000
	amroutine->aminsertcleanup = NULL;
#endif
	amroutine->ambulkdelete = ivfflatbulkdelete;
	amroutine->amvacuumcleanup = ivfflatvacuumcleanup;
    amroutine->amcanreturn = NULL;	/* tuple not included in heapsort */
    // 四个静态函数单独实现，不破坏ivfflat.h的设计
    amroutine->amcostestimate = ivfjlcostestimate;
	amroutine->amoptions = ivfjloptions;
	amroutine->amproperty = NULL;	/* TODO AMPROP_DISTANCE_ORDERABLE */
	amroutine->ambuildphasename = ivfjlbuildphasename;
	amroutine->amvalidate = ivfjlvalidate;
#if PG_VERSION_NUM >= 140000
	amroutine->amadjustmembers = NULL;
#endif
	amroutine->ambeginscan = ivfjlbeginscan;
	amroutine->amrescan = ivfjlrescan;
	amroutine->amgettuple = ivfflatgettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = ivfjlendscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;

	/* Interface functions to support parallel index scans */
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan = NULL;
	amroutine->amparallelrescan = NULL;

	PG_RETURN_POINTER(amroutine);
}

// 生成 JL 投影矩阵
void GenerateJLProjection(JLProjection *proj, int original_dim, int reduced_dim, MemoryContext ctx) {
    proj->original_dim = original_dim;
    proj->reduced_dim = reduced_dim;
    proj->matrix = (float *) MemoryContextAlloc(ctx, sizeof(float) * reduced_dim * original_dim);
    for (int i = 0; i < reduced_dim * original_dim; i++) {
        // ±1/sqrt(reduced_dim) 随机投影
        proj->matrix[i] = ((RandomDouble() > 0.5) ? 1.0f : -1.0f) / sqrtf((float)reduced_dim);
    }
}

// JL 投影
void JLProjectVector(const JLProjection *proj, float * src, float * dst) {
    for (int i = 0; i < proj->reduced_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < proj->original_dim; j++) {
            sum += proj->matrix[i * proj->original_dim + j] * src[j];
        }
        dst[i] = sum;
    }
	// 可替换并行矩阵乘法
}

// 释放 JL 投影矩阵
void FreeJLProjection(JLProjection *proj) {
    if (proj->matrix) {
        pfree(proj->matrix);
        proj->matrix = NULL;
    }
}

// JL 投影矩阵序列化到元数据页
void WriteJLToMetaPage(Page page, JLProjection *proj) {
    char *ptr = (char *)PageGetContents(page) + sizeof(IvfflatMetaPageData);
    memcpy(ptr, &proj->original_dim, sizeof(int));
    memcpy(ptr + sizeof(int), &proj->reduced_dim, sizeof(int));
    memcpy(ptr + 2 * sizeof(int), proj->matrix, sizeof(float) * proj->reduced_dim * proj->original_dim);
}

// JL 投影矩阵反序列化
void ReadJLFromMetaPage(Page page, JLProjection *proj, MemoryContext ctx) {
    char *ptr = (char *)PageGetContents(page) + sizeof(IvfflatMetaPageData);
    memcpy(&proj->original_dim, ptr, sizeof(int));
    memcpy(&proj->reduced_dim, ptr + sizeof(int), sizeof(int));
    proj->matrix = (float *)MemoryContextAlloc(ctx, sizeof(float) * proj->reduced_dim * proj->original_dim);
    memcpy(proj->matrix, ptr + 2 * sizeof(int), sizeof(float) * proj->reduced_dim * proj->original_dim);
}
