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

extern int			ivfflat_probes;
extern int			ivfflat_iterative_scan;
extern int			ivfflat_max_probes;
static relopt_kind ivfflat_relopt_kind;

static const struct config_enum_entry ivfflat_iterative_scan_options[] = {
	{"off", IVFFLAT_ITERATIVE_SCAN_OFF, false},
	{"relaxed_order", IVFFLAT_ITERATIVE_SCAN_RELAXED, false},
	{NULL, 0, false}
};

/*
 * Initialize IVFJL
 */
void
IvfjlInit(void)
{
	/* Similar to IvfflatInit but for IVFJL - can be extended later */
	/* Currently reuses ivfflat configurations */
}

/*
 * Get build phase name for IVFJL
 */
static char *
ivfjlbuildphasename(int64 phasenum)
{
	switch (phasenum)
	{
		case PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE:
			return "initializing";
		case PROGRESS_IVFFLAT_PHASE_KMEANS:
			return "performing k-means clustering";
		case PROGRESS_IVFFLAT_PHASE_ASSIGN:
			return "assigning tuples to lists";
		case PROGRESS_IVFFLAT_PHASE_LOAD:
			return "loading tuples";
		default:
			return NULL;
	}
}

/*
 * Estimate IVFJL scan cost
 */
static void
ivfjlcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
				  Cost *indexStartupCost, Cost *indexTotalCost,
				  Selectivity *indexSelectivity, double *indexCorrelation
#if PG_VERSION_NUM >= 100000
				  , double *indexPages
#endif
)
{
	/* Reuse ivfflat cost estimation for now */
	/* TODO: Implement IVFJL-specific cost estimation considering JL projection overhead */
	
	GenericCosts costs;
	int			lists;
	double		ratio;
	Relation	indexRel;

	/* Never use index without order */
	if (path->pathkeys == NIL)
	{
		*indexStartupCost = get_float8_infinity();
		*indexTotalCost = get_float8_infinity();
		*indexSelectivity = 0;
		*indexCorrelation = 0;
#if PG_VERSION_NUM >= 100000
		*indexPages = 0;
#endif
		return;
	}

	MemSet(&costs, 0, sizeof(costs));
	genericcostestimate(root, path, loop_count, &costs);

	indexRel = index_open(path->indexinfo->indexoid, NoLock);
	lists = IvfflatGetLists(indexRel);
	index_close(indexRel, NoLock);

	/* Adjust for IVFJL specifics */
	ratio = (double) ivfflat_probes / lists;
	if (ratio > 1.0)
		ratio = 1.0;

	*indexStartupCost = costs.indexStartupCost;
	*indexTotalCost = costs.indexStartupCost + ratio * costs.indexTotalCost;
	*indexSelectivity = costs.indexSelectivity;
	*indexCorrelation = costs.indexCorrelation;
#if PG_VERSION_NUM >= 100000
	*indexPages = costs.numIndexPages;
#endif
}

/*
 * IVFJL handler function
 */
PG_FUNCTION_INFO_V1(ivfjlhandler);
Datum
ivfjlhandler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	amroutine->amstrategies = 0;
	amroutine->amsupport = 5;  /* Same as ivfflat */
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;
	amroutine->amcanbackward = false;
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = false;
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = false;
	amroutine->amsearchnulls = false;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = false;
	amroutine->amcaninclude = false;
#if PG_VERSION_NUM >= 130000
	amroutine->amusemaintenanceworkmem = false;
#endif
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype = InvalidOid;

	/* Interface functions */
	amroutine->ambuild = ivfjlbuild;
	amroutine->ambuildempty = ivfjlbuildempty;
	amroutine->aminsert = ivfjlinsert;
	amroutine->ambulkdelete = ivfbulkdelete;
	amroutine->amvacuumcleanup = ivfvacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = ivfjlcostestimate;
	amroutine->amoptions = ivfflat_options;  /* Reuse ivfflat options */
	amroutine->amproperty = NULL;
	amroutine->ambuildphasename = ivfjlbuildphasename;
	amroutine->amvalidate = ivfflat_validate;  /* Reuse ivfflat validation */
#if PG_VERSION_NUM >= 140000
	amroutine->amadjustmembers = NULL;
#endif
	amroutine->ambeginscan = ivfjlbeginscan;
	amroutine->amrescan = ivfrescan;
	amroutine->amgettuple = NULL;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = ivfendscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;
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
void JLProjectVector(const JLProjection *proj, const float *src, float *dst) {
    for (int i = 0; i < proj->reduced_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < proj->original_dim; j++) {
            sum += proj->matrix[i * proj->original_dim + j] * src[j];
        }
        dst[i] = sum;
    }
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

/*
 * C++接口函数：直接创建IVFJL索引
 * 这个函数可以从C++代码中直接调用来创建IVFJL索引
 */
bool CreateIvfjlIndex(const char* table_name, const char* index_name, const char* column_name, int lists)
{
    StringInfoData str;
    const char *sql;
    int ret;
    
    if (!table_name || !index_name || !column_name)
        return false;
        
    if (lists < IVFFLAT_MIN_LISTS || lists > IVFFLAT_MAX_LISTS)
        lists = IVFFLAT_DEFAULT_LISTS;
    
    /* 构建CREATE INDEX SQL语句 */
    initStringInfo(&str);
    appendStringInfo(&str,
        "CREATE INDEX %s ON %s USING ivfjl (%s vector_l2_ops) WITH (lists = %d)",
        index_name, table_name, column_name, lists);
    
    sql = str.data;
    
    /* 执行SQL语句 */
    PG_TRY();
    {
        /* 开始事务 */
        if (!IsTransactionState())
            StartTransactionCommand();
            
        /* 执行SQL */
        ret = SPI_connect();
        if (ret != SPI_OK_CONNECT)
        {
            pfree(str.data);
            return false;
        }
        
        ret = SPI_execute(sql, false, 0);
        if (ret != SPI_OK_UTILITY)
        {
            SPI_finish();
            pfree(str.data);
            return false;
        }
        
        SPI_finish();
        
        /* 提交事务 */
        CommitTransactionCommand();
        
        pfree(str.data);
        return true;
    }
    PG_CATCH();
    {
        /* 回滚事务 */
        AbortCurrentTransaction();
        pfree(str.data);
        return false;
    }
    PG_END_TRY();
}

