#ifndef IVFJL_H
#define IVFJL_H

#include "postgres.h"
#include "vector.h"
#include "ivfflat.h"

/* IVFJL parameters */
#define IVFJL_DEFAULT_CANDIDATE_MULTIPLES	10
#define IVFJL_MIN_CANDIDATE_MULTIPLES		1
#define IVFJL_MAX_CANDIDATE_MULTIPLES		100
#define IVFJL_DEFAULT_PROBES	1

#define IVFJL_DEFAULT_REDUCED_DIM 32

#define IVFJL_LIST_SIZE(size)	(offsetof(IvfjlListData, jlCenter) + size)

extern bool ivfjl_enable_reorder;
extern int ivfjl_reorder_candidates;

// #define IvfjlPageGetMeta(page)	((IvfjlMetaPage) PageGetContents(page))
// #define IvfjlPageGetOpaque(page)	((IvfjlPageOpaque) PageGetSpecialPointer(page))


// /* JL 投影结构体 */
// typedef struct JLProjection {
//     int original_dim;
//     int reduced_dim;
//     float *matrix; // 行优先，大小 reduced_dim * original_dim
// } JLProjection;

// typedef struct IvfjlOptions
// {
//     IvfflatOptions          base;        /* 复用 ivfflat 的 Options */
// 	bool		            reorder;		/* 是否使用原始向量进行重排序 */
// 	int			            reorderCandidates; /* 重排序候选集大小 */
// }	IvfjlOptions;

// typedef struct IvfjlBuildState {
//     IvfflatBuildState       base;       /* 复用 ivfflat 的 build state */
    // JLProjection            jlProj;     /* JL 投影信息 */
//     bool                    reorder;    /* 是否使用原始向量进行重排序 */ 
//     int                     reorderCandidates; /* 重排序候选集倍数 */
// 	VectorArray jlCenters; // jl优化
// }   IvfjlBuildState;

// typedef struct IvfjlMetaPageData
// {
//     IvfflatMetaPageData     base;       /* 复用 ivfflat 的 build state */
	// uint16		            jlDimensions; // jl投影后的维度
// }	IvfjlMetaPageData;

// typedef IvfjlMetaPageData * IvfjlMetaPage;

/* IVFJL扫描状态结构体 */
typedef struct IvfjlScanOpaqueData {
    IvfflatScanOpaqueData   base;               /* 复用 ivfflat 的 scan opaque */
    JLProjection            jlProj;             /* JL 投影矩阵 */
    int			            jlDimensions;
	Datum		            jlValue;		    /*jl投影后的值*/
    bool                    reorder;            /* 是否使用原始向量进行重排序 */
    int                     reorderCandidates;  /* 重排序候选集大小 */
}   IvfjlScanOpaqueData;

typedef struct IvfjlListData
{
	BlockNumber startPage;
	BlockNumber insertPage;
	Vector		jlCenter;	 	/*jl投影之后的聚类中心*/
}			IvfjlListData;

typedef IvfjlListData * IvfjlList;

typedef IvfjlScanOpaqueData* IvfjlScanOpaque;

void IvfflatInit(void);

void GenerateJLProjection(JLProjection *proj, int original_dim, int reduced_dim, MemoryContext ctx);/*生成 JL 投影矩阵*/
// void JLProjectVector(const JLProjection *proj, Vector *srcVector, Vector *dstVector);/*对向量做 JL 投影*/
void JLProjectVector(const JLProjection *proj, float *src, float *dst);/*对向量做 JL 投影*/
void FreeJLProjection(JLProjection *proj);/*释放 JL 投影矩阵*/

void WriteJLToMetaPage(Page page, JLProjection *proj);/*JL 投影矩阵序列化到元数据页*/
void ReadJLFromMetaPage(Page page, JLProjection *proj, MemoryContext ctx);/*JL 投影矩阵反序列化*/



void IvfjlInit(void);   /*IVFJL初始化函数*/
Datum ivfjlhandler(PG_FUNCTION_ARGS);/*IVFJL处理器函数*/

IndexBuildResult *ivfjlbuild(Relation heap, Relation index, IndexInfo *indexInfo);/*JL 版本批量建索引主流程*/
void ivfjlbuildempty(Relation index);/*JL 版本空表建索引*/
void IvfjlBuildIndex(Relation heap, Relation index, IndexInfo *indexInfo, IvfflatBuildState * buildstate, ForkNumber forkNum);
void IvfjlInitBuildState(IvfflatBuildState * buildstate, Relation index, JLProjection *);
void IvfjlFreeBuildState(IvfflatBuildState * buildstate);
void IvfjlCreateMetaPage(Relation index, int dimensions, int jlDimensions, int lists, ForkNumber forkNum, JLProjection *jlproj);
void IvfjlCreateListPages(Relation index, VectorArray centers, VectorArray jlCenters, int original_dim, int jl_dim, int lists, ForkNumber forkNum, ListInfo** listInfo, JLProjection* jlproj);
void IvfjlGetMetaPageInfo(Relation index, int *lists, int *dimensions); 
/*jl扫描相关的函数*/
IndexScanDesc ivfjlbeginscan(Relation index, int nkeys, int norderbys);
void ivfjlrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
bool ivfjlgettuple(IndexScanDesc scan, ScanDirection dir);
void ivfjlendscan(IndexScanDesc scan);

bool ivfjlinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
                 ,bool indexUnchanged
#endif
                 ,IndexInfo *indexInfo);/*JL 版本单条插入主流程*/
IndexScanDesc ivfjlbeginscan(Relation index, int nkeys, int norderbys);/*对向量做 JL 投影*/

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif
