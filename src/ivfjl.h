#ifndef IVFJL_H
#define IVFJL_H

#include "postgres.h"
#include "vector.h"
#include "ivfflat.h"

#define IVFJL_REDUCED_DIM 64

// JL 投影结构体
typedef struct JLProjection {
    int original_dim;
    int reduced_dim;
    float *matrix; // 行优先，大小 reduced_dim * original_dim
} JLProjection;

// 生成 JL 投影矩阵
void GenerateJLProjection(JLProjection *proj, int original_dim, int reduced_dim, MemoryContext ctx);
// 对向量做 JL 投影
void JLProjectVector(const JLProjection *proj, const float *src, float *dst);
// 释放 JL 投影矩阵
void FreeJLProjection(JLProjection *proj);

// JL 投影矩阵序列化到元数据页
void WriteJLToMetaPage(Page page, JLProjection *proj);
// JL 投影矩阵反序列化
void ReadJLFromMetaPage(Page page, JLProjection *proj, MemoryContext ctx);

// ivfjl build state
typedef struct IvfjlBuildState {
    IvfflatBuildState base; // 复用 ivfflat 的 build state
    JLProjection jlproj;    // JL 投影信息
} IvfjlBuildState;

// IVFJL初始化函数
void IvfjlInit(void);

// IVFJL处理器函数
Datum ivfjlhandler(PG_FUNCTION_ARGS);

// JL 版本批量建索引主流程
IndexBuildResult *ivfjlbuild(Relation heap, Relation index, IndexInfo *indexInfo);
// JL 版本空表建索引
void ivfjlbuildempty(Relation index);
// JL 版本单条插入主流程
bool ivfjlinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
                 Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
                 ,bool indexUnchanged
#endif
                 ,IndexInfo *indexInfo);
// JL 版本查询主流程
IndexScanDesc ivfjlbeginscan(Relation index, int nkeys, int norderbys);

#ifdef __cplusplus
extern "C" {
#endif

// C++接口函数：直接创建IVFJL索引
// 参数：table_name - 表名，index_name - 索引名，column_name - 列名，lists - 聚类数量
bool CreateIvfjlIndex(const char* table_name, const char* index_name, const char* column_name, int lists);

#ifdef __cplusplus
}
#endif

#endif
