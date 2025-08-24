#ifndef IVFBUILD_H
#define IVFBUILD_H

#include "postgres.h"
#include "access/reloptions.h"
#include "catalog/index.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/guc.h"
#include "ivfjl.h"
#include "ivfflat.h"


void BuildIndex(Relation heap, Relation index, IndexInfo* indexInfo,
    IvfflatBuildState* buildstate, ForkNumber forkNum);
void InitBuildState(IvfflatBuildState* buildstate, Relation heap, Relation index, IndexInfo* indexInfo);
void FreeBuildState(IvfflatBuildState *buildstate);
void ComputeCenters(IvfflatBuildState *buildstate);
void CreateListPages(Relation index, VectorArray centers, int dimensions, int lists, ForkNumber forkNum, ListInfo **listInfo);
void CreateEntryPages(IvfflatBuildState *buildstate, ForkNumber forkNum);
Buffer IvfflatNewBuffer(Relation index, ForkNumber forkNum);
void IvfflatInitRegisterPage(Relation index, Buffer *buf, Page *page, GenericXLogState **state);
void IvfflatCommitBuffer(Buffer buf, GenericXLogState *state);
void IvfjlBuildIndex(Relation heap, Relation index, IndexInfo *indexInfo,
    IvfflatBuildState * buildstate, ForkNumber forkNum);
void IvfjlInitBuildState(IvfflatBuildState * buildstate, Relation index, JLProjection* jlproj);
// void IvfjlComputeCenters(IvfflatBuildState * buildstate);
void IvfjlCreateMetaPage(Relation index, int dimensions, int JLDimensions, int lists, ForkNumber forkNum, JLProjection *jlproj);
void IvfjlFreeBuildState(IvfflatBuildState* buildstate);
void IvfjlCreateListPages(Relation index, VectorArray centers, VectorArray jlCenters, int original_dim, int jl_dim, int lists, ForkNumber forkNum, ListInfo** listInfo, JLProjection *jlproj);

#endif