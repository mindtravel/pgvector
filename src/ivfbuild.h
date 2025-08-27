#include "postgres.h"
#include "access/reloptions.h"
#include "catalog/index.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/guc.h"
#include "ivfflat.h"

/*这两个函数会在其他.c文件中使用，所以要添加这个头文件*/
/* Build functions */
void InitBuildState(IvfflatBuildState * buildstate, Relation heap, Relation index, IndexInfo *indexInfo);
void FreeBuildState(IvfflatBuildState * buildstate);
void ComputeCenters(IvfflatBuildState * buildstate);
void CreateMetaPage(Relation index, int dimensions, int lists, ForkNumber forkNum);
void CreateListPages(Relation index, VectorArray centers, int dimensions, int lists, ForkNumber forkNum, ListInfo** listInfo);
void CreateEntryPages(IvfflatBuildState * buildstate, ForkNumber forkNum);
void BuildIndex(Relation heap, Relation index, IndexInfo *indexInfo, IvfflatBuildState * buildstate, ForkNumber forkNum);