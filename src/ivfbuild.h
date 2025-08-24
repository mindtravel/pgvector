#include "postgres.h"
#include "access/reloptions.h"
#include "catalog/index.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/guc.h"
#include "ivfjl.h"
#include "ivfflat.h"
#include "ivfjlbuild.h"

/*这两个函数会在其他.c文件中使用，所以要添加这个头文件*/
void InitBuildState(IvfflatBuildState* buildstate, Relation heap, Relation index, IndexInfo* indexInfo);
void CreateEntryPages(IvfflatBuildState* buildstate, ForkNumber forkNum);