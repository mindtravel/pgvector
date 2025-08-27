#include "postgres.h"

#include <float.h>

#include "access/generic_xlog.h"
#include "ivfflat.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/memutils.h"

void InsertTuple(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heapRel);