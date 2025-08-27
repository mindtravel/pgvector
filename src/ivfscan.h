#include "postgres.h"

#include <float.h>

#include "access/relscan.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "lib/pairingheap.h"
#include "ivfflat.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"

#define GetScanList(ptr) pairingheap_container(IvfflatScanList, ph_node, ptr)
#define GetScanListConst(ptr) pairingheap_const_container(IvfflatScanList, ph_node, ptr)

int CompareLists(const pairingheap_node *a, const pairingheap_node *b, void *arg);
void GetScanLists(IndexScanDesc scan, Datum value);
void GetScanItems(IndexScanDesc scan, Datum value);
Datum GetScanValue(IndexScanDesc scan);
Tuplesortstate *InitScanSortState(TupleDesc tupdesc);