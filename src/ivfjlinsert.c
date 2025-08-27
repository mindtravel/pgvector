#include "postgres.h"

#include <float.h>

#include "access/generic_xlog.h"
#include "ivfflat.h"
#include "ivfjl.h"
#include "ivfinsert.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/memutils.h"
/*
 * Insert a single tuple into IVFJL index
 */
bool
ivfjlinsertboth(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
            ,bool indexUnchanged
#endif
            ,IndexInfo *indexInfo)
{
	if (isnull[0])
		return false;   

    /* Call ivfflatinsert with original vectors */
    // bool resultOrigin;
    ivfflatinsert(index, values, isnull, heap_tid, heap, checkUnique
    #if PG_VERSION_NUM >= 140000
                            ,indexUnchanged
    #endif
                            ,indexInfo);

    /*TODO: insert jl vectors*/

    // IvfjlScanOpaque so;
    // float *originalX;
    // float *projectedX;
    // Datum originalValue;
    // Datum projectedValue;

    // /* Get original vector */
    // originalValue = values[0];
    // originalX = (float *) DatumGetPointer(originalValue);
    
    // /* Allocate space for projected vector */
    // projectedX = (float *) palloc(IVFJL_DEFAULT_REDUCED_DIM * sizeof(float));
    
    // /* Read JL projection matrix from metapage */
    // Buffer metaBuf = ReadBuffer(index, IVFFLAT_METAPAGE_BLKNO);
    // LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
    // Page metaPage = BufferGetPage(metaBuf);
    // JLProjection jlProj;
    // ReadJLFromMetaPage(metaPage, &jlProj, CurrentMemoryContext);
    // UnlockReleaseBuffer(metaBuf);
    
    // /* Perform JL projection */
    // JLProjectVector(&jlProj, originalX, projectedX);
    
    // /* Convert projected vector back to Datum */
    // projectedValue = PointerGetDatum(projectedX);
    
    // /* Create new values array with both original and projected vectors */
    // bool newIsnull = false;
    // ivfjlinsert(index, &projectedValue, newIsnull, heap_tid, heap, checkUnique
    //     #if PG_VERSION_NUM >= 140000
    //                             ,indexUnchanged
    //     #endif
    //                             ,indexInfo);
        
    // /* Clean up */
    // pfree(projectedX);
    // FreeJLProjection(&jlProj);
    
    return false;
}

// /*
//  * Insert a tuple into IVFJL index with dual-page storage
//  */
// bool
// ivfjlinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
// #if PG_VERSION_NUM >= 140000
//             ,bool indexUnchanged
// #endif
//             ,IndexInfo *indexInfo)
// {
//     MemoryContext oldCtx;
//     MemoryContext insertCtx;
//     Datum originalValue;
//     Vector originalVector;
//     Vector projectedVector;
//     Datum projectedValue;
//     bool result;
    
//     /* Skip nulls */
//     if (isnull[0])
//         return false;
    
//     /*
//      * Use memory context since detoast, JL projection, and
//      * index_form_tuple can allocate
//      */
//     insertCtx = AllocSetContextCreate(CurrentMemoryContext,
//                                       "Ivfjl insert temporary context",
//                                       ALLOCSET_DEFAULT_SIZES);
//     oldCtx = MemoryContextSwitchTo(insertCtx);
    
//     /* Get original vector */
//     originalValue = values[0];
    
//     /* Convert to Vector for JL projection */
//     originalVector.dim = VECTOR_DIM(originalValue);
//     originalVector.x = (float *) DatumGetPointer(originalValue);
    
//     /* Allocate space for projected vector */
//     projectedVector.dim = IVFJL_DEFAULT_REDUCED_DIM;  // 或者从元数据页读取
//     projectedVector.x = (float *) palloc(projectedVector.dim * sizeof(float));
    
//     /* Read JL projection matrix from metapage */
//     Buffer metaBuf = ReadBuffer(index, IVFFLAT_METAPAGE_BLKNO);
//     LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
//     Page metaPage = BufferGetPage(metaBuf);
//     JLProjection jlProj;
//     ReadJLFromMetaPage(metaPage, &jlProj, CurrentMemoryContext);
//     UnlockReleaseBuffer(metaBuf);
    
//     /* Perform JL projection */
//     JLProjectVector(&jlProj, &originalVector, &projectedVector);
    
//     /* Convert projected vector back to Datum */
//     projectedValue = PointerGetDatum(&projectedVector);
    
//     /* Insert original vector to original vector pages */
//     result = IvfjlInsertOriginalVector(index, &originalValue, isnull, heap_tid, heap);
    
//     if (result)
//     {
//         /* Insert projected vector to projected vector pages */
//         result = IvfjlInsertProjectedVector(index, &projectedValue, heap_tid, heap);
//     }
    
//     /* Clean up */
//     pfree(projectedVector.x);
//     FreeJLProjection(&jlProj);
    
//     /* Delete memory context */
//     MemoryContextSwitchTo(oldCtx);
//     MemoryContextDelete(insertCtx);
    
//     return result;
// }


/*
 * ivfflat套壳
 */
bool
ivfjlinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
			  Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
			  ,bool indexUnchanged
#endif
			  ,IndexInfo *indexInfo
)
{
	MemoryContext oldCtx;
	MemoryContext insertCtx;

	/* Skip nulls */
	if (isnull[0])
		return false;

	/*
	 * Use memory context since detoast, IvfflatNormValue, and
	 * index_form_tuple can allocate
	 */
	insertCtx = AllocSetContextCreate(CurrentMemoryContext,
									  "Ivfflat insert temporary context",
									  ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(insertCtx);

	/* Insert tuple */
	InsertTuple(index, values, isnull, heap_tid, heap);

	/* Delete memory context */
	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(insertCtx);

	return false;
}