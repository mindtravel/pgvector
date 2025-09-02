#ifndef VECTOR_BATCH_H
#define VECTOR_BATCH_H

#include "postgres.h"
#include "vector.h"

#define VECTOR_BATCH_MAX_VECTORS 1024

// 向量数组结构
typedef struct VectorBatch
{
	int32		vl_len_;		/* varlena header */
	int16		count;			/* 向量数量 */
	int16		dim;			/* 向量维度 */
	int32		unused;			/* 保留字段 */
	Vector		vectors[FLEXIBLE_ARRAY_MEMBER];  /* 向量数据 */
} VectorBatch;

// 批量查询结果结构
typedef struct BatchQueryResult
{
	int			query_id;		/* 查询ID */
	ItemPointer	heap_tid;		/* 记录TID */
	float8		distance;		/* 距离值 */
} BatchQueryResult;

// 批量查询状态结构 
typedef struct BatchQueryState
{
	VectorBatch *queries;		/* 查询向量数组 */
	int			current_query;	/* 当前处理的查询索引 */
	int			k;				/* 每个查询的k值 */
	float8		radius;			/* 范围查询半径 */
	char		*distance_op;	/* 距离操作符 */
} BatchQueryState;

#define VECTOR_BATCH_SIZE(count, dim) \
	(offsetof(VectorBatch, vectors) + (count) * VECTOR_SIZE(dim))

#define VectorBatchGetVector(arr, idx) \
	((Vector*) ((char*)(arr)->vectors + (idx) * VECTOR_SIZE((arr)->dim)))

// 函数声明
VectorBatch *InitVectorBatch(int count, int dim);
void VectorBatchSetVector(VectorBatch *arr, int idx, Vector *vec);
Vector *VectorBatchGetVectorCopy(VectorBatch *arr, int idx);

// PostgreSQL函数接口
Datum vector_array_in(PG_FUNCTION_ARGS);
Datum vector_array_out(PG_FUNCTION_ARGS);
Datum vector_array_recv(PG_FUNCTION_ARGS);
Datum vector_array_send(PG_FUNCTION_ARGS);
Datum vector_array_from_array(PG_FUNCTION_ARGS);

Datum batch_l2_distance(PG_FUNCTION_ARGS);
Datum batch_cosine_distance(PG_FUNCTION_ARGS);
Datum batch_vector_negative_inner_product(PG_FUNCTION_ARGS);

Datum batch_knn_search(PG_FUNCTION_ARGS);
Datum batch_range_search(PG_FUNCTION_ARGS);
Datum batch_knn_operator(PG_FUNCTION_ARGS);  /* 新增：批量KNN操作符函数 */

#endif /* VECTOR_BATCH_H */
