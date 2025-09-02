#include "postgres.h"

#include <math.h>
#include "catalog/pg_type.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/syscache.h"
#include "access/htup_details.h"
#include "funcapi.h"
#include "vector.h"
#include "vector_batch.h"

/* declare I/O functions from vector.c so we can use DirectFunctionCall */
extern Datum vector_in(PG_FUNCTION_ARGS);
extern Datum vector_out(PG_FUNCTION_ARGS);

PG_FUNCTION_INFO_V1(vector_batch_in);
PG_FUNCTION_INFO_V1(vector_batch_out);
PG_FUNCTION_INFO_V1(vector_batch_recv);
PG_FUNCTION_INFO_V1(vector_batch_send);
PG_FUNCTION_INFO_V1(vector_batch_from_array);
PG_FUNCTION_INFO_V1(batch_l2_distance);
PG_FUNCTION_INFO_V1(batch_cosine_distance);
PG_FUNCTION_INFO_V1(batch_vector_negative_inner_product);
PG_FUNCTION_INFO_V1(batch_knn_search);
PG_FUNCTION_INFO_V1(batch_range_search);

/*
 * 创建向量数组
 */
VectorBatch *
InitVectorBatch(int count, int dim)
{
	VectorBatch *result;
	Size		size;

	if (count < 1 || count > VECTOR_BATCH_MAX_VECTORS)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("vector array cannot have more than %d vectors", VECTOR_BATCH_MAX_VECTORS)));

	if (dim < 1 || dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("vector cannot have more than %d dimensions", VECTOR_MAX_DIM)));

	size = VECTOR_BATCH_SIZE(count, dim);
	result = (VectorBatch *) palloc0(size);
	
	SET_VARSIZE(result, size);
	result->count = count;
	result->dim = dim;

	return result;
}

/*
 * 设置向量数组中的向量
 */
void
VectorBatchSetVector(VectorBatch *arr, int idx, Vector *vec)
{
	Vector *target;
	Size vec_size;

	if (idx < 0 || idx >= arr->count)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("vector array index out of bounds")));

	if (vec->dim != arr->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d", vec->dim, arr->dim)));

	target = VectorBatchGetVector(arr, idx);
	vec_size = VECTOR_SIZE(vec->dim);
	memcpy(target, vec, vec_size);
}

/*
 * 获取向量数组中的向量副本
 */
Vector *
VectorBatchGetVectorCopy(VectorBatch *arr, int idx)
{
	Vector *source, *result;
	Size vec_size;

	if (idx < 0 || idx >= arr->count)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("vector array index out of bounds")));

	source = VectorBatchGetVector(arr, idx);
	vec_size = VECTOR_SIZE(arr->dim);
	result = (Vector *) palloc(vec_size);
	memcpy(result, source, vec_size);

	return result;
}

/*
 * 向量数组输入函数
 */
Datum
vector_batch_in(PG_FUNCTION_ARGS)
{
	char	   *str = PG_GETARG_CSTRING(0);
	Oid			typioparam = PG_GETARG_OID(1);
	int32		typmod = PG_GETARG_INT32(2);
	VectorBatch *result;
	int			count = 0, dim = 0;
	char	   *ptr;
	char	   *token;
	Vector	   *vec;
	int			i;

	/* 解析格式: [[1,2,3],[4,5,6],[7,8,9]] */
	if (str[0] != '[')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("malformed vector array literal")));

	/* 计算向量数量和维度 */
	ptr = str + 1;
	while (*ptr && *ptr != ']')
	{
		if (*ptr == '[')
		{
			count++;
			if (count == 1)
			{
				/* 计算第一个向量的维度 */
				char *temp = ptr + 1;
				while (*temp && *temp != ']')
				{
					if (*temp == ',')
						dim++;
					temp++;
				}
				dim++; /* 最后一个元素 */
			}
		}
		ptr++;
	}

	if (count == 0 || dim == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("empty vector array")));

	result = InitVectorBatch(count, dim);

	/* 解析每个向量 */
	ptr = str + 1;
	i = 0;
	while (*ptr && *ptr != ']' && i < count)
	{
		if (*ptr == '[')
		{
			char *end = ptr + 1;
			int nest_count = 1;
			int vec_len;
			char *vec_str;

			/* 找到匹配的右括号 */
			while (*end && nest_count > 0)
			{
				if (*end == '[') nest_count++;
				else if (*end == ']') nest_count--;
				end++;
			}
			
			if (nest_count != 0)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("malformed vector in array")));

			/* 提取向量字符串并解析 */
			vec_len = end - ptr;
			vec_str = palloc(vec_len + 1);
			memcpy(vec_str, ptr, vec_len);
			vec_str[vec_len] = '\0';

			/* 调用vector_in解析单个向量 */
			vec = DatumGetVector(DirectFunctionCall3(vector_in,
													CStringGetDatum(vec_str),
													ObjectIdGetDatum(InvalidOid),
													Int32GetDatum(-1)));
			
			VectorBatchSetVector(result, i, vec);
			pfree(vec_str);
			i++;
			ptr = end;
		}
		else
		{
			ptr++;
		}
	}

	PG_RETURN_POINTER(result);
}

/*
 * 向量数组输出函数
 */
Datum
vector_batch_out(PG_FUNCTION_ARGS)
{
	VectorBatch *arr = (VectorBatch *) PG_GETARG_POINTER(0);
	StringInfoData buf;
	int			i;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');

	for (i = 0; i < arr->count; i++)
	{
		Vector *vec = VectorBatchGetVector(arr, i);
		char *vec_str = DatumGetCString(DirectFunctionCall1(vector_out, PointerGetDatum(vec)));
		
		if (i > 0)
			appendStringInfoChar(&buf, ',');
		
		appendStringInfoString(&buf, vec_str);
		pfree(vec_str);
	}

	appendStringInfoChar(&buf, ']');
	PG_RETURN_CSTRING(buf.data);
}

/*
 * 向量数组接收函数
 */
Datum
vector_batch_recv(PG_FUNCTION_ARGS)
{
	StringInfo	buf = (StringInfo) PG_GETARG_POINTER(0);
	VectorBatch *result;
	int			count, dim;
	int			i;

	count = pq_getmsgint(buf, sizeof(int16));
	dim = pq_getmsgint(buf, sizeof(int16));
	
	elog(LOG, "DEBUG: vector_batch_recv: 接收到 %d 个向量，每个向量 %d 维", count, dim);

	result = InitVectorBatch(count, dim);

	for (i = 0; i < count; i++)
	{
		Vector *vec = VectorBatchGetVector(result, i);
		int j;
		
		vec->dim = dim;
		SET_VARSIZE(vec, VECTOR_SIZE(dim));
		
		for (j = 0; j < dim; j++)
		{
			vec->x[j] = pq_getmsgfloat4(buf);
		}
	}

	PG_RETURN_POINTER(result);
}

/*
 * 向量数组发送函数
 */
Datum
vector_batch_send(PG_FUNCTION_ARGS)
{
	VectorBatch *arr = (VectorBatch *) PG_GETARG_POINTER(0);
	StringInfoData buf;
	int			i, j;

	pq_begintypsend(&buf);
	pq_sendint16(&buf, arr->count);
	pq_sendint16(&buf, arr->dim);

	for (i = 0; i < arr->count; i++)
	{
		Vector *vec = VectorBatchGetVector(arr, i);
		for (j = 0; j < vec->dim; j++)
		{
			pq_sendfloat4(&buf, vec->x[j]);
		}
	}

	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/*
 * 从PostgreSQL数组创建向量数组
 */
Datum
vector_batch_from_array(PG_FUNCTION_ARGS)
{
	ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
	VectorBatch *result;
	Datum	   *elems;
	bool	   *nulls;
	int			nelems;
	int			i;

	if (ARR_NDIM(array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector array must be one-dimensional")));

	deconstruct_array(array, ARR_ELEMTYPE(array), -1, false, 'i',
					  &elems, &nulls, &nelems);

	if (nelems == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector array cannot be empty")));

	/* 获取第一个向量的维度 */
	Vector *first_vec = DatumGetVector(elems[0]);
	int dim = first_vec->dim;

	result = InitVectorBatch(nelems, dim);

	for (i = 0; i < nelems; i++)
	{
		if (nulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("vector array cannot contain null values")));

		Vector *vec = DatumGetVector(elems[i]);
		VectorBatchSetVector(result, i, vec);
	}

	pfree(elems);
	pfree(nulls);

	PG_RETURN_POINTER(result);
}

/*
 * 批量L2距离计算
 */
Datum
batch_l2_distance(PG_FUNCTION_ARGS)
{
	VectorBatch *queries = (VectorBatch *) PG_GETARG_POINTER(0);
	Vector	   *target = PG_GETARG_VECTOR_P(1);
	ArrayType  *result;
	Datum	   *distances;
	int			i;

	// /* 调试输出：显示接收到的向量个数和维度 */
	// elog(LOG, "DEBUG: batch_l2_distance: 接收到 %d 个查询向量，每个向量 %d 维",
	// 	queries->count, queries->dim);

	if (queries->dim != target->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d", queries->dim, target->dim)));

	distances = palloc(sizeof(Datum) * queries->count);

	for (i = 0; i < queries->count; i++)
	{
		Vector *query = VectorBatchGetVector(queries, i);
		float8 distance = 0.0;
		int j;

		for (j = 0; j < query->dim; j++)
		{
			float8 diff = query->x[j] - target->x[j];
			distance += diff * diff;
		}

		distances[i] = Float8GetDatum(sqrt(distance));
	}

	result = construct_array(distances, queries->count, FLOAT8OID,
							 sizeof(float8), FLOAT8PASSBYVAL, 'd');

	pfree(distances);
	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * 批量余弦距离计算
 */
Datum
batch_cosine_distance(PG_FUNCTION_ARGS)
{
	VectorBatch *queries = (VectorBatch *) PG_GETARG_POINTER(0);
	Vector	   *target = PG_GETARG_VECTOR_P(1);
	ArrayType  *result;
	Datum	   *distances;
	int			i;

	// /* 调试输出：显示接收到的向量个数和维度 */
	// elog(LOG, "DEBUG: batch_cosine_distance: 接收到 %d 个查询向量，每个向量 %d 维",
	// 	queries->count, queries->dim);

	if (queries->dim != target->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d", queries->dim, target->dim)));

	distances = palloc(sizeof(Datum) * queries->count);

	for (i = 0; i < queries->count; i++)
	{
		Vector *query = VectorBatchGetVector(queries, i);
		float8 dotproduct = 0.0;
		float8 norm_query = 0.0;
		float8 norm_target = 0.0;
		float8 distance;
		int j;

		for (j = 0; j < query->dim; j++)
		{
			dotproduct += query->x[j] * target->x[j];
			norm_query += query->x[j] * query->x[j];
			norm_target += target->x[j] * target->x[j];
		}

		norm_query = sqrt(norm_query);
		norm_target = sqrt(norm_target);

		if (norm_query == 0.0 || norm_target == 0.0)
			distance = 0.0;
		else
			distance = 1.0 - (dotproduct / (norm_query * norm_target));

		distances[i] = Float8GetDatum(distance);
	}

	result = construct_array(distances, queries->count, FLOAT8OID,
							 sizeof(float8), FLOAT8PASSBYVAL, 'd');

	pfree(distances);
	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * 批量内积计算
 */
Datum
batch_vector_negative_inner_product(PG_FUNCTION_ARGS)
{
	VectorBatch *queries = (VectorBatch *) PG_GETARG_POINTER(0);
	Vector	   *target = PG_GETARG_VECTOR_P(1);
	ArrayType  *result;
	Datum	   *products;
	int			i;

	// /* 调试输出：显示接收到的向量个数和维度 */
	// elog(LOG, "DEBUG: batch_inner_product: 接收到 %d 个查询向量，每个向量 %d 维",
	// 	queries->count, queries->dim);

	if (queries->dim != target->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d", queries->dim, target->dim)));

	products = palloc(sizeof(Datum) * queries->count);

	for (i = 0; i < queries->count; i++)
	{
		Vector *query = VectorBatchGetVector(queries, i);
		float8 product = 0.0;
		int j;

		for (j = 0; j < query->dim; j++)
		{
			product += query->x[j] * target->x[j];
		}

		products[i] = Float8GetDatum(-product); /* 负内积用于最大值查找 */
	}

	result = construct_array(products, queries->count, FLOAT8OID,
							 sizeof(float8), FLOAT8PASSBYVAL, 'd');

	pfree(products);
	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * 批量KNN搜索 - 这是一个简化版本，实际实现需要与索引集成
 */
Datum
batch_knn_search(PG_FUNCTION_ARGS)
{
	
	/* 这个函数需要与索引系统深度集成，这里只是接口定义 */
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("batch_knn_search requires index integration")));
	
	PG_RETURN_NULL();
}

/*
 * 批量KNN操作符 <<->> 
 * 输入：vector_batch 和 vector
 * 输出：SETOF record (query_id, embedding_id, distance)
 * 行为：对每个查询向量，计算与目标向量的距离，返回所有结果
 */
Datum
batch_knn_operator(PG_FUNCTION_ARGS)
{
	VectorBatch *queries = (VectorBatch *) PG_GETARG_POINTER(0);
	Vector	   *target = PG_GETARG_VECTOR_P(1);
	FuncCallContext *funcctx;
	BatchQueryResult *result;
	
	/* 第一次调用时初始化 */
	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc tupdesc;
		
		/* 创建函数上下文 */
		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
		
		/* 创建元组描述符 */
		tupdesc = CreateTemplateTupleDesc(3);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "query_id", INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "embedding_id", INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "distance", FLOAT8OID, -1, 0);
		
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		
		/* 分配结果数组 */
		funcctx->max_calls = queries->count;
		funcctx->user_fctx = palloc(sizeof(BatchQueryResult) * queries->count);
		
		/* 计算所有距离并存储结果 */
		result = (BatchQueryResult *) funcctx->user_fctx;
		for (int i = 0; i < queries->count; i++)
		{
			Vector *query = VectorBatchGetVector(queries, i);
			float8 distance = 0.0;
			
			/* 计算L2距离 */
			for (int j = 0; j < query->dim; j++)
			{
				float8 diff = query->x[j] - target->x[j];
				distance += diff * diff;
			}
			distance = sqrt(distance);
			
			/* 存储结果 */
			result[i].query_id = i;
			result[i].heap_tid = NULL; /* 这里暂时设为NULL，实际使用时应该传递真实的TID */
			result[i].distance = distance;
		}
		
		MemoryContextSwitchTo(oldcontext);
	}
	
	/* 后续调用时返回结果 */
	funcctx = SRF_PERCALL_SETUP();
	result = (BatchQueryResult *) funcctx->user_fctx;
	
	if (funcctx->call_cntr < funcctx->max_calls)
	{
		Datum values[3];
		bool nulls[3];
		HeapTuple tuple;
		
		/* 构建元组 */
		MemSet(nulls, 0, sizeof(nulls));
		values[0] = Int32GetDatum(result[funcctx->call_cntr].query_id);
		values[1] = Int32GetDatum(funcctx->call_cntr); /* 临时使用call_cntr作为embedding_id */
		values[2] = Float8GetDatum(result[funcctx->call_cntr].distance);
		
		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * 批量范围查询 - 占位实现，防止扩展加载失败
 */
Datum
batch_range_search(PG_FUNCTION_ARGS)
{
	/* 这个函数需要与索引深度集成，这里仅提供占位定义 */
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("batch_range_search requires index integration")));

	PG_RETURN_NULL();
}
