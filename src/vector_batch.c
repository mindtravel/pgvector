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


#ifdef USE_CUDA
#include "cuda/cuda_wrapper.h"
#endif

/* 批量扫描结果结构 */
typedef struct BatchScanResult
{
    int query_id;      /* 查询ID */
    int row_id;        /* 行ID */
    float8 distance;   /* 距离 */
    Datum embedding;   /* 向量数据 */
} BatchScanResult;

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
	VectorBatch *result;
	int			count = 0, dim = 0;
	char	   *ptr;
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

	elog(LOG, "DEBUG: vector_batch_in: 接收到 %d 个向量，每个向量 %d 维", count, dim);

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
	
	/* 获取第一个向量的维度 */
	Vector *first_vec;
	int dim;
	Vector *vec;

	elog(LOG, "vector_batch_from_array: 开始执行");

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


	first_vec = DatumGetVector(elems[0]);
	dim = first_vec->dim;

	result = InitVectorBatch(nelems, dim);

	for (i = 0; i < nelems; i++)
	{
		if (nulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("vector array cannot contain null values")));

		vec = DatumGetVector(elems[i]);
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

	/* 调试输出：显示接收到的向量个数和维度 */
	elog(LOG, "batch_l2_distance: 开始执行，查询向量数量=%d，维度=%d", 
		 queries->count, queries->dim);

	if (queries->dim != target->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d", queries->dim, target->dim)));

#ifdef USE_CUDA
	/* 尝试使用GPU批量计算 */
	elog(LOG, "batch_l2_distance: 尝试使用GPU批量计算");
	
	/* 检查CUDA是否可用 */
	if (cuda_is_available()) {		
		/* 准备批量查询数据 */
		float *batch_query_data;
		float *target_data;
		float *gpu_distances;
		Vector *query;
		CudaCenterSearchContext* ctx;
		
		elog(LOG, "batch_l2_distance: CUDA可用，开始GPU批量计算");

		batch_query_data = (float *) palloc(queries->count * queries->dim * sizeof(float));
		target_data = (float *) palloc(queries->dim * sizeof(float));
		gpu_distances = (float *) palloc(queries->count * sizeof(float));
		
		/* 复制查询向量数据 */
		for (i = 0; i < queries->count; i++) {
			query = VectorBatchGetVector(queries, i);
			memcpy(&batch_query_data[i * queries->dim], query->x, queries->dim * sizeof(float));
		}
		
		/* 复制目标向量数据 */
		memcpy(target_data, target->x, queries->dim * sizeof(float));
		
		elog(LOG, "batch_l2_distance: 数据准备完成，开始GPU批量计算");
		
		/* 创建GPU上下文 - 使用批量支持 */
		ctx = cuda_center_search_init(queries->count, queries->dim, false);
		if (ctx) {
			elog(LOG, "batch_l2_distance: GPU上下文创建成功");
			
			/* 上传查询向量数据到GPU */
			if (cuda_upload_centers(ctx, batch_query_data) == 0) {
				elog(LOG, "batch_l2_distance: 查询向量数据上传成功");
				
				/* 使用CUDA Wrapper中的批量距离计算函数 */
				if (cuda_compute_batch_center_distances(ctx, target_data, 1, gpu_distances) == 0) {
					elog(LOG, "batch_l2_distance: GPU批量距离计算成功");
					
					/* 转换结果 */
					distances = palloc(sizeof(Datum) * queries->count);
					for (i = 0; i < queries->count; i++) {
						distances[i] = Float8GetDatum(gpu_distances[i]);
					}
					
					result = construct_array(distances, queries->count, FLOAT8OID,
											 sizeof(float8), FLOAT8PASSBYVAL, 'd');
					
					/* 清理GPU资源 */
					cuda_center_search_cleanup(ctx);
					pfree(batch_query_data);
					pfree(target_data);
					pfree(gpu_distances);
					pfree(distances);
					
					elog(LOG, "batch_l2_distance: GPU批量计算完成，返回结果");
					PG_RETURN_ARRAYTYPE_P(result);
				} else {
					elog(WARNING, "batch_l2_distance: GPU批量距离计算失败，回退到CPU计算");
				}
			} else {
				elog(WARNING, "batch_l2_distance: 查询向量数据上传失败，回退到CPU计算");
			}
			
			cuda_center_search_cleanup(ctx);
		} else {
			elog(WARNING, "batch_l2_distance: GPU上下文创建失败，回退到CPU计算");
		}
		
		pfree(batch_query_data);
		pfree(target_data);
		pfree(gpu_distances);
	} else {
		elog(LOG, "batch_l2_distance: CUDA不可用，使用CPU计算");
	}
#else
	elog(LOG, "batch_l2_distance: 未编译CUDA支持，使用CPU计算");
#endif

	/* CPU计算（回退方案） */
	elog(LOG, "batch_l2_distance: 开始CPU计算");
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
	elog(LOG, "batch_l2_distance: CPU计算完成，返回结果");
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

