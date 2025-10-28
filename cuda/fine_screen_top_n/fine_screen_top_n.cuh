#ifndef FINE_SCREEN_TOP_N
#define FINE_SCREEN_TOP_N

#include "pch.h"


/**
 * 精筛topn融合算子: 
 * 入参为query查询向量 h_query_group，query对应的cluster h_query_cluster_group
 * 需要一个cluster和对应的连续空间的map，用于查询cluster在实际数组中位置
 * cluster对应的query的倒排索引 h_cluster_query_offset, cluster-query的数组内容 h_cluster_query_data, 
 * cluster和offset（一个map），cluster_vector_index , cluster_vector 全量cluster向量（二维指针，h_cluster_vector[i]指向第i/n_cluster个cluster的i%n_cluster个向量）  
 * nquery，ncluster， n_dim，n_topn
 * max_cluster_id为cluster_id的最大值，用于确定cluster_map的大小
 * 出参 为query对应的topn向量的原始索引 h_query_topn_index
 * 出参 为query对应的topn向量的距离 h_query_topn_dist (一维数组，连续存储所有query的topn距离)
 * 目前还没有边界条件判断，如果聚类向量数量小于k，可能出问题
 **/
void fine_screen_top_n(
    float* h_query_group, int* h_query_cluster_group, int* h_cluster_query_offset, int* h_cluster_query_data,
    int* cluster_map,
    int* h_cluster_vector_index, int* h_cluster_vector_num, float** h_cluster_vector,
    int n_query, int n_cluster, int distinct_cluster_count, int n_dim, int n_topn, int max_cluster_id, int tol_vector,
    int* h_query_topn_index, float* h_query_topn_dist
);
