#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "test_utils.cuh"
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <set>

struct ClusterQueryData {
    int* cluster_map; //cluster_idx -> cluster_id
    int* cluster_query_data; //cluster_id -> query_ids
    int* cluster_query_offset; //cluster_id -> query_offset
    int* cluster_query_data_size; //cluster_id -> query_count
    int  cluster_size;
    int* cluster_vector_index; //cluster_id -> vector_index
    float** cluster_vector; //cluster_id -> vectors
    int* cluster_vector_num; //cluster_id -> vector_num
    int  tol_vector; 
};

void _check_cuda_last_error(const char *file, int line)
{
    // 调用 cudaGetLastError() 来获取最后一个异步错误
    // 这个函数开销极小，因为它不会同步设备，只是查询一个错误标志
    // 重要：它会清除当前的错误状态，以便下次检查不会重复报告同一个旧错误
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        // 如果检测到错误，打印详细信息，包括错误描述、发生检查的文件和行号
        fprintf(stderr, "[CUDA Last Error]: %s ---- Location: %s:%d\n",
                cudaGetErrorString(err), file, line);
        
        // 在调试时，立即终止程序是一个好习惯，可以防止程序在错误状态下继续运行导致更多混乱
        cudaDeviceReset(); // 尝试清理CUDA资源
        exit(EXIT_FAILURE);
    }
}

/**
 * 分配向量组的空间
 */
void** malloc_vector_list(size_t n_batch, size_t n_dim, size_t elem_size) {
    // 分配连续的数据内存块
    void* data = malloc(n_batch * n_dim * elem_size);
    if (data == NULL) return NULL;
    
    // 分配行指针数组
    void** vector_list = (void**)malloc(n_batch * sizeof(void*));
    if (vector_list == NULL) {
        free(data);
        return NULL;
    }
    
    // 计算每个元素的字节偏移量并设置指针
    for (size_t i = 0; i < n_batch; i++) {
        vector_list[i] = (char*)data + i * n_dim * elem_size;
    }
    
    return vector_list;
}

/**
 * 释放向量组的空间
 */
void free_vector_list(void** vector_list) {
    if (vector_list != NULL) {
        free(vector_list[0]);
        free(vector_list);
    }
}

/*
* 生成向量组
*/ 
float** generate_vector_list(int n_batch, int n_dim) {
    // 分配连续的内存
    float** vector_list = (float**)malloc_vector_list(n_batch, n_dim, sizeof(float));

    for (int i = 0; i < n_batch; i++) {
        for (int j = 0; j < n_dim; j++) {
            // vector_list[i][j] = 1.0f;
            // vector_list[i][j] = (float)i + (float)j;
            vector_list[i][j] = (float)rand() / RAND_MAX * 20.0f - 10.0f;
            // vector_list[i][j] = (float)(i+j + 1.0f);
            // if(i == 1 && j == 0)
            //     vector_list[i][j] = 1.0f;
            // else
            //     vector_list[i][j] = 0.0f;
        }
    }    

    return vector_list;
}

/*
* 生成大规模向量组
*/ 
float*** generate_large_scale_vectors(int n_lists, int n_batch, int n_dim) {
    std::cout << "生成大规模数据: " << n_lists << " lists, " 
              << n_batch << " vectors per list, " 
              << n_dim << " dimensions" << std::endl;
    
    // 分配三级指针结构
    float*** vector_lists = (float***)malloc(n_lists * sizeof(float**));
    
    for (int list_id = 0; list_id < n_lists; list_id++) {
        // 为每个list分配连续内存
        vector_lists[list_id] = (float**)malloc_vector_list(n_batch, n_dim, sizeof(float));
        
        // 初始化数据
        for (int i = 0; i < n_batch; i++) {
            for (int j = 0; j < n_dim; j++) {
                // 生成随机数据 [-10, 10]
                vector_lists[list_id][i][j] = (float)rand() / RAND_MAX * 20.0f - 10.0f;
            }
        }
        
        if (list_id % 100 == 0) {
            std::cout << "已生成 " << list_id << "/" << n_lists << " lists" << std::endl;
        }
    }
    
    std::cout << "大规模数据生成完成 ✓" << std::endl;
    return vector_lists;
}


//把原始的聚类中心topk数据转为cluster-query倒排数据
void* generate_cluster_query_data(int* query_cluster_group, int n_query, int k, int batch_size) {
    // 第一步：将query-cluster映射转换为cluster-query倒排映射
    std::map<int, std::set<int>> cluster_query_map;
    for (int i = 0; i < n_query; i++) {
        for (int j = 0; j < k; j++) {
            cluster_query_map[query_cluster_group[i * k + j]].insert(i);
        }
    }
    
    // 第二步：计算需要多少个batch（比如10个cluster，batch_size=3，需要4个batch：[3,3,3,1]）
    int total_clusters = cluster_query_map.size();
    int num_batches = (total_clusters + batch_size - 1) / batch_size; // 向上取整
    
    // 第三步：分配ClusterQueryData数组（每个batch一个）
    ClusterQueryData* cluster_query_data_array = (ClusterQueryData*)malloc(num_batches * sizeof(ClusterQueryData));
    
    // 第四步：为每个batch组装数据
    int cluster_idx = 0;
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        // 计算当前batch的cluster数量
        int clusters_in_this_batch = std::min(batch_size, total_clusters - batch_idx * batch_size);
        
        ClusterQueryData& current_batch = cluster_query_data_array[batch_idx];
        
        // 1. cluster_map: 存储当前batch中所有cluster的ID
        current_batch.cluster_map = (int*)malloc(clusters_in_this_batch * sizeof(int));
        
        // 2. cluster_size: 当前batch中的cluster数量
        current_batch.cluster_size = clusters_in_this_batch;
        
        // 3. cluster_query_offset: 每个cluster在query_data中的偏移量
        current_batch.cluster_query_offset = (int*)malloc(clusters_in_this_batch * sizeof(int));
        
        // 4. cluster_query_data_size: 每个cluster对应的query数量
        current_batch.cluster_query_data_size = (int*)malloc(clusters_in_this_batch * sizeof(int));
        
        // 5. cluster_vector_index: 每个cluster在向量数组中的起始索引
        current_batch.cluster_vector_index = (int*)malloc(clusters_in_this_batch * sizeof(int));
        
        // 6. cluster_vector_num: 每个cluster的向量数量
        current_batch.cluster_vector_num = (int*)malloc(clusters_in_this_batch * sizeof(int));
        
        // 获取当前batch的cluster迭代器
        auto it = cluster_query_map.begin();
        std::advance(it, batch_idx * batch_size);
        
        // 第一遍：统计总query数量和总向量数量
        int total_queries_in_batch = 0;
        int total_vectors_in_batch = 0;
        
        for (int i = 0; i < clusters_in_this_batch; i++) {
            int cluster_id = it->first;
            int query_count = it->second.size();
            
            // 获取向量数量（不实际分配内存）
            float** dummy_ptr = nullptr;
            int vector_num = 0;
            get_cluster_vector(cluster_id, dummy_ptr, &vector_num);
            
            total_queries_in_batch += query_count;
            total_vectors_in_batch += vector_num;
            it++;
        }
        
        // 分配query数据内存
        current_batch.cluster_query_data = (int*)malloc(total_queries_in_batch * sizeof(int));
        
        // 分配向量数据内存
        current_batch.cluster_vector = (float**)malloc(clusters_in_this_batch * sizeof(float*));
        
        // 第二遍：填充所有数据
        it = cluster_query_map.begin();
        std::advance(it, batch_idx * batch_size);
        
        int query_data_idx = 0;
        int vector_offset = 0;
        
        for (int i = 0; i < clusters_in_this_batch; i++) {
            int cluster_id = it->first;
            int query_count = it->second.size();
            
            // 填充cluster基本信息
            current_batch.cluster_map[i] = cluster_id;
            current_batch.cluster_query_data_size[i] = query_count;
            current_batch.cluster_query_offset[i] = query_data_idx;
            
            // 填充query数据
            for (int query_id : it->second) {
                current_batch.cluster_query_data[query_data_idx++] = query_id;
            }
            
            // 获取向量数据
            float** cluster_vector_ptr = nullptr;
            int vector_num = 0;
            get_cluster_vector(cluster_id, cluster_vector_ptr, &vector_num);
            
            // 填充向量信息
            current_batch.cluster_vector_num[i] = vector_num;
            current_batch.cluster_vector_index[i] = vector_offset;
            current_batch.cluster_vector[i] = cluster_vector_ptr;
            
            vector_offset += vector_num;
            cluster_idx++;
            it++;
        }
        
        current_batch.tol_vector = total_vectors_in_batch;
    }
    
    return (void*)cluster_query_data_array;
}

void get_cluster_vector(int cluster_id, float** cluster_vector, int* vector_num) {
    // 空方法，从索引里面获取原始向量
    
}

