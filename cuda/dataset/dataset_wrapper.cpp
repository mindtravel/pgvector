#include <stdexcept>
#include <cstdio>
#include <cstring>
#include "../pch.h"
#include "dataset.cuh"
#include "../kmeans/kmeans.cuh"

extern "C" {
/**
 * C包装函数：创建 ClusterDataset 对象
 * 
 * @return 返回 ClusterDataset 指针
 */
ClusterDataset* cluster_dataset_create(void) {
    try {
        ClusterDataset* dataset = new ClusterDataset();
        return dataset;
    } catch (const std::exception& e) {
        fprintf(stderr, "cluster_dataset_create: 异常 - %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "cluster_dataset_create: 未知异常\n");
        return nullptr;
    }
}

/**
 * C包装函数：销毁 ClusterDataset 对象
 * 
 * @param dataset ClusterDataset 指针
 */
void cluster_dataset_destroy(ClusterDataset* dataset) {
    if (!dataset) {
        return;
    }
    try {
        dataset->release();
        delete dataset;
    } catch (const std::exception& e) {
        fprintf(stderr, "cluster_dataset_destroy: 异常 - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "cluster_dataset_destroy: 未知异常\n");
    }
}

/**
 * C包装函数：使用 K-means 初始化 ClusterDataset
 * 
 * @param dataset_ptr ClusterDataset 指针
 * @param h_data 输入数据 [n_total_vectors, vector_dim]（host指针）
 * @param n_total_vectors 总向量数
 * @param vector_dim 向量维度
 * @param n_clusters cluster数量
 * @param h_objective 可选的输出：K-means目标函数值（可以为nullptr）
 * @param kmeans_iters K-means迭代次数
 * @param use_minibatch 是否使用minibatch算法
 * @param distance_mode 距离类型（0=L2, 1=COSINE）
 * @param seed 随机种子
 * @param batch_size 批处理大小
 * @param device_id GPU设备ID
 * @return 0表示成功，-1表示失败
 */
int cluster_dataset_init_with_kmeans(
    ClusterDataset* dataset,
    float* h_data,
    int n_total_vectors,
    int vector_dim,
    int n_clusters,
    float* h_objective,
    int kmeans_iters,
    int use_minibatch,
    int distance_mode,
    unsigned int seed,
    int batch_size,
    int device_id
) {
    if (!dataset) {
        fprintf(stderr, "cluster_dataset_init_with_kmeans: dataset_ptr 为 NULL\n");
        return -1;
    }
    
    if (!h_data) {
        fprintf(stderr, "cluster_dataset_init_with_kmeans: h_data 为 NULL\n");
        return -1;
    }
    
    try {
        DistanceType dist_type = (distance_mode == 0) ? L2_DISTANCE : COSINE_DISTANCE;
        
        // 使用 init_with_kmeans 函数，传入用户数据
        dataset->init_with_kmeans(
            n_total_vectors,
            vector_dim,
            n_clusters,
            h_objective,
            h_data,  // 传入用户数据
            kmeans_iters,
            use_minibatch != 0,
            dist_type,
            seed,
            batch_size,
            device_id
        );
        
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "cluster_dataset_init_with_kmeans: 异常 - %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "cluster_dataset_init_with_kmeans: 未知异常\n");
        return -1;
    }
}

/**
 * C包装函数：释放 ClusterDataset 占用的内存
 * 
 * @param dataset ClusterDataset 指针
 */
void cluster_dataset_release(ClusterDataset* dataset) {
    if (!dataset) {
        return;
    }
    try {
        dataset->release();
    } catch (const std::exception& e) {
        fprintf(stderr, "cluster_dataset_release: 异常 - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "cluster_dataset_release: 未知异常\n");
    }
}

/**
 * C包装函数：检查 ClusterDataset 是否有效
 * 
 * @param dataset ClusterDataset 指针
 * @return 1表示有效，0表示无效
 */
int cluster_dataset_is_valid(ClusterDataset* dataset) {
    if (!dataset) {
        return 0;
    }
    try {
        return dataset->is_valid() ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

/**
 * C包装函数：获取 ClusterDataset 的数据指针（用于读取）
 * 
 * @param dataset ClusterDataset 指针
 * @param reordered_data_out 输出：重排后的数据指针
 * @param reordered_indices_out 输出：重排后的索引指针
 * @param centroids_out 输出：聚类中心指针
 * @param cluster_info_out 输出：cluster信息结构
 * @param n_total_vectors_out 输出：总向量数
 * @param vector_dim_out 输出：向量维度
 * @return 0表示成功，-1表示失败
 */
int cluster_dataset_get_data(
    ClusterDataset* dataset,
    float** reordered_data_out,
    int** reordered_indices_out,
    float** centroids_out,
    long long** cluster_offsets_out,  // 改为 long long* 以匹配 ClusterInfo.offsets
    int** cluster_counts_out,
    int* n_clusters_out,
    int* n_total_vectors_out,
    int* vector_dim_out
) {
    if (!dataset) {
        fprintf(stderr, "cluster_dataset_get_data: dataset_ptr 为 NULL\n");
        return -1;
    }
    
    try {
        if (!dataset->is_valid()) {
            fprintf(stderr, "cluster_dataset_get_data: dataset 无效\n");
            return -1;
        }
        
        if (reordered_data_out) {
            *reordered_data_out = dataset->reordered_data;
        }
        if (reordered_indices_out) {
            *reordered_indices_out = dataset->reordered_indices;
        }
        if (centroids_out) {
            *centroids_out = dataset->centroids;
        }
        if (cluster_offsets_out) {
            *cluster_offsets_out = dataset->cluster_info.offsets;
        }
        if (cluster_counts_out) {
            *cluster_counts_out = dataset->cluster_info.counts;
        }
        if (n_clusters_out) {
            *n_clusters_out = dataset->cluster_info.k;
        }
        if (n_total_vectors_out) {
            *n_total_vectors_out = dataset->n_total_vectors;
        }
        if (vector_dim_out) {
            *vector_dim_out = dataset->vector_dim;
        }
        
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "cluster_dataset_get_data: 异常 - %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "cluster_dataset_get_data: 未知异常\n");
        return -1;
    }
}
}

