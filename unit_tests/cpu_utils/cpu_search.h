#ifndef CPU_SEARCH_H
#define CPU_SEARCH_H

#include <vector>
#include <thread>
#include <algorithm>
#include <limits>
#include "../../cuda/kmeans/kmeans.cuh"
#include "cpu_distance.h"

// CPU版本余弦版本k近邻计算（用于验证）- 多线程版本
void cpu_cos_distance_topk(const float** query_vectors, const float** data_vectors, 
    int** topk_index, float** topk_dist,
    int n_query, int n_batch, int n_dim, int k) {

    // 计算每个向量的L2范数（平方和，用于后续计算）
    float* query_norms = (float*)malloc(n_query * sizeof(float));
    float* data_norms = (float*)malloc(n_batch * sizeof(float));

    // 计算query向量的平方和
    compute_squared_sums_batch(query_vectors, query_norms, n_query, n_dim);

    // 计算data向量的平方和
    compute_squared_sums_batch(data_vectors, data_norms, n_batch, n_dim);

    // 多线程处理：每个线程处理一部分query
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // 默认4个线程
        num_threads = std::min(num_threads, static_cast<unsigned int>(n_query));  // 不超过query数量

    // 每个线程处理的工作函数
    auto process_query_range = [&](int start_q, int end_q) {
        // 每个query需要自己的cos_pairs数组
        std::vector<std::pair<float, int>> cos_pairs(n_batch);

        for (int i = start_q; i < end_q; i++) {
            // 计算当前query与所有data向量的余弦距离
            for (int j = 0; j < n_batch; j++) {
                // 使用统一的 dot_product 函数
                float dot = dot_product(query_vectors[i], data_vectors[j], n_dim);
                float cos_sim;
                // 计算余弦相似度
                if (query_norms[i] < 1e-6f || data_norms[j] < 1e-6f)
                cos_sim = 0.0f;  // 如果任一向量接近零向量，相似度为0
                else
                cos_sim = 1.0f - (dot / sqrt(query_norms[i] * data_norms[j]));

                // 存储余弦相似度和对应的数据索引（直接使用 j）
                cos_pairs[j] = std::make_pair(cos_sim, j);
            }

            // 使用partial_sort只排序前k个元素，提高效率
            int topk_count = std::min(k, n_batch);
            std::partial_sort(cos_pairs.begin(), cos_pairs.begin() + topk_count, cos_pairs.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) 
            {
                if(a.first != b.first){
                    return a.first < b.first;  // 降序排序（距离小的在前）
                }
                else{
                    return a.second > b.second;
                }
            });

            // 提取前k个最相似的索引
            for (int j = 0; j < topk_count; ++j) {
                topk_index[i][j] = cos_pairs[j].second;
                topk_dist[i][j] = cos_pairs[j].first;
            }
        }
    };

    // 分配任务给各个线程
    std::vector<std::thread> threads;
    int queries_per_thread = (n_query + num_threads - 1) / num_threads;

    for (unsigned int t = 0; t < num_threads; t++) {
        int start_q = t * queries_per_thread;
        int end_q = std::min(start_q + queries_per_thread, n_query);
        if (start_q < n_query) {
            threads.emplace_back(process_query_range, start_q, end_q);
        }
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    free(query_norms);
    free(data_norms);
}

/**
 * CPU搜索参考实现
 * 用于验证GPU实现的正确性
 */

/**
 * CPU Coarse-Fine搜索实现（多线程）
 * 
 * 在聚好类的数据集上进行topk搜索：
 * 1. Coarse阶段：计算query到所有cluster中心的距离，选择前n_probes个最近的cluster
 * 2. Fine阶段：遍历选中的cluster中的所有向量，计算距离，选择topk
 * 
 * @param n_query 查询数量
 * @param dim 向量维度
 * @param k cluster数量
 * @param n_probes 粗筛选择的cluster数
 * @param topk 最终输出的topk数量
 * @param query_batch 查询向量 [n_query, dim]
 * @param reordered_data 重排后的数据 [n, dim]（按cluster连续存储）
 * @param centroids 聚类中心 [k, dim]
 * @param cluster_info cluster信息（offsets和counts）
 * @param distance_mode 距离类型（L2_DISTANCE 或 COSINE_DISTANCE）
 * @param out_index 输出索引 [n_query, topk]（原始索引）
 * @param out_dist 输出距离 [n_query, topk]
 * @param coarse_index 粗筛结果索引 [n_query, n_probes]，如果为nullptr则不输出
 * @param coarse_dist 粗筛结果距离 [n_query, n_probes]，如果为nullptr则不输出
 */
inline void cpu_coarse_fine_search(
    int n_query,
    int dim,
    int k,
    int n_probes,
    int topk,
    const float* query_batch,      // [n_query, dim]
    const float* reordered_data,   // [n, dim] 重排后的数据（按cluster连续存储）
    const float* centroids,        // [k, dim] 聚类中心
    const ClusterInfo& cluster_info, // cluster信息
    DistanceType distance_mode,    // L2_DISTANCE 或 COSINE_DISTANCE
    int** out_index,               // [n_query, topk] 输出索引（重排后的位置索引）
    float** out_dist,              // [n_query, topk] 输出距离
    int** coarse_index = nullptr, // [n_query, n_probes] 粗筛结果索引（可选）
    float** coarse_dist = nullptr  // [n_query, n_probes] 粗筛结果距离（可选）
) {
    // 计算总向量数
    int n_total = 0;
    if (cluster_info.k > 0) {
        n_total = (int)(cluster_info.offsets[cluster_info.k - 1] + cluster_info.counts[cluster_info.k - 1]);
    }
    struct Pair { float dist; int idx; };
    
    const unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
    auto worker = [&](int start, int end) {
        std::vector<Pair> tmp(k);
        std::vector<Pair> fine_buffer;
        
        for (int qi = start; qi < end; ++qi) {
            const float* query = query_batch + (size_t)qi * dim;
            
            // Coarse: 计算query到所有cluster中心的距离
            for (int cid = 0; cid < k; ++cid) {
                const float* center = centroids + (size_t)cid * dim;
                float dist = (distance_mode == L2_DISTANCE) 
                    ? l2_distance_squared(query, center, dim)
                    : cosine_distance(query, center, dim);
                tmp[cid] = {dist, cid};
            }
            
            // 选择前n_probes个最近的cluster
            std::partial_sort(tmp.begin(), tmp.begin() + n_probes, tmp.end(),
                            [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            
            // 输出粗筛结果（如果提供了输出数组）
            if (coarse_index != nullptr && coarse_dist != nullptr) {
                for (int probe_idx = 0; probe_idx < n_probes; ++probe_idx) {
                    coarse_index[qi][probe_idx] = tmp[probe_idx].idx;
                    coarse_dist[qi][probe_idx] = tmp[probe_idx].dist;
                }
            }
            
            // Fine: 遍历选中的cluster中的所有向量
            fine_buffer.clear();
            for (int probe_idx = 0; probe_idx < n_probes; ++probe_idx) {
                int cid = tmp[probe_idx].idx;
                long long cluster_offset = cluster_info.offsets[cid];
                int cluster_size = cluster_info.counts[cid];
                
                for (int vid = 0; vid < cluster_size; ++vid) {
                    int global_idx = (int)(cluster_offset + vid);
                    const float* vec = reordered_data + (size_t)global_idx * dim;
                    float dist = (distance_mode == L2_DISTANCE)
                        ? l2_distance_squared(query, vec, dim)
                        : cosine_distance(query, vec, dim);
                    fine_buffer.push_back({dist, global_idx});
                }
            }
            
            // 选择topk
            if (fine_buffer.size() > (size_t)topk) {
                std::nth_element(fine_buffer.begin(), fine_buffer.begin() + topk, fine_buffer.end(),
                               [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
                fine_buffer.resize(topk);
                std::sort(fine_buffer.begin(), fine_buffer.end(),
                         [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            } else {
                std::sort(fine_buffer.begin(), fine_buffer.end(),
                         [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            }
            
            // 输出结果（直接返回重排后的位置索引）
            for (int i = 0; i < topk && i < (int)fine_buffer.size(); ++i) {
                int reordered_pos = fine_buffer[i].idx;
                if (reordered_pos >= 0 && reordered_pos < n_total) {
                    out_index[qi][i] = reordered_pos;  // 直接返回重排后的位置索引
                } else {
                    out_index[qi][i] = -1;
                }
                out_dist[qi][i] = fine_buffer[i].dist;
            }
            // 填充剩余位置（如果候选数不足topk）
            for (int i = (int)fine_buffer.size(); i < topk; ++i) {
                out_index[qi][i] = -1;
                out_dist[qi][i] = std::numeric_limits<float>::max();
            }
        }
    };
    
    std::vector<std::thread> threads;
    int chunk = (n_query + num_threads - 1) / num_threads;
    for (unsigned t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end = std::min(n_query, start + chunk);
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    for (auto& t : threads) t.join();
}

/**
 * CPU Coarse-Fine搜索实现（多线程）
 * 
 * 在聚好类的数据集上进行topk搜索：
 * 1. Coarse阶段：计算query到所有cluster中心的距离，选择前n_probes个最近的cluster
 * 2. Fine阶段：遍历选中的cluster中的所有向量，计算距离，选择topk
 * 
 * @param n_query 查询数量
 * @param dim 向量维度
 * @param k cluster数量
 * @param n_probes 粗筛选择的cluster数
 * @param topk 最终输出的topk数量
 * @param query_batch 查询向量 [n_query, dim]
 * @param reordered_data 重排后的数据 [n, dim]（按cluster连续存储）
 * @param centroids 聚类中心 [k, dim]
 * @param cluster_info cluster信息（offsets和counts）
 * @param distance_mode 距离类型（L2_DISTANCE 或 COSINE_DISTANCE）
 * @param out_index 输出索引 [n_query, topk]（原始索引）
 * @param out_dist 输出距离 [n_query, topk]
 * @param coarse_index 粗筛结果索引 [n_query, n_probes]，如果为nullptr则不输出
 * @param coarse_dist 粗筛结果距离 [n_query, n_probes]，如果为nullptr则不输出
 */
 inline void cpu_coarse_fine_search_lookup(
    int n_query,
    int dim,
    int k,
    int n_probes,
    int topk,
    const float* query_batch,      // [n_query, dim]
    const float* reordered_data,   // [n, dim] 重排后的数据（按cluster连续存储）
    const float* centroids,        // [k, dim] 聚类中心
    const ClusterInfo& cluster_info, // cluster信息
    DistanceType distance_mode,    // L2_DISTANCE 或 COSINE_DISTANCE
    int** out_index,               // [n_query, topk] 输出索引（原始索引）
    float** out_dist,              // [n_query, topk] 输出距离
    int** coarse_index = nullptr, // [n_query, n_probes] 粗筛结果索引（可选）
    float** coarse_dist = nullptr,  // [n_query, n_probes] 粗筛结果距离（可选）
    const int* reordered_indices = nullptr  // [n] 回表映射数组（可选）
) {
    // 计算总向量数
    int n_total = 0;
    if (cluster_info.k > 0) {
        n_total = (int)(cluster_info.offsets[cluster_info.k - 1] + cluster_info.counts[cluster_info.k - 1]);
    }
    struct Pair { float dist; int idx; };
    
    const unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
    auto worker = [&](int start, int end) {
        std::vector<Pair> tmp(k);
        std::vector<Pair> fine_buffer;
        
        for (int qi = start; qi < end; ++qi) {
            const float* query = query_batch + (size_t)qi * dim;
            
            // Coarse: 计算query到所有cluster中心的距离
            for (int cid = 0; cid < k; ++cid) {
                const float* center = centroids + (size_t)cid * dim;
                float dist = (distance_mode == L2_DISTANCE) 
                    ? l2_distance_squared(query, center, dim)
                    : cosine_distance(query, center, dim);
                tmp[cid] = {dist, cid};
            }
            
            // 选择前n_probes个最近的cluster
            std::partial_sort(tmp.begin(), tmp.begin() + n_probes, tmp.end(),
                            [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            
            // 输出粗筛结果（如果提供了输出数组）
            if (coarse_index != nullptr && coarse_dist != nullptr) {
                for (int probe_idx = 0; probe_idx < n_probes; ++probe_idx) {
                    coarse_index[qi][probe_idx] = tmp[probe_idx].idx;
                    coarse_dist[qi][probe_idx] = tmp[probe_idx].dist;
                }
            }
            
            // Fine: 遍历选中的cluster中的所有向量
            fine_buffer.clear();
            for (int probe_idx = 0; probe_idx < n_probes; ++probe_idx) {
                int cid = tmp[probe_idx].idx;
                long long cluster_offset = cluster_info.offsets[cid];
                int cluster_size = cluster_info.counts[cid];
                
                for (int vid = 0; vid < cluster_size; ++vid) {
                    int global_idx = (int)(cluster_offset + vid);
                    const float* vec = reordered_data + (size_t)global_idx * dim;
                    float dist = (distance_mode == L2_DISTANCE)
                        ? l2_distance_squared(query, vec, dim)
                        : cosine_distance(query, vec, dim);
                    fine_buffer.push_back({dist, global_idx});
                }
            }
            
            // 选择topk
            if (fine_buffer.size() > (size_t)topk) {
                std::nth_element(fine_buffer.begin(), fine_buffer.begin() + topk, fine_buffer.end(),
                               [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
                fine_buffer.resize(topk);
                std::sort(fine_buffer.begin(), fine_buffer.end(),
                         [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            } else {
                std::sort(fine_buffer.begin(), fine_buffer.end(),
                         [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            }
            
            // 输出结果（进行回表操作，转换为原始索引）
            for (int i = 0; i < topk && i < (int)fine_buffer.size(); ++i) {
                int reordered_pos = fine_buffer[i].idx;
                if (reordered_pos >= 0 && reordered_pos < n_total) {
                    // 如果提供了回表映射数组，进行回表操作
                    if (reordered_indices != nullptr) {
                        out_index[qi][i] = reordered_indices[reordered_pos];  // 转换为原始索引
                    } else {
                        out_index[qi][i] = reordered_pos;  // 如果没有回表映射，直接返回重排后的位置索引
                    }
                } else {
                    out_index[qi][i] = -1;
                }
                out_dist[qi][i] = fine_buffer[i].dist;
            }
            // 填充剩余位置（如果候选数不足topk）
            for (int i = (int)fine_buffer.size(); i < topk; ++i) {
                out_index[qi][i] = -1;
                out_dist[qi][i] = std::numeric_limits<float>::max();
            }
        }
    };
    
    std::vector<std::thread> threads;
    int chunk = (n_query + num_threads - 1) / num_threads;
    for (unsigned t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end = std::min(n_query, start + chunk);
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    for (auto& t : threads) t.join();
}

#endif // CPU_SEARCH_H

