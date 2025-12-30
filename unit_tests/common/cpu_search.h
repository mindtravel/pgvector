#ifndef CPU_SEARCH_H
#define CPU_SEARCH_H

#include <vector>
#include <thread>
#include <algorithm>
#include <limits>
#include "../../cuda/kmeans/kmeans.cuh"
#include "cpu_distance.h"

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
 * @param out_index 输出索引 [n_query, topk]
 * @param out_dist 输出距离 [n_query, topk]
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
    int** out_index,               // [n_query, topk] 输出索引
    float** out_dist               // [n_query, topk] 输出距离
) {
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
            
            // 输出结果
            for (int i = 0; i < topk && i < (int)fine_buffer.size(); ++i) {
                out_index[qi][i] = fine_buffer[i].idx;
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

