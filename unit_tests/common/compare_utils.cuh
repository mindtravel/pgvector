#ifndef COMPARE_UTILS_H
#define COMPARE_UTILS_H
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <algorithm>
#include "output_macros.cuh"
#include "params_macros.cuh"
#include "metrics_collector.cuh"
#include "vector_nd.cuh"
#include "test_config.h"
/**
 * 比较数字：浮点数采用绝对误差
 */
template<typename T>
bool compare_numbers_relative(T a, T b, float epsilon) {
    if constexpr (std::is_floating_point_v<T>) {
        /* 浮点数 */
        return std::abs(a - b) * 2.0 / (std::abs(a) + std::abs(b)) < epsilon;
    } else {
        /* 整数 */
        return a == b;
    }
}

/**
* 比较数字：浮点数采用相对误差
*/
template<typename T>
bool compare_numbers(T a, T b, float epsilon = 1e-5) {
    if constexpr (std::is_floating_point_v<T>) {
        /* 浮点数 */
        return std::abs(a - b) < epsilon;
    } else {
        /* 整数 */
        return a == b;
    }
}

/* 
* 比较一维数组
*/
template<typename T>
bool compare_1D(T* a, T* b, int n, float epsilon = 1e-5f) {
    std::atomic<int> err_happens(0);
    std::mutex debug_mutex;
    std::atomic<bool> first_error(true);
    
    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), n);
    const int chunk_size = (n + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, n);
            
            for(int i = start; i < end; ++i){
                if (!compare_numbers(a[i], b[i], epsilon)) {
                    int err_count = err_happens.fetch_add(1) + 1;
                    
                    if(DEBUG){
                        std::lock_guard<std::mutex> lock(debug_mutex);
                        if(err_count == 1){
                            COUT_ENDL("mismatch!");
                            COUT_TABLE("i", "a[i]", "b[i]", "diff");
                        }
                        COUT_TABLE(i, a[i], b[i], a[i] - b[i]);
                    }
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    int total_errors = err_happens.load();
    if(total_errors == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(total_errors);
    return false;
}

template<typename T>
bool compare_2D(T* a, T* b, int nx, int ny, float epsilon = 1e-5f) {
    std::atomic<int> err_happens(0);
    std::mutex debug_mutex;
    
    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), nx);
    const int chunk_size = (nx + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, nx);
            
            for(int i = start; i < end; ++i){
                for (int j = 0; j < ny; j++) {
                    if (!compare_numbers(a[i][j], b[i][j], epsilon)) {
                        int err_count = err_happens.fetch_add(1) + 1;
                        
                        if(DEBUG){
                            std::lock_guard<std::mutex> lock(debug_mutex);
                            if(err_count == 1){
                                COUT_ENDL("mismatch!");
                                COUT_TABLE("i", "j", "a[i][j]", "b[i][j]", "diff");
                            }                    
                            COUT_TABLE(i, j, a[i][j], b[i][j], a[i][j] - b[i][j]);
                        }
                    }
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    int total_errors = err_happens.load();
    if(total_errors == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(total_errors);
    return false;
}

template<typename T>
bool compare_set_1D(T** a, T** b, int n, float epsilon = 1e-5f) {
    std::atomic<int> err_happens(0);
    std::mutex debug_mutex;
    
    std::sort(a, a + n);
    std::sort(b, b + n);

    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), n);
    const int chunk_size = (n + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, n);
            
            for(int i = start; i < end; ++i){
                if (!compare_numbers(a[i], b[i], epsilon)) {
                    int err_count = err_happens.fetch_add(1) + 1;
                    
                    if(DEBUG){
                        std::lock_guard<std::mutex> lock(debug_mutex);
                        if(err_count == 1){
                            COUT_ENDL("mismatch!");
                            COUT_TABLE("i", "a[i]", "b[i]", "diff");
                        }
                        COUT_TABLE(i, a[i], b[i], a[i] - b[i]);
                    }
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    int total_errors = err_happens.load();
    if(total_errors == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(total_errors);
    return false;
}

template<typename T>
bool compare_set_2D(T** a, T** b, int nx, int ny, float epsilon = 1e-5f) {
    std::atomic<int> err_happens(0);
    std::mutex debug_mutex;
    
    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), nx);
    const int chunk_size = (nx + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, nx);
            
            for(int i = start; i < end; ++i){
                std::sort(a[i], a[i] + ny);
                std::sort(b[i], b[i] + ny);

                for (int j = 0; j < ny; j++) {
                    if (!compare_numbers(a[i][j], b[i][j], epsilon)) {
                        int err_count = err_happens.fetch_add(1) + 1;
                        
                        if(DEBUG){
                            std::lock_guard<std::mutex> lock(debug_mutex);
                            if(err_count == 1){
                                COUT_ENDL("mismatch!");
                                COUT_TABLE("i", "j", "a[i][j]", "b[i][j]", "diff");
                            }                    
                            COUT_TABLE(i, j, a[i][j], b[i][j], a[i][j] - b[i][j]);
                        }
                    }
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    int total_errors = err_happens.load();
    if(total_errors == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(total_errors);
    return false;
}

template<typename T>
bool compare_set_2D_relative(T** a, T** b, int nx, int ny, float epsilon = 1e-5f) {
    std::atomic<int> err_happens(0);
    std::mutex debug_mutex;
    
    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), nx);
    const int chunk_size = (nx + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, nx);
            
            for(int i = start; i < end; ++i){
                std::sort(a[i], a[i] + ny);
                std::sort(b[i], b[i] + ny);

                for (int j = 0; j < ny; j++) {
                    if (!compare_numbers_relative(a[i][j], b[i][j], epsilon)) {
                        int err_count = err_happens.fetch_add(1) + 1;
                        
                        if(DEBUG){
                            std::lock_guard<std::mutex> lock(debug_mutex);
                            if(err_count == 1){
                                COUT_ENDL("mismatch!");
                                COUT_TABLE("i", "j", "a[i][j]", "b[i][j]", "diff");
                            }                    
                            COUT_TABLE(i, j, a[i][j], b[i][j], a[i][j] - b[i][j]);
                        }
                    }
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    int total_errors = err_happens.load();
    if(total_errors == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(total_errors);
    return false;
}

/**
* 计算两个集合（一维数组）中完全相同的元素个数（考虑epsilon容差）
* 使用双指针方式遍历已排序的a/b，统计相等元素个数
* 返回值为完全相同的元素个数
*/
template<typename T>
int count_equal_elements_set_1D(T** a, T** b, int n, float epsilon = 1e-5f) {
    std::sort(a, a + n);
    std::sort(b, b + n);
    int i = 0, j = 0;
    int count_equal = 0;
    while (i < n && j < n) {
        if (compare_numbers(a[i], b[j], epsilon)) {
            count_equal++;
            i++;
            j++;
        }
        else if (a[i] < b[j] - epsilon) {
            i++;
        }
        else {
            j++;
        }
    }
    if (DEBUG && count_equal != n)
        COUT_ENDL("Number of inequal elements:", n - count_equal, "out of", n);
    return count_equal == n;
}

/**
* 计算两个集合（二维数组）中完全相同的元素个数（考虑epsilon容差）
* 使用双指针方式遍历已排序的a/b，统计相等元素个数
* 返回值为完全相同的元素个数
*/
template<typename T>
int count_equal_elements_set_2D(T** a, T** b, int nx, int ny, float epsilon = 1e-5f) {
    std::atomic<int> count_equal(0);
    std::mutex debug_mutex;
    
    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), nx);
    const int chunk_size = (nx + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, nx);
            int local_count = 0;
            
            for(int i = start; i < end; ++i){
                std::sort(a[i], a[i] + ny);
                std::sort(b[i], b[i] + ny);

                int j = 0, k = 0;
                while (j < ny && k < ny) {
                    if (compare_numbers(a[i][j], b[i][k], epsilon)) {
                        local_count++;
                        j++;
                        k++;
                    }
                    else if (a[i][j] < b[i][k] - epsilon) {
                        j++;
                    }
                    else {
                        k++;
                    }
                }
            }
            
            count_equal.fetch_add(local_count);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    int total_count = count_equal.load();
    if (DEBUG && total_count != nx * ny) {
        std::lock_guard<std::mutex> lock(debug_mutex);
        COUT_ENDL("Number of inequal elements:", nx * ny - total_count, "out of", nx * ny);
    }
    return total_count == nx * ny;
}

/**
 * 不匹配信息结构体：记录发生不匹配的距离对的位置和内容
 */
struct MismatchInfo {
    int query_idx;      // 查询索引
    int pos;            // 在结果中的位置
    float cpu_dist;     // CPU版本的距离值
    float gpu_dist;     // GPU版本的距离值
    int cpu_idx;        // CPU版本的索引（如果可用）
    int gpu_idx;        // GPU版本的索引（如果可用）
    bool cpu_only;      // 是否只有CPU独有的
    
    MismatchInfo(int qi, int p, float c_dist, float g_dist, int c_idx = -1, int g_idx = -1, bool c_only = false)
        : query_idx(qi), pos(p), cpu_dist(c_dist), gpu_dist(g_dist), cpu_idx(c_idx), gpu_idx(g_idx), cpu_only(c_only) {}
};

/**
 * 智能比较函数：记录不匹配的距离对位置和内容
 * 使用排序+双指针，只记录CPU或GPU独有的结果
 * 返回不匹配信息的列表
 */
template<typename T>
std::vector<MismatchInfo> compare_set_2D_with_mismatches(
    T** cpu_dist, T** gpu_dist, 
    int** cpu_idx, int** gpu_idx,
    int nx, int ny, 
    float epsilon = 1e-5f,
    bool use_relative = false) {
    
    std::vector<MismatchInfo> mismatches;
    std::mutex mismatch_mutex;
    
    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), nx);
    const int chunk_size = (nx + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, nx);
            
            // 创建本地副本进行排序（避免修改原始数据）
            std::vector<std::pair<T, int>> cpu_sorted(ny);
            std::vector<std::pair<T, int>> gpu_sorted(ny);
            
            for(int i = start; i < end; ++i){
                // 准备排序数据（保留索引信息）
                for (int j = 0; j < ny; j++) {
                    cpu_sorted[j] = {cpu_dist[i][j], (cpu_idx ? cpu_idx[i][j] : -1)};
                    gpu_sorted[j] = {gpu_dist[i][j], (gpu_idx ? gpu_idx[i][j] : -1)};
                }
                
                // 按距离排序
                std::sort(cpu_sorted.begin(), cpu_sorted.end(), 
                    [](const auto& a, const auto& b) { return a.first < b.first; });
                std::sort(gpu_sorted.begin(), gpu_sorted.end(), 
                    [](const auto& a, const auto& b) { return a.first < b.first; });
                
                // 使用双指针找出只在CPU或GPU中出现的元素
                int cpu_ptr = 0, gpu_ptr = 0;
                int cpu_pos = 0, gpu_pos = 0;  // 记录在排序结果中的位置
                
                while (cpu_ptr < ny || gpu_ptr < ny) {
                    if (cpu_ptr >= ny) {
                        // CPU已遍历完，剩余的GPU元素都是GPU独有的
                        while (gpu_ptr < ny) {
                            std::lock_guard<std::mutex> lock(mismatch_mutex);
                            mismatches.emplace_back(
                                i, gpu_pos, 
                                cpu_sorted[gpu_ptr].first,  // CPU距离设为最大值表示不存在
                                gpu_sorted[gpu_ptr].first,
                                cpu_sorted[gpu_ptr].second,
                                gpu_sorted[gpu_ptr].second,
                                false
                            );
                            gpu_ptr++;
                            gpu_pos++;
                        }
                        break;
                    }
                    
                    if (gpu_ptr >= ny) {
                        // GPU已遍历完，剩余的CPU元素都是CPU独有的
                        while (cpu_ptr < ny) {
                            std::lock_guard<std::mutex> lock(mismatch_mutex);
                            mismatches.emplace_back(
                                i, cpu_pos,
                                cpu_sorted[cpu_ptr].first,
                                gpu_sorted[gpu_ptr].first,  // GPU距离设为最大值表示不存在
                                cpu_sorted[cpu_ptr].second,
                                gpu_sorted[gpu_ptr].second,  // GPU索引不存在
                                true
                            );
                            cpu_ptr++;
                            cpu_pos++;
                        }
                        break;
                    }
                    
                    // 比较当前CPU和GPU元素的距离
                    bool dist_match = use_relative
                        ? compare_numbers_relative(cpu_sorted[cpu_ptr].first, gpu_sorted[gpu_ptr].first, epsilon)
                        : compare_numbers(cpu_sorted[cpu_ptr].first, gpu_sorted[gpu_ptr].first, epsilon);
                    
                    if (dist_match) {
                        // 距离匹配，说明两个都查到了，排除（无论索引是否相同）
                        cpu_ptr++;
                        gpu_ptr++;
                        cpu_pos++;
                        gpu_pos++;
                    } else {
                        // 距离不匹配
                        if (cpu_sorted[cpu_ptr].first < gpu_sorted[gpu_ptr].first) {
                            // CPU距离更小，说明这是CPU独有的
                            std::lock_guard<std::mutex> lock(mismatch_mutex);
                            mismatches.emplace_back(
                                i, cpu_pos,
                                cpu_sorted[cpu_ptr].first,
                                gpu_sorted[gpu_ptr].first,  // GPU距离设为最大值表示不存在
                                cpu_sorted[cpu_ptr].second,
                                gpu_sorted[gpu_ptr].second,  // GPU索引不存在
                                true
                            );
                            cpu_ptr++;
                            cpu_pos++;
                        } else {
                            // GPU距离更小，说明这是GPU独有的
                            std::lock_guard<std::mutex> lock(mismatch_mutex);
                            mismatches.emplace_back(
                                i, gpu_pos,
                                cpu_sorted[cpu_ptr].first,  // CPU距离设为最大值表示不存在
                                gpu_sorted[gpu_ptr].first,
                                cpu_sorted[cpu_ptr].second,  // CPU索引不存在
                                gpu_sorted[gpu_ptr].second,
                                false
                            );
                            gpu_ptr++;
                            gpu_pos++;
                        }
                    }
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return mismatches;
}

/**
 * 智能比较函数（相对误差版本）
 */
template<typename T>
std::vector<MismatchInfo> compare_set_2D_relative_with_mismatches(
    T** cpu_dist, T** gpu_dist, 
    int** cpu_idx, int** gpu_idx,
    int nx, int ny, 
    float epsilon = 1e-5f) {
    return compare_set_2D_with_mismatches(cpu_dist, gpu_dist, cpu_idx, gpu_idx, nx, ny, epsilon, true);
}

#endif