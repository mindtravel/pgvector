#ifndef CPU_ARRAY_UTILS_H
#define CPU_ARRAY_UTILS_H

#include <thread>
#include <vector>
#include <random>
#include <algorithm>

/**
 * CPU数组工具函数库
 * 包含数组初始化和通用工具函数
 */

/**
 * 多线程并行初始化浮点数组（使用随机数填充）
 * 使用多线程并行初始化大数据数组，每个线程使用独立的随机数生成器
 * 
 * @param data 要初始化的数组指针
 * @param size 数组大小（元素个数）
 * @param seed 基础随机种子（默认1234），每个线程使用 seed + thread_id
 * @param min_val 随机数最小值（默认-1.0f）
 * @param max_val 随机数最大值（默认1.0f）
 * @param num_threads 线程数（默认0表示使用硬件并发数）
 */
inline void init_array_multithreaded(
    float* data,
    size_t size,
    unsigned int seed = 1234,
    float min_val = -1.0f,
    float max_val = 1.0f,
    int num_threads = 0
) {
    if (num_threads <= 0) {
        num_threads = std::max(1, (int)std::thread::hardware_concurrency());
    }
    
    const size_t chunk_size = (size + num_threads - 1) / num_threads;
    std::vector<std::thread> workers;
    
    for (int tid = 0; tid < num_threads; ++tid) {
        workers.emplace_back([=]() {
            const size_t start = tid * chunk_size;
            const size_t end = std::min(start + chunk_size, size);
            
            // 每个线程使用基于线程ID的seed，确保可重复性
            std::mt19937 rng(seed + tid);
            std::uniform_real_distribution<float> dist(min_val, max_val);
            
            for (size_t i = start; i < end; ++i) {
                data[i] = dist(rng);
            }
        });
    }
    
    // 等待所有线程完成
    for (auto& t : workers) {
        t.join();
    }
}

#endif // CPU_ARRAY_UTILS_H

