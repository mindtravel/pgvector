#ifndef CPU_ARRAY_UTILS_H
#define CPU_ARRAY_UTILS_H

#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <algorithm>
#include <type_traits>
/**
 * CPU数组工具函数库
 * 包含数组初始化和通用工具函数
 */


 
 /**
  * 泛型多线程并行初始化数组
  * 
  * @tparam T 数组元素类型 (支持 int, float, double 等算术类型)
  * @param data 要初始化的数组指针
  * @param size 数组大小
  * @param seed 随机种子
  * @param min_val 最小值
  * @param max_val 最大值
  * @param num_threads 线程数
  */
 template <typename T>
 inline void init_array_multithreaded(
     T* data,
     size_t size,
     unsigned int seed = 1234,
     T min_val = static_cast<T>(-10), 
     T max_val = static_cast<T>(10),
     int num_threads = 0
 ) {
     // 编译期检查：确保 T 是算术类型（整数或浮点数），避免传入结构体导致报错难看
     static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type (int, float, double, etc.)");
 
     // 1. 确定线程数
     if (num_threads <= 0) {
         unsigned int hw_concurrency = std::thread::hardware_concurrency();
         num_threads = std::max(1, (int)hw_concurrency);
     }
     
     // 2. 根据 T 的类型，在编译期决定使用哪种随机分布器
     // 如果是浮点型 -> uniform_real_distribution
     // 如果是整型   -> uniform_int_distribution
     using DistributionType = typename std::conditional<
         std::is_floating_point<T>::value,
         std::uniform_real_distribution<T>,
         std::uniform_int_distribution<T>
     >::type;
 
     // 计算每个线程的任务块大小
     const size_t chunk_size = (size + num_threads - 1) / num_threads;
     std::vector<std::thread> workers;
     workers.reserve(num_threads);
     
     // 3. 启动线程
     for (int tid = 0; tid < num_threads; ++tid) {
         workers.emplace_back([=]() {
             const size_t start = tid * chunk_size;
             // 防止越界
             if (start >= size) return;
             const size_t end = std::min(start + chunk_size, size);
             
             // 每个线程独立的 RNG，无锁且线程安全
             std::mt19937 rng(seed + tid);
             
             // 使用上面推导出的 DistributionType
             DistributionType dist(min_val, max_val);
             
             // 填充数据
             for (size_t i = start; i < end; ++i) {
                 data[i] = dist(rng);
             }
         });
     }
     
     // 4. 等待结束
     for (auto& t : workers) {
         if (t.joinable()) {
             t.join();
         }
     }
 }

#endif // CPU_ARRAY_UTILS_H

