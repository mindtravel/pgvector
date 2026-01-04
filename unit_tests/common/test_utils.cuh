#ifndef TEST_UTILS_H
#define TEST_UTILS_H
#include <string>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "../../cuda/pch.h"
#include "output_macros.cuh"
#include "params_macros.cuh"
#include "metrics_collector.cuh"
#include "vector_nd.cuh"
#include "compare_utils.cuh"


#define DEBUG true /* debug模式：用于寻找正确输出和测试函数输出的差异 */
#define QUIET false /* 静默模式：不打印日志（用于重复运行）*/
/**
 * @brief 用宏简化计时语法，并将结果（毫秒）存入指定的变量。
 * @param TESTNAME 字符串字面量，用于日志输出的测试名称。
 * @param ... 要执行计时的代码块。
 */
#define MEASURE_MS(TESTNAME, ...) \
    do { \
        auto __measure_start = std::chrono::high_resolution_clock::now(); \
        __VA_ARGS__; \
        auto __measure_end = std::chrono::high_resolution_clock::now(); \
        auto __measure_dur = std::chrono::duration_cast<std::chrono::microseconds>(__measure_end - __measure_start); \
        COUT_VAL((TESTNAME), ((double)__measure_dur.count()/1000.0) , "ms"); \
    } while(0)

/**
 * @brief 用宏简化计时语法，并将结果（毫秒）存入指定的变量。
 * @param TESTNAME 字符串字面量，用于日志输出的测试名称。
 * @param TIMESPEND_VAR 用于接收耗时结果的变量（例如 double, long long 等类型）。
 * @param ... 要执行计时的代码块。
 */
#define MEASURE_MS_AND_SAVE(TESTNAME, TIMESPEND_VAR, ...) \
    do { \
        auto __measure_start = std::chrono::high_resolution_clock::now(); \
        __VA_ARGS__; \
        auto __measure_end = std::chrono::high_resolution_clock::now(); \
        auto __measure_dur = std::chrono::duration_cast<std::chrono::microseconds>(__measure_end - __measure_start); \
        (TIMESPEND_VAR) = __measure_dur.count() / 1000.0; \
    } while(0);


inline bool check_pass(std::string test_names, bool pass){
    COUT_ENDL( (test_names), (pass) ? "✅ PASS" : "❌ FAIL");
    return pass;
}

bool compare_float(float a, float b, float epsilon = 1e-5f);
bool compare_float_relative(float a, float b, float epsilon = 1e-5f);

// 函数声明
void* generate_cluster_query_data(int* query_cluster_group, int n_query, int k, int batch_size);
void get_cluster_vector(int cluster_id, float** cluster_vector, int* vector_num);

#endif