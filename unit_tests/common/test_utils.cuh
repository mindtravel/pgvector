#ifndef TEST_UTILS_H
#define TEST_UTILS_H
#include <string>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "output_macros.cuh"
#include "params_macros.cuh"
#include "metrics_collector.cuh"

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

void _check_cuda_last_error(const char *file, int line);
#define CHECK_CUDA_ERRORS _check_cuda_last_error(__FILE__, __LINE__);/*用于捕捉CUDA函数错误信息的宏*/

bool compare_float(float a, float b, float epsilon = 1e-5f);
bool compare_float_relative(float a, float b, float epsilon = 1e-5f);

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
    int err_happens = 0;
    for(int i = 0; i < n; ++i){
        if (!compare_numbers(a[i], b[i], epsilon)) {
            if(DEBUG){
                if(err_happens == 0){
                    COUT_ENDL("mismatch!");
                    COUT_TABLE("i", "a[i]", "b[i]", "diff");
                }
                COUT_TABLE(i, a[i], b[i], a[i] - b[i]);
            }

            err_happens ++;
        }
    }
    if(err_happens == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(err_happens);
    return false;
}

template<typename T>
bool compare_2D(T* a, T* b, int nx, int ny, float epsilon = 1e-5f) {
    int err_happens = 0;
    for(int i = 0; i < nx; ++i){
        for (int j = 0; j < ny; j++) {
            if (!compare_numbers(a[i][j], b[i][j], epsilon)) {
                if(DEBUG){
                    if(err_happens == 0){
                        COUT_ENDL("mismatch!");
                        COUT_TABLE("i", "j", "a[i][j]", "b[i][j]", "diff");
                    }                    
                    COUT_TABLE(i, j, a[i][j], b[i][j], a[i][j] - b[i][j]);
                }
    
                err_happens ++;
            }
        }
    }
    if(err_happens == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(err_happens);
    return false;
}

template<typename T>
bool compare_set_1D(T** a, T** b, int n, float epsilon = 1e-5f) {
    int err_happens = 0;
    std::sort(a, a + n);
    std::sort(b, b + n);

    for(int i = 0; i < n; ++i){
        if (!compare_numbers(a[i], b[i], epsilon)) {
            if(DEBUG){
                if(err_happens == 0){
                    COUT_ENDL("mismatch!");
                    COUT_TABLE("i", "a[i]", "b[i]", "diff");
                }
                COUT_TABLE(i, a[i], b[i], a[i] - b[i]);
            }

            err_happens ++;
        }
    }
    if(err_happens == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(err_happens);
    return false;
}

template<typename T>
bool compare_set_2D(T** a, T** b, int nx, int ny, float epsilon = 1e-5f) {
    int err_happens = 0;
    for(int i = 0; i < nx; ++i){
        std::sort(a[i], a[i] + ny);
        std::sort(b[i], b[i] + ny);

        for (int j = 0; j < ny; j++) {
            if (!compare_numbers(a[i][j], b[i][j], epsilon)) {
                if(DEBUG){
                    if(err_happens == 0){
                        COUT_ENDL("mismatch!");
                        COUT_TABLE("i", "j", "a[i][j]", "b[i][j]", "diff");
                    }                    
                    COUT_TABLE(i, j, a[i][j], b[i][j], a[i][j] - b[i][j]);
                }
    
                err_happens ++;
            }
        }
    }
    if(err_happens == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(err_happens);
    return false;
}

template<typename T>
bool compare_set_2D_relative(T** a, T** b, int nx, int ny, float epsilon = 1e-5f) {
    int err_happens = 0;
    for(int i = 0; i < nx; ++i){
        std::sort(a[i], a[i] + ny);
        std::sort(b[i], b[i] + ny);

        for (int j = 0; j < ny; j++) {
            if (!compare_numbers_relative(a[i][j], b[i][j], epsilon)) {
                if(DEBUG){
                    if(err_happens == 0){
                        COUT_ENDL("mismatch!");
                        COUT_TABLE("i", "j", "a[i][j]", "b[i][j]", "diff");
                    }                    
                    COUT_TABLE(i, j, a[i][j], b[i][j], a[i][j] - b[i][j]);
                }
    
                err_happens ++;
            }
        }
    }
    if(err_happens == 0){
        return true;
    }
    if(DEBUG)
        COUT_ENDL(err_happens);
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
    int count_equal = 0;

    for(int i = 0; i < nx; ++i){
        std::sort(a[i], a[i] + ny);
        std::sort(b[i], b[i] + ny);

        int j = 0, k = 0;
        while (j < ny && k < ny) {
            if (compare_numbers(a[i][j], b[i][k], epsilon)) {
                count_equal++;
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
    if (DEBUG && count_equal != nx * ny)
        COUT_ENDL("Number of inequal elements:", nx * ny - count_equal, "out of", nx * ny);
    return count_equal == nx * ny;
}

float** generate_vector_list(int n_batch, int n_dim);
float*** generate_large_scale_vectors(int n_lists, int n_batch, int n_dim);

void** malloc_vector_list(size_t n_batch, size_t n_dim, size_t elem_size);
void free_vector_list(void** vector_list);

/**
 * 计算向量的平方和（sum of squares）
 * @param vector 向量数据
 * @param n_dim 向量维度
 * @return 平方和
 */
inline float compute_squared_sum(const float* vector, int n_dim) {
    float sum = 0.0f;
    for (int d = 0; d < n_dim; d++) {
        sum += vector[d] * vector[d];
    }
    return sum;
}

/**
 * 计算向量的 L2 范数（L2 norm）
 * @param vector 向量数据
 * @param n_dim 向量维度
 * @return L2 范数（sqrt(sum of squares)）
 */
inline float compute_l2_norm(const float* vector, int n_dim) {
    float sum = compute_squared_sum(vector, n_dim);
    return sqrtf(sum);
}

/**
 * 批量计算向量的平方和
 * @param vectors 向量数组 [n_batch][n_dim]
 * @param squared_sums 输出的平方和数组 [n_batch]
 * @param n_batch 向量数量
 * @param n_dim 向量维度
 */
inline void compute_squared_sums_batch(float** vectors, float* squared_sums, int n_batch, int n_dim) {
    for (int i = 0; i < n_batch; i++) {
        squared_sums[i] = compute_squared_sum(vectors[i], n_dim);
    }
}

/**
 * 批量计算向量的 L2 范数
 * @param vectors 向量数组 [n_batch][n_dim]
 * @param norms 输出的 L2 范数数组 [n_batch]
 * @param n_batch 向量数量
 * @param n_dim 向量维度
 */
inline void compute_l2_norms_batch(float** vectors, float* norms, int n_batch, int n_dim) {
    for (int i = 0; i < n_batch; i++) {
        norms[i] = compute_l2_norm(vectors[i], n_dim);
    }
}

// 函数声明
void* generate_cluster_query_data(int* query_cluster_group, int n_query, int k, int batch_size);
void get_cluster_vector(int cluster_id, float** cluster_vector, int* vector_num);

#endif