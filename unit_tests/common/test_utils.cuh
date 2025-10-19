#ifndef TEST_UTILS_H
#define TEST_UTILS_H
#include <string>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

#define DEBUG true

/**
 * 用宏简化计时语法
 */
#define MEASURE_MS(TESTNAME, ...) \
    do { \
        auto __measure_start = std::chrono::high_resolution_clock::now(); \
        __VA_ARGS__; \
        auto __measure_end = std::chrono::high_resolution_clock::now(); \
        auto __measure_dur = std::chrono::duration_cast<std::chrono::milliseconds>(__measure_end - __measure_start); \
        COUT_VAL((TESTNAME), __measure_dur.count(), "ms"); \
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
        auto __measure_dur = std::chrono::duration_cast<std::chrono::milliseconds>(__measure_end - __measure_start); \
        (TIMESPEND_VAR) = __measure_dur.count(); \
        /* COUT_VAL 宏用于在控制台打印日志，可以保留或移除 */ \
        /* 假设 COUT_VAL 宏的定义类似于： */ \
        /* #define COUT_VAL(name, val, unit) std::cout << (name) << ": " << (val) << " " << (unit) << std::endl */ \
        COUT_VAL((TESTNAME), (TIMESPEND_VAR), "ms"); \
    } while(0)

/**
 * 用宏简化cout语法
 */
inline void cout_endl() {
    std::cout << std::endl;    
}

template<typename T>
inline void cout_endl(T&& first) {
    std::cout << std::setprecision(5) << std::forward<T>(first) << std::endl;    
}

template<typename T, typename... Args>
inline void cout_endl(T&& first, Args&&... rest) {
    std::cout << std::setprecision(5) << std::forward<T>(first) << ' ';
    cout_endl(std::forward<Args>(rest)...);
}

inline void cout_impl() {
    std::cout << '\n';
}

template<typename T, typename... Args>
inline void cout_impl(T&& first, Args&&... rest) {
    std::cout << std::setprecision(5) << std::forward<T>(first) << ' ';
    cout_impl(std::forward<Args>(rest)...);
}

inline void table_impl() {
    std::cout << '\n';
}

template<typename T, typename... Args>
inline void table_impl(T&& first, Args&&... rest) {
    std::cout << std::setprecision(5) << std::forward<T>(first) << '\t';
    table_impl(std::forward<Args>(rest)...);
}

#define COUT_TABLE(...) do { table_impl(__VA_ARGS__); } while(0)
#define COUT_VAL(...) do { cout_impl(__VA_ARGS__); } while(0)
#define COUT_ENDL(...) do { cout_endl(__VA_ARGS__); } while(0)

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
            if(err_happens == 0){
                COUT_ENDL("mismatch!");
                COUT_TABLE("i", "a[i]", "b[i]", "diff");
            }

            err_happens ++;
            if(DEBUG == true)
                COUT_TABLE(i, a[i], b[i], a[i] - b[i]);
        }
    }
    if(err_happens == 0){
        COUT_ENDL("all match!");
        return true;
    }
    COUT_ENDL(err_happens);
    return false;
}

template<typename T>
bool compare_2D(T* a, T* b, int nx, int ny, float epsilon = 1e-5f) {
    int err_happens = 0;
    for(int i = 0; i < nx; ++i){
        for (int j = 0; j < ny; j++) {
            if (!compare_numbers(a[i][j], b[i][j], epsilon)) {
                if(err_happens == 0){
                    COUT_ENDL("mismatch!");
                    COUT_TABLE("i", "j", "a[i][j]", "b[i][j]", "diff");
                }
    
                err_happens ++;
                if(DEBUG == true)
                    COUT_TABLE(i, j, a[i][j], b[i][j], a[i][j] - b[i][j]);
            }
        }
    }
    if(err_happens == 0){
        COUT_ENDL("all match!");
        return true;
    }
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
            if(err_happens == 0){
                COUT_ENDL("mismatch!");
                COUT_TABLE("i", "a[i]", "b[i]", "diff");
            }

            err_happens ++;
            if(DEBUG == true)
                COUT_TABLE(i, a[i], b[i], a[i] - b[i]);
        }
    }
    if(err_happens == 0){
        COUT_ENDL("all match!");
        return true;
    }
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
                if(err_happens == 0){
                    COUT_ENDL("mismatch!");
                    COUT_TABLE("i", "j", "a[i][j]", "b[i][j]", "diff");
                }
    
                err_happens ++;
                if(DEBUG == true)
                    COUT_TABLE(i, j, a[i][j], b[i][j], a[i][j] - b[i][j]);
            }
        }
    }
    if(err_happens == 0){
        COUT_ENDL("all match!");
        return true;
    }
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
    if (count_equal != n)
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
    if (count_equal != nx * ny)
        COUT_ENDL("Number of inequal elements:", nx * ny - count_equal, "out of", nx * ny);
    return count_equal == nx * ny;
}

float** generate_vector_list(int n_batch, int n_dim);
float*** generate_large_scale_vectors(int n_lists, int n_batch, int n_dim);

/**
 * 打印host一维数组
 */
template <typename T>
void table_1D(std::string name, T* arr, int nx) {
    COUT_ENDL("[debug]:", name);
    COUT_TABLE("i", "value");
    for(int i = 0; i < nx; ++i){
        COUT_TABLE(i, arr[i]);
    }
    COUT_ENDL();
}

/**
 * 打印host二维数组
 */
template <typename T>
void table_2D(std::string name, T** arr, int nx, int ny) {
    COUT_ENDL("[debug]:", name);
    COUT_TABLE("i", "j", "value");
    for(int i = 0; i < nx; ++i){
        for(int j = 0; j < ny; ++j){
            COUT_TABLE(i, j, arr[i][j]);
        }
    }
    COUT_ENDL();
}

/**
 * 打印device一维数组
 */
template <typename T>
void table_cuda_1D(std::string name, T* d_arr, int nx) {
    COUT_ENDL("[debug on device]:", name);

    size_t arr_size = nx * sizeof(T);

    T* h_arr = (T*)malloc(arr_size);
    cudaMemcpy(h_arr, d_arr, arr_size, cudaMemcpyDeviceToHost);

    COUT_TABLE("i", "value");
    for(int i = 0; i < nx; ++i){
        COUT_TABLE(i, h_arr[i]);
    }
    COUT_ENDL();
    
    free(h_arr);  // 释放分配的内存
}

/**
 * 打印device二维数组
 */
template <typename T>
void table_cuda_2D(std::string name, T* d_arr, int nx, int ny) {
    COUT_ENDL("[debug on device]:", name);
    size_t arr_size = nx * ny * sizeof(T);

    T* h_arr = (T*)malloc(arr_size);
    cudaMemcpy(h_arr, d_arr, arr_size, cudaMemcpyDeviceToHost);

    COUT_TABLE("i", "j", "value");
    for(int i = 0; i < nx; ++i){
        for(int j = 0; j < ny; ++j){
            COUT_TABLE(i, j, h_arr[i*ny + j]);
        }
    }
    COUT_ENDL();
    
    free(h_arr);  // 释放分配的内存
}

/**
 * 打印host一维数组
 */
template <typename T>
void print_1D(std::string name, T* arr, int nx) {
    COUT_ENDL("[debug]:", name);
    for(int i = 0; i < nx; ++i){
        COUT_VAL(arr[i]);
    }
    COUT_ENDL();
}

/**
 * 打印host二维数组
 */
template <typename T>
void print_2D(std::string name, T** arr, int nx, int ny) {
    COUT_ENDL("[debug]:", name);
    for(int i = 0; i < nx; ++i){
        for(int j = 0; j < ny; ++j){
            COUT_VAL(arr[i][j]);
        }
        COUT_ENDL();
    }
    COUT_ENDL();
}

/**
 * 打印device一维数组
 */
template <typename T>
void print_cuda_1D(std::string name, T* d_arr, int nx) {
    COUT_ENDL("[debug on device]:", name);

    size_t arr_size = nx * sizeof(T);

    T* h_arr = (T*)malloc(arr_size);
    cudaMemcpy(h_arr, d_arr, arr_size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < nx; ++i){
        COUT_VAL(i, h_arr[i]);
    }
    COUT_ENDL();
    
    free(h_arr);  // 释放分配的内存
}

/**
 * 打印device二维数组
 */
template <typename T>
void print_cuda_2D(std::string name, T* d_arr, int nx, int ny) {
    COUT_ENDL("[debug on device]:", name);
    size_t arr_size = nx * ny * sizeof(T);

    T* h_arr = (T*)malloc(arr_size);
    cudaMemcpy(h_arr, d_arr, arr_size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < nx; ++i){
        for(int j = 0; j < ny; ++j){
            COUT_VAL(h_arr[i*ny + j]);
        }
        COUT_ENDL();
    }
    COUT_ENDL();
    
    free(h_arr);  // 释放分配的内存
}

void** malloc_vector_list(size_t n_batch, size_t n_dim, size_t elem_size);
void free_vector_list(void** vector_list);

#endif