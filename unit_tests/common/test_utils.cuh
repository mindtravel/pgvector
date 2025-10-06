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
    return true;
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
    return true;
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
    return true;
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
    return true;
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