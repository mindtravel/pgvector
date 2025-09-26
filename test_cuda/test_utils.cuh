#ifndef TEST_UTILS_H
#define TEST_UTILS_H
#include <string>
#include <iostream>
#include <iomanip>
#include <cstdlib>
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


bool float_equal(float a, float b, float epsilon = 1e-5f);
bool float_equal_relative(float a, float b, float epsilon = 1e-5f);

bool matrix_equal(float* a, float* b, int rows, int cols, float epsilon = 1e-5f);

bool equal_2D_float(float** a, float** b, int rows, int cols, float epsilon = 1e-5f);
bool equal_2D_int(int** a, int** b, int rows, int cols);

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
    COUT_ENDL("[debug]:", name);

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
    COUT_ENDL("[debug]:", name);
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
    COUT_ENDL("[debug]:", name);

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
    COUT_ENDL("[debug]:", name);
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