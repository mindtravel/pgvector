#ifndef VECTOR_ND_H
#define VECTOR_ND_H

#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <set>
#include <type_traits>
#include "../../cuda/dataset/cpu_array_utils.h"

// /**
//  * 分配向量组的空间
//  */
// void** malloc_vector_list(size_t n_batch, size_t n_dim, size_t elem_size) {
//     // 分配连续的数据内存块
//     void* data = malloc(n_batch * n_dim * elem_size);
//     if (data == NULL) return NULL;
    
//     // 分配行指针数组
//     void** vector_list = (void**)malloc(n_batch * sizeof(void*));
//     if (vector_list == NULL) {
//         free(data);
//         return NULL;
//     }
    
//     // 计算每个元素的字节偏移量并设置指针
//     for (size_t i = 0; i < n_batch; i++) {
//         vector_list[i] = (char*)data + i * n_dim * elem_size;
//     }
    
//     return vector_list;
// }

/**
 * 生成泛型二维数组
 * 内存布局：[ T* 指针列表 ] [ padding ] [ T 数据体 ]
 */
template <typename T>
T** malloc_vector_2D(int nx, int ny) {
    // 1. 计算尺寸
    size_t ptr_section_size = nx * sizeof(T*);
    size_t data_section_size = nx * ny * sizeof(T);
    
    // 2. 计算对齐
    size_t alignment = alignof(T); 
    size_t data_offset = (ptr_section_size + alignment - 1) & ~(alignment - 1);
    
    size_t total_size = data_offset + data_section_size;

    // 3. 分配单一连续内存块
    // 使用 char* 方便按字节操作指针偏移
    char* raw_memory = (char*)malloc(total_size);
    if (!raw_memory) return nullptr;

    // 4. 构建结构
    T** ptr_head = reinterpret_cast<T**>(raw_memory);
    T* data_head = reinterpret_cast<T*>(raw_memory + data_offset);

    // 5. 链接指针（这是实现 arr[i][j] 的关键）
    for (size_t i = 0; i < nx; ++i) {
        ptr_head[i] = data_head + (i * ny);
    }

    return ptr_head;
}
/**
* 生成泛型二维数组
* 内存布局：[ T* 指针列表 ] [ padding ] [ T 数据体 ]
*/
template <typename T>
T** malloc_vector_3D(int nx, int ny, int nz) {

    // 1. 计算各个部分的字节大小
    size_t size_L1 = nx * sizeof(T**);       // 第一层指针 (页索引)
    size_t size_L2 = nx * ny * sizeof(T*);   // 第二层指针 (行索引)
    size_t size_Data = nx * ny * nz * sizeof(T); // 数据体

    // 2. 计算对齐
    // 指针部分通常是自然对齐的，但数据部分需要根据类型 T 对齐
    size_t alignment = alignof(T);
    size_t ptr_total_size = size_L1 + size_L2;
    
    // 计算指针部分结束后，数据部分开始的偏移量（向上取整对齐）
    size_t data_offset = (ptr_total_size + alignment - 1) & ~(alignment - 1);
    size_t total_alloc_size = data_offset + size_Data;

    // 3. 一次性分配内存
    char* raw_memory = (char*)malloc(total_alloc_size);
    if (!raw_memory) return nullptr;

    // 4. 定位各个区域的起始指针
    T*** ptr_L1 = reinterpret_cast<T***>(raw_memory);
    T**  ptr_L2 = reinterpret_cast<T**>(raw_memory + size_L1);
    T*   data_head = reinterpret_cast<T*>(raw_memory + data_offset);

    // 5. 链接指针（Wiring）
    for (size_t i = 0; i < nx; ++i) {
        ptr_L1[i] = ptr_L2 + (i * ny);
    }

    // 5.2 连接 L2 -> Data
    // ptr_L2[j] 应该指向第 j 行在 Data 中的起始位置
    // 注意：这里的 j 是展平后的索引，范围是 0 到 nx*ny - 1
    for (size_t i = 0; i < nx * ny; ++i) {
        ptr_L2[i] = data_head + (i * nz);
    }
    return ptr_L1;
}



// template <typename T>
// T*** malloc_vector_3D(int nx, int ny, int nz) {
//     if (nx == 0 || ny == 0 || nz == 0) return nullptr;
//     T*** ptr_head = malloc_vector_3D<T>(nx, ny, nz);
//     return ptr_head;
// }

template <typename T>
T** generate_vector_2D(int nx, int ny) {
    if (nx == 0 || ny == 0) return nullptr;
    using RwT = std::remove_const_t<T>;
    RwT** ptr_head = malloc_vector_2D<RwT>(nx, ny);
    init_array_multithreaded(ptr_head[0], nx * ny);
    return (T**)ptr_head;
}

template <typename T>
T*** generate_vector_3D(int nx, int ny, int nz) {
    if (nx == 0 || ny == 0 || nz == 0) return nullptr;
    using RwT = std::remove_const_t<T>;
    RwT*** ptr_head = malloc_vector_3D<RwT>(nx, ny, nz);
    init_array_multithreaded(ptr_head[0][0], nx * ny * nz);
    return (T***)ptr_head;
}

/**
* 释放资源
* 注意：由于我们采用单块内存分配，只需要 free 数组头指针即可
*/
template <typename T>
void free_vector(const void* vector) {
    free(const_cast<void*>(vector));
}

/*
* 生成向量组
*/ 
template <typename T>
T** malloc_vector_list(int n_batch, int n_dim) {
    if (n_batch == 0 || n_dim == 0) return nullptr;
    T** vector_list = malloc_vector_2D<T>(n_batch, n_dim);
    return vector_list;
}

template <typename T>
T** generate_vector_list(int n_batch, int n_dim) {
    T** vector_list = generate_vector_2D<T>(n_batch, n_dim);
    return vector_list;
}

template <typename T>
void free_vector_list(T** vector_list) {
    free_vector<T>(vector_list);
}

#endif // VECTOR_ND_H