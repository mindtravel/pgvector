#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "test_utils.cuh"
#include <stdlib.h>
#include <string.h>
/*
* 比较浮点数
*/
bool float_equal(float a, float b, float epsilon) {
    return std::abs(a - b) < epsilon;
}

bool float_equal_relative(float a, float b, float epsilon) {
    return std::abs(a - b) * 2.0 / (std::abs(a) + std::abs(b)) < epsilon;
}

/* 
* 比较向量
*/
bool vector_equal(float* a, float* b, int n, float epsilon = 1e-5f) {
    for (int i = 0; i < n; i++) {
        if (!float_equal(a[i], b[i], epsilon)) {
            return false;
        }
    }
    return true;
}

/**
 * 分配向量组的空间
 */
void** malloc_vector_list(size_t n_batch, size_t n_dim, size_t elem_size) {
    // 分配连续的数据内存块
    void* data = malloc(n_batch * n_dim * elem_size);
    if (data == NULL) return NULL;
    
    // 分配行指针数组
    void** vector_list = (void**)malloc(n_batch * sizeof(void*));
    if (vector_list == NULL) {
        free(data);
        return NULL;
    }
    
    // 计算每个元素的字节偏移量并设置指针
    for (size_t i = 0; i < n_batch; i++) {
        vector_list[i] = (char*)data + i * n_dim * elem_size;
    }
    
    return vector_list;
}

/**
 * 释放向量组的空间
 */
void free_vector_list(void** vector_list) {
    if (vector_list != NULL) {
        free(vector_list[0]);
        free(vector_list);
    }
}

/**
 * 比较矩阵（列主序）
 * 
 */
bool matrix_equal(float* a, float* b, int rows, int cols, float epsilon) {
    int err_happens = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (!float_equal(a[i], b[i], epsilon)) {
            if(err_happens == 0){
                COUT_ENDL("mismatch!");
                COUT_TABLE("i", "a[i]", "b[i]", "diff");
            }

            err_happens ++;
            if(DEBUG == true)
                COUT_TABLE(i, a[i], b[i], a[i] - b[i]);
            return false;
        }
    }
    COUT_ENDL(err_happens);
    return true;
}

/**
 * 比较矩阵（列主序）
 * 
 */
bool equal_2D_float(float** a, float** b, int rows, int cols, float epsilon) {
    int err_happens = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            if (!float_equal_relative(a[i][j], b[i][j], epsilon)) {
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
    COUT_ENDL(err_happens);
    return true;
}

bool equal_2D_int(int** a, int** b, int rows, int cols) {
    std::cout << "result" << std::endl;
    int err_happens = 0;
    if(DEBUG == true)
        std::cout << "i" <<  "\t"  << "j" << "\t" << "a[i][j]" <<  "\t"  << "b[i][j]" <<  "\t" << "diff" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if(DEBUG == true)
                std::cout << std::setprecision(5) << i <<  "\t"  << j << "\t" << a[i][j] <<  "\t"  << b[i][j] <<  "\t" << a[i][j] - b[i][j] << std::endl;
            if (!a[i][j] == b[i][j]) {
                err_happens ++;
                // return false;
            }
        }
    }
    std::cout << err_happens << std::endl;
    return true;
}

/*
* 生成向量组
*/ 
float** generate_vector_list(int n_batch, int n_dim) {
    // 分配连续的内存
    float** vector_list = (float**)malloc_vector_list(n_batch, n_dim, sizeof(float));

    for (int i = 0; i < n_batch; i++) {
        for (int j = 0; j < n_dim; j++) {
            // vector_list[i][j] = 1.0f;
            // vector_list[i][j] = (float)i + (float)j;
            // vector_list[i][j] = (float)rand() / RAND_MAX * 20.0f - 10.0f;
            vector_list[i][j] = (float)(i+j + 1.0f);
            // if(i == 1 && j == 0)
            //     vector_list[i][j] = 1.0f;
            // else
            //     vector_list[i][j] = 0.0f;
        }
    }    

    return vector_list;
}

/*
* 生成大规模向量组 (1024*1024*512)
*/ 
float*** generate_large_scale_vectors(int n_lists, int n_batch, int n_dim) {
    std::cout << "生成大规模数据: " << n_lists << " lists, " 
              << n_batch << " vectors per list, " 
              << n_dim << " dimensions" << std::endl;
    
    // 分配三级指针结构
    float*** vector_lists = (float***)malloc(n_lists * sizeof(float**));
    
    for (int list_id = 0; list_id < n_lists; list_id++) {
        // 为每个list分配连续内存
        vector_lists[list_id] = (float**)malloc_vector_list(n_batch, n_dim, sizeof(float));
        
        // 初始化数据
        for (int i = 0; i < n_batch; i++) {
            for (int j = 0; j < n_dim; j++) {
                // 生成随机数据 [-10, 10]
                vector_lists[list_id][i][j] = (float)rand() / RAND_MAX * 20.0f - 10.0f;
            }
        }
        
        if (list_id % 100 == 0) {
            std::cout << "已生成 " << list_id << "/" << n_lists << " lists" << std::endl;
        }
    }
    
    std::cout << "大规模数据生成完成 ✓" << std::endl;
    return vector_lists;
}
