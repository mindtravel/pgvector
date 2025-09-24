/*
 * 简化的 int 索引版本 select_topk 测试
 * 避免复杂的依赖问题
 */

#include "../../cuda/select_topk.cuh"
#include "../../cuda/select_topk.cuh"
#include "../../test_cuda/test_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

void cuda_test_select_topk() {
    std::cout << "=== cuda_test_select_topk 测试 ===" << std::endl;
    
    // 参数设置
    int batch_size = 3;    // 批次大小
    int len = 5;           // 每行的长度
    int k = 2;             // 选择的top-k数量
    
    // 分配GPU内存
    float *d_in_val, *d_out_val;
    int *d_in_idx, *d_out_idx;
    
    size_t in_val_size = batch_size * len * sizeof(float);
    size_t in_idx_size = batch_size * len * sizeof(int);
    size_t out_val_size = batch_size * k * sizeof(float);
    size_t out_idx_size = batch_size * k * sizeof(int);
    
    cudaMalloc(&d_in_val, in_val_size);
    cudaMalloc(&d_in_idx, in_idx_size);
    cudaMalloc(&d_out_val, out_val_size);
    cudaMalloc(&d_out_idx, out_idx_size);
    
    // 准备测试数据
    std::vector<float> h_in_val(batch_size * len);
    std::vector<int> h_in_idx(batch_size * len);
    std::vector<float> h_out_val(batch_size * k);
    std::vector<int> h_out_idx(batch_size * k);

    std::mt19937 gen(1);
    std::uniform_int_distribution<int> dist(1, 100); // 生成 1 到 100 间的整数
    
    // 生成简单的测试数据
    for (int i = 0; i < batch_size * len; i++) {
        h_in_val[i] = float(i % 10) + dist(gen);  // 简单的模式
        h_in_idx[i] = 1000 + dist(gen);       // 简单的索引
    };
    
    // 复制数据到GPU
    cudaMemcpy(d_in_val, h_in_val.data(), in_val_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_idx, h_in_idx.data(), in_idx_size, cudaMemcpyHostToDevice);
    
    // 创建RAFT资源管理器
    raft::resources handle;
    
    // 创建输入矩阵视图
    auto in_val_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
        d_in_val, batch_size, len);
    auto in_idx_view = raft::make_device_matrix_view<const int, int64_t, raft::row_major>(
        d_in_idx, batch_size, len);
    
    // 创建输出矩阵视图
    auto out_val_view = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
        d_out_val, batch_size, k);
    auto out_idx_view = raft::make_device_matrix_view<int, int64_t, raft::row_major>(
        d_out_idx, batch_size, k);
    
    // 调用 select_k 函数
    std::cout << "调用 select_k 函数..." << std::endl;
    select_k(handle,
             in_val_view,
             std::make_optional(in_idx_view),
             out_val_view,
             out_idx_view,
             true,  // select_min = true
             false, // sorted = false
             SelectAlgo::kAuto,
             std::nullopt);
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA错误: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // 复制结果回CPU
    cudaMemcpy(h_out_val.data(), d_out_val, out_val_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_idx.data(), d_out_idx, out_idx_size, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < batch_size; i++) {
        COUT_ENDL("批次", i , ":");
        COUT_ENDL("  原始数据: ");
        
        // 显示原始数据
        for (int j = 0; j < len; j++) {
            int idx = i * len + j;
            COUT_VAL("(",h_in_val[idx], ",", h_in_idx[idx] ,")");
        }
        COUT_ENDL();
        
        // 显示选择的结果
        COUT_ENDL("  Top-", k, " 结果: ");
        for (int j = 0; j < k; j++) {
            int idx = i * k + j;
            COUT_VAL("(",h_out_val[idx], ",", h_out_idx[idx] ,")");
        }
        COUT_ENDL();
    }
    
    // 清理GPU内存
    cudaFree(d_in_val);
    cudaFree(d_in_idx);
    cudaFree(d_out_val);
    cudaFree(d_out_idx);
    
    COUT_ENDL("test finished!");
}

int main() {
    try {
        cuda_test_select_topk();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
