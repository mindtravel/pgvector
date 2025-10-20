/**
 * Unit tests for Bitonic Sort class
 * 
 * 本测试专门测试 bitonic.cu 中的 Bitonic 类的正确性
 * 测试内容包括：
 * 1. Bitonic::sort() - 完整的 bitonic 排序
 * 2. Bitonic::merge() - bitonic 合并操作
 * 3. 不同的 Size 参数（1, 2, 4, 8）
 * 4. 升序和降序排序
 */

#include "pch.h"
#include "../common/test_utils.cuh"
#include <algorithm>
#include <vector>

#include "../../cuda/warpsortfilter/bitonic.cuh"
// #include "../../cuda/warpsort_utils.cuh"
#define EPSILON 1e-5f

using namespace pgvector::bitonic;

// ============================================================================
// Test Kernels
// ============================================================================

/**
 * 测试 Bitonic::sort() 方法
 * 
 * @tparam Size 每个线程持有的元素数量
 * @tparam Ascending 是否升序排序
 */
template<int Size, bool Ascending, typename T, typename IdxT>
__global__ void test_bitonic_sort_kernel(
    const T* __restrict__ input,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx,
    int total_size)
{
    const int lane = laneId();
    const int warp_width = kWarpSize;
    
    /* 每个线程持有 Size 个元素 */
    T keys[Size];
    IdxT indices[Size];
    
    /* 从全局内存加载数据到寄存器 */
    #pragma unroll
    for (int i = 0; i < Size; i++) {
        int idx = i * warp_width + lane;
        if (idx < total_size) {
            keys[i] = input[idx];
            indices[i] = static_cast<IdxT>(idx);
        } else {
            /* 填充哨兵值 */
            keys[i] = Ascending ? upper_bound<T>() : lower_bound<T>();
            indices[i] = static_cast<IdxT>(idx);
        }
    }
    
    /* 执行 bitonic sort */
    Bitonic<Size> sorter(Ascending, warp_width);
    sorter.sort(keys, indices);
    
    /* 将结果写回全局内存 */
    #pragma unroll
    for (int i = 0; i < Size; i++) {
        int idx = i * warp_width + lane;
        if (idx < total_size) {
            output_vals[idx] = keys[i];
            output_idx[idx] = indices[i];
        }
    }
}

/**
 * 测试 Bitonic::merge() 方法
 * 
 * 假设输入已经按反向排序好（ascending 时数据是降序，descending 时数据是升序）
 * 
 * @tparam Size 每个线程持有的元素数量
 * @tparam Ascending 目标排序方向
 */
template<int Size, bool Ascending, typename T, typename IdxT>
__global__ void test_bitonic_merge_kernel(
    const T* __restrict__ input,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx,
    int total_size)
{
    const int lane = laneId();
    const int warp_width = kWarpSize;
    
    /* 每个线程持有 Size 个元素 */
    T keys[Size];
    IdxT indices[Size];
    
    /* 从全局内存加载数据到寄存器 */
    #pragma unroll
    for (int i = 0; i < Size; i++) {
        int idx = i * warp_width + lane;
        if (idx < total_size) {
            keys[i] = input[idx];
            indices[i] = static_cast<IdxT>(idx);
        } else {
            /* 填充哨兵值 */
            keys[i] = Ascending ? upper_bound<T>() : lower_bound<T>();
            indices[i] = static_cast<IdxT>(idx);
        }
    }
    
    /* 执行 bitonic merge */
    Bitonic<Size> merger(Ascending, warp_width);
    merger.merge(keys, indices);
    
    /* 将结果写回全局内存 */
    #pragma unroll
    for (int i = 0; i < Size; i++) {
        int idx = i * warp_width + lane;
        if (idx < total_size) {
            output_vals[idx] = keys[i];
            output_idx[idx] = indices[i];
        }
    }
}

// ============================================================================
// CPU Reference Implementations
// ============================================================================

/**
 * CPU 参考实现：标准排序
 */
template<typename T, typename IdxT>
void cpu_sort(
    const T* input,
    T* output_vals,
    IdxT* output_idx,
    int n,
    bool ascending)
{
    std::vector<std::pair<T, IdxT>> pairs;
    pairs.reserve(n);
    
    for (int i = 0; i < n; i++) {
        pairs.push_back({input[i], static_cast<IdxT>(i)});
    }
    
    if (ascending) {
        std::sort(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
    } else {
        std::sort(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }
    
    for (int i = 0; i < n; i++) {
        output_vals[i] = pairs[i].first;
        output_idx[i] = pairs[i].second;
    }
}

// ============================================================================
// Test Functions
// ============================================================================

/**
 * 测试 Bitonic::sort() 方法
 */
template<int Size, bool Ascending>
bool test_bitonic_sort_impl(int n_elements)
{
    using T = float;
    using IdxT = int;
    
    const int total_size = Size * kWarpSize;
    COUT_ENDL("测试配置: Size=", Size, "Ascending=", Ascending, 
              "n_elements=", n_elements, "total_size=", total_size);
    
    /* 分配主机内存 */
    T* h_input = (T*)malloc(total_size * sizeof(T));
    T* h_gpu_vals = (T*)malloc(total_size * sizeof(T));
    T* h_cpu_vals = (T*)malloc(total_size * sizeof(T));
    IdxT* h_gpu_idx = (IdxT*)malloc(total_size * sizeof(IdxT));
    IdxT* h_cpu_idx = (IdxT*)malloc(total_size * sizeof(IdxT));
    
    /* 生成随机测试数据 */
    srand(42);
    for (int i = 0; i < n_elements; i++) {
        h_input[i] = static_cast<T>(rand() % 1000) / 10.0f;
    }
    /* 填充哨兵值 */
    for (int i = n_elements; i < total_size; i++) {
        h_input[i] = Ascending ? upper_bound<T>() : lower_bound<T>();
    }
    
    // print_1D("h_input", h_input, std::min(total_size, 64));

    /* 分配设备内存 */
    T *d_input, *d_output_vals;
    IdxT *d_output_idx;
    cudaMalloc(&d_input, total_size * sizeof(T));
    cudaMalloc(&d_output_vals, total_size * sizeof(T));
    cudaMalloc(&d_output_idx, total_size * sizeof(IdxT));
    CHECK_CUDA_ERRORS;
    
    /* 拷贝数据到设备 */
    cudaMemcpy(d_input, h_input, total_size * sizeof(T), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;
    
    /* 启动 GPU kernel (使用单个 warp) */
    dim3 block(32);  /* 单个 warp */
    dim3 grid(1);
    test_bitonic_sort_kernel<Size, Ascending, T, IdxT><<<grid, block>>>(
        d_input, d_output_vals, d_output_idx, total_size);
    CHECK_CUDA_ERRORS;
    
    /* 拷贝结果回主机 */
    cudaMemcpy(h_gpu_vals, d_output_vals, total_size * sizeof(T), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_idx, d_output_idx, total_size * sizeof(IdxT), 
               cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    
    /* CPU 参考实现 */
    cpu_sort<T, IdxT>(h_input, h_cpu_vals, h_cpu_idx, total_size, Ascending);
    
    /* 验证结果 */
    bool vals_match = compare_1D(h_gpu_vals, h_cpu_vals, total_size, EPSILON);
    // bool idx_match = compare_1D(h_gpu_idx, h_cpu_idx, total_size, 0);
    bool pass = vals_match;
    // bool pass = vals_match && idx_match;
    
    if (!pass) {
        COUT_ENDL("❌ 测试失败！");
        /* 调试信息 */
        // if (!vals_match) {
        //     COUT_ENDL("值不匹配");
        //     print_1D("GPU vals", h_gpu_vals, std::min(total_size, 64));
        //     print_1D("CPU vals", h_cpu_vals, std::min(total_size, 64));
        // }
        // if (!idx_match) {
        //     COUT_ENDL("索引不匹配");
        //     print_1D("GPU idx", h_gpu_idx, std::min(total_size, 64));
        //     print_1D("CPU idx", h_cpu_idx, std::min(total_size, 64));
        // }
    } else {
        COUT_ENDL("✅ 测试通过！");
    }
    
    /* 清理资源 */
    free(h_input);
    free(h_gpu_vals);
    free(h_cpu_vals);
    free(h_gpu_idx);
    free(h_cpu_idx);
    cudaFree(d_input);
    cudaFree(d_output_vals);
    cudaFree(d_output_idx);
    
    return pass;
}

/**
 * 测试 Bitonic::merge() 方法
 * 
 * 为了测试 merge，我们需要创建两个按反方向排序的数组
 * 
 * 注意：Bitonic sort 不是稳定排序，对于相同的值，其相对顺序可能改变
 * 因此我们只验证值的排序正确性，不验证索引的稳定性
 */
template<int Size, bool Ascending>
bool test_bitonic_merge_impl(int n_elements)
{
    using T = float;
    using IdxT = int;
    
    const int total_size = Size * kWarpSize;
    COUT_ENDL("测试配置: Size=", Size, "Ascending=", Ascending, 
              "n_elements=", n_elements, "total_size=", total_size);
    
    /* 分配主机内存 */
    T* h_input = (T*)malloc(total_size * sizeof(T));
    T* h_gpu_vals = (T*)malloc(total_size * sizeof(T));
    T* h_cpu_vals = (T*)malloc(total_size * sizeof(T));
    IdxT* h_gpu_idx = (IdxT*)malloc(total_size * sizeof(IdxT));
    IdxT* h_cpu_idx = (IdxT*)malloc(total_size * sizeof(IdxT));
    
    /* 生成唯一的测试数据（避免重复值）并按反方向排序 */
    srand(42);
    std::vector<std::pair<T, IdxT>> pairs;
    for (int i = 0; i < n_elements; i++) {
        /* 生成唯一的值：使用索引作为基础，加上小的随机扰动 */
        T val = static_cast<T>(i);
        pairs.push_back({val, static_cast<IdxT>(i)});
    }

    int mid = n_elements / 2;
    
    /* 前半部分：按目标方向排序 */
    if (Ascending) {
        std::sort(pairs.begin(), pairs.begin() + mid,
            [](const auto& a, const auto& b) { return a.first < b.first; });
    } else {
        std::sort(pairs.begin(), pairs.begin() + mid,
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }
    
    /* 后半部分：按相反方向排序 */
    if (Ascending) {
        std::sort(pairs.begin() + mid, pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    } else {
        std::sort(pairs.begin() + mid, pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    
    /* 将排序后的数据填充到 h_input */
    for (int i = 0; i < n_elements; i++) {
        h_input[i] = pairs[i].first;
    }
    /* 填充哨兵值 */
    for (int i = n_elements; i < total_size; i++) {
        h_input[i] = Ascending ? upper_bound<T>() : lower_bound<T>();
    }
    
    /* 分配设备内存 */
    T *d_input, *d_output_vals;
    IdxT *d_output_idx;
    cudaMalloc(&d_input, total_size * sizeof(T));
    cudaMalloc(&d_output_vals, total_size * sizeof(T));
    cudaMalloc(&d_output_idx, total_size * sizeof(IdxT));
    CHECK_CUDA_ERRORS;
    
    /* 拷贝数据到设备 */
    cudaMemcpy(d_input, h_input, total_size * sizeof(T), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;
    
    /* 启动 GPU kernel (使用单个 warp) */
    dim3 block(32);
    dim3 grid(1);
    test_bitonic_merge_kernel<Size, Ascending, T, IdxT><<<grid, block>>>(
        d_input, d_output_vals, d_output_idx, total_size);
    CHECK_CUDA_ERRORS;
    
    /* 拷贝结果回主机 */
    cudaMemcpy(h_gpu_vals, d_output_vals, total_size * sizeof(T), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_idx, d_output_idx, total_size * sizeof(IdxT), 
               cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    
    /* CPU 参考实现：merge 后应该是正确的排序 */
    cpu_sort<T, IdxT>(h_input, h_cpu_vals, h_cpu_idx, total_size, Ascending);
    
    /* 验证结果 */
    bool vals_match = compare_1D(h_gpu_vals, h_cpu_vals, total_size, EPSILON);
    bool idx_match = compare_1D(h_gpu_idx, h_cpu_idx, total_size, 0);
    bool pass = vals_match && idx_match;
    
    if (!pass) {
        COUT_ENDL("❌ FAIL");
        /* 调试信息 */
        if (!vals_match) {
            COUT_ENDL("值不匹配");
            print_1D("Input", h_input, std::min(total_size, 64));
            print_1D("GPU vals", h_gpu_vals, std::min(total_size, 64));
            print_1D("CPU vals", h_cpu_vals, std::min(total_size, 64));
        }
        if (!idx_match) {
            COUT_ENDL("索引不匹配");
            print_1D("GPU idx", h_gpu_idx, std::min(total_size, 64));
            print_1D("CPU idx", h_cpu_idx, std::min(total_size, 64));
        }
    } else {
        COUT_ENDL("✅ PASS");
    }
    
    /* 清理资源 */
    free(h_input);
    free(h_gpu_vals);
    free(h_cpu_vals);
    free(h_gpu_idx);
    free(h_cpu_idx);
    cudaFree(d_input);
    cudaFree(d_output_vals);
    cudaFree(d_output_idx);
    
    return pass;
}

// ============================================================================
// Main Test Runners
// ============================================================================

/**
 * 测试 Bitonic::sort() 的各种配置
 */
bool test_bitonic_sort()
{
    COUT_ENDL("\n=== 测试 Bitonic::sort() ===");
    
    bool all_pass = true;
    
    /* Test 1: Size=1, 升序 */
    COUT_ENDL("\n--- Test 1: Size=1, Ascending=true ---");
    all_pass &= test_bitonic_sort_impl<1, true>(32);
    
    /* Test 2: Size=1, 降序 */
    COUT_ENDL("\n--- Test 2: Size=1, Ascending=false ---");
    all_pass &= test_bitonic_sort_impl<1, false>(32);
    
    /* Test 3: Size=2, 升序 */
    COUT_ENDL("\n--- Test 3: Size=2, Ascending=true ---");
    all_pass &= test_bitonic_sort_impl<2, true>(64);
    
    /* Test 4: Size=2, 降序 */
    COUT_ENDL("\n--- Test 4: Size=2, Ascending=false ---");
    all_pass &= test_bitonic_sort_impl<2, false>(64);
    
    /*一个 warp 最多排序 64 个元素（一个线程才两个，怎么这么少），所以 buffer + queue 最多32个 */ 

    /* Test 5: Size=4, 升序 */
    COUT_ENDL("\n--- Test 5: Size=4, Ascending=true ---");
    all_pass &= test_bitonic_sort_impl<4, true>(128);
    
    /* Test 6: Size=4, 降序 */
    COUT_ENDL("\n--- Test 6: Size=4, Ascending=false ---");
    all_pass &= test_bitonic_sort_impl<4, false>(128);
    
    /* Test 7: Size=8, 升序 */
    COUT_ENDL("\n--- Test 7: Size=8, Ascending=true ---");
    all_pass &= test_bitonic_sort_impl<8, true>(256);
    
    /* Test 8: Size=8, 降序 */
    COUT_ENDL("\n--- Test 8: Size=8, Ascending=false ---");
    all_pass &= test_bitonic_sort_impl<8, false>(256);
    
    return all_pass;
}

/**
 * 测试 Bitonic::merge() 的各种配置
 */
bool test_bitonic_merge()
{
    COUT_ENDL("\n=== 测试 Bitonic::merge() ===");
    
    bool all_pass = true;
    
    /* Test 1: Size=1, 升序 */
    COUT_ENDL("\n--- Test 1: Size=1, Ascending=true ---");
    all_pass &= test_bitonic_merge_impl<1, true>(32);
    
    /* Test 2: Size=1, 降序 */
    COUT_ENDL("\n--- Test 2: Size=1, Ascending=false ---");
    all_pass &= test_bitonic_merge_impl<1, false>(32);
    
    /* Test 3: Size=2, 升序 */
    COUT_ENDL("\n--- Test 3: Size=2, Ascending=true ---");
    all_pass &= test_bitonic_merge_impl<2, true>(64);
    
    /* Test 4: Size=2, 降序 */
    COUT_ENDL("\n--- Test 4: Size=2, Ascending=false ---");
    all_pass &= test_bitonic_merge_impl<2, false>(64);
    
    /* Test 5: Size=4, 升序 */
    COUT_ENDL("\n--- Test 5: Size=4, Ascending=true ---");
    all_pass &= test_bitonic_merge_impl<4, true>(128);
    
    /* Test 6: Size=4, 降序 */
    COUT_ENDL("\n--- Test 6: Size=4, Ascending=false ---");
    all_pass &= test_bitonic_merge_impl<4, false>(128);
    
    /* Test 7: Size=8, 升序 */
    COUT_ENDL("\n--- Test 7: Size=8, Ascending=true ---");
    all_pass &= test_bitonic_merge_impl<8, true>(256);
    
    /* Test 8: Size=8, 降序 */
    COUT_ENDL("\n--- Test 8: Size=8, Ascending=false ---");
    all_pass &= test_bitonic_merge_impl<8, false>(256);
    
    return all_pass;
}

// ============================================================================
// Main
// ============================================================================

int main()
{
    COUT_ENDL("  Bitonic 排序类单元测试");

    /* 运行测试 */
    bool test1 = test_bitonic_sort();
    // bool test2 = test_bitonic_merge();
    
    return 0;
}
