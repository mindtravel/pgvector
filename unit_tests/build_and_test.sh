#!/bin/bash

# 构建和测试脚本

echo "=== PGVector 单元测试构建脚本 ==="

# 检查是否在正确的目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "错误: 请在unit_tests目录下运行此脚本"
    exit 1
fi

# 创建构建目录
echo "创建构建目录..."
mkdir -p build
cd build

# 配置CMake
echo "配置CMake..."
# cmake .. -DCMAKE_BUILD_TYPE=Release
cmake .. -DCMAKE_BUILD_TYPE=Debug

# 编译
echo "编译项目..."
# make -j$(nproc)
# make test_integrated_coarse_fine -j$(nproc)
make test_integrated_coarse_fine_pipeline -j$(nproc)
# make test_fusion_cos_topk -j$(nproc)
# make test_fusion_cos_topk_fine
# make test_fusion_cos_topk_fine_v3_fixed_probe -j$(nproc)
# make test_fusion_cos_topk_fine_v5 -j$(nproc)
# make test_count_query_probes_kernel -j$(nproc)
# make test_fill_invalid_values_kernel -j$(nproc)
# make test_select_k -j$(nproc)

# 运行测试
echo "运行测试..."
# echo "=================================="
# echo "运行 vector_normalizer 测试..."
# ./test_vector_normalizer/test_vector_normalizer
# echo "=================================="
# echo "运行 print_cuda 测试..."
# ./test_print_cuda/test_print_cuda
# echo "=================================="
# echo "运行 matrix_multiply 测试..."
# ./test_matrix_multiply/test_matrix_multiply
# echo "=================================="
# echo "运行 cos_distance 测试..."
# ./test_cos_distance/test_cos_distance
# # echo "=================================="
# echo "运行 fusion_cos_topk 测试..."
# ./test_fusion_cos_topk/test_fusion_cos_topk $1
# echo "=================================="
# echo "运行 fusion_cos_topk_fine 测试..."
# ./test_fusion_cos_topk_fine/test_fusion_cos_topk_fine $1
# echo "=================================="
# echo "运行 warp sort 测试..."
# ./test_warpsort/test_warpsort
# echo "=================================="
# echo "运行 stream pass data 测试..."
# ./test_stream_pass_data/test_stream_pass_data
# echo "=================================="
# echo "运行 warp sort 测试..."
# ./test_bitonic/test_bitonic
# echo "=================================="
# echo "运行 final_topk 测试..."
# ./test_final_topk/test_final_topk
# echo "=================================="
# echo "运行 integrated_coarse_fine 测试..."
./test_integrated_coarse_fine_pipeline/test_integrated_coarse_fine_pipeline
# ./test_integrated_coarse_fine/test_integrated_coarse_fine
# echo "=================================="
# echo "运行 fusion_cos_topk_fine_v3_fixed_probe 测试..."
# ./test_fusion_cos_topk_fine_v3_fixed_probe/test_fusion_cos_topk_fine_v3_fixed_probe 
# echo "=================================="
# echo "运行 fusion_cos_topk_fine_v5 测试..."
# ./test_fusion_cos_topk_fine_v5/test_fusion_cos_topk_fine_v5
# echo "=================================="
# echo "运行 select_k 测试..."
# ./test_select_k/test_select_k 
# echo "=================================="

echo "所有测试完成！"
