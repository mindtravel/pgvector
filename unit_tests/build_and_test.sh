#!/bin/bash
set -e

echo "=== PGVector 单元测试构建脚本 ==="

if [ ! -f "CMakeLists.txt" ]; then
  echo "错误: 请在unit_tests目录下运行此脚本"
  exit 1
fi

echo "清理并创建构建目录..."
# rm -rf build
mkdir -p build

echo "配置CMake..."
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 \
  -DCMAKE_BUILD_TYPE=Debug

# 编译
echo "编译项目..."
cd build
# make -j$(nproc)
make test_kmeans -j$(nproc)
make test_ivfflat_search -j$(nproc)

# make test_ivf_search_coarse_fine -j$(nproc)
# make test_ivf_search_coarse_fine_pipeline -j$(nproc)
# make test_fusion_cos_topk -j$(nproc)
# make test_fusion_cos_topk_fine
# make test_count_query_probes_kernel -j$(nproc)
# make test_fill_invalid_values_kernel -j$(nproc)
# make test_select_k -j$(nproc)

# 运行测试
# echo "运行测试..."
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
# echo "运行 warp sort 测试..."
# ./test_warpsort/test_warpsort
# echo "=================================="
# echo "运行 warp sort 测试..."
# ./test_bitonic/test_bitonic
# echo "=================================="
# echo "运行 ivfflat_search_pipeline 测试..."
# ./test_ivfflat_search_pipeline/test_ivfflat_search_pipeline
# echo "=================================="
# echo "运行 kmeans 测试..."
# ./test_kmeans/test_kmeans
# echo "=================================="
# echo "运行 ivfflat_search 测试..."
./test_ivfflat_search/test_ivfflat_search
# echo "=================================="
# echo "运行 select_k 测试..."
# ./test_select_k/test_select_k 
# echo "=================================="

echo "所有测试完成！"
