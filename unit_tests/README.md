# PGVector 单元测试

这个目录包含了PGVector项目的单元测试，使用CMake进行构建管理。

## 目录结构

```
unit_tests/
├── CMakeLists.txt              # 主CMake配置
├── build_and_test.sh          # 构建和测试脚本
├── common/                    # 公共头文件
│   └── test_utils.cuh         # 测试工具头文件
├── test_vector_normalizer/    # 向量归一化测试
│   ├── CMakeLists.txt
│   └── test_vector_normalizer.cu
├── test_print_cuda/           # CUDA打印功能测试
│   ├── CMakeLists.txt
│   └── test_print_cuda.cu
├── test_fusion_cosine_topk/   # 融合余弦Top-K测试
│   ├── CMakeLists.txt
│   └── test_fusion_cosine_topk.cu
└── README.md                  # 本文件
```

## 如何编译和测试

### 编译和测试所有项目

### 方法1：使用构建脚本（推荐）

```bash
cd /root/pgvector/unit_tests
./build_and_test.sh
```

### 方法2：手动构建

```bash
cd /root/pgvector/unit_tests

# 创建构建目录
mkdir -p build
cd build

# 配置CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

# 运行所有测试
./test_vector_normalizer/test_vector_normalizer
./test_print_cuda/test_print_cuda
./test_cosine_distance/test_cosine_distance
```

### 方法3：使用CMake测试框架

```bash
cd /root/pgvector/unit_tests
mkdir -p build && cd build
cmake ..
make
ctest --output-on-failure
```

## 测试内容

### test_vector_normalizer
1. **性能测试**：比较CPU和GPU归一化的性能
2. **大规模压力测试**：测试大量向量的异步归一化处理
3. **正确性验证**：确保归一化结果的正确性

### test_print_cuda
1. **CUDA内存操作测试**：测试GPU内存分配和拷贝
2. **打印功能测试**：测试CUDA设备数据的打印功能
3. **基础功能验证**：验证CUDA基础操作的正确性

### test_fusion_cosine_topk
1. **融合余弦距离测试**：测试融合优化的余弦距离计算
2. **Top-K选择测试**：测试Top-K最近邻选择算法
3. **性能对比测试**：比较融合版本与标准版本的性能
4. **正确性验证**：确保融合算法的计算结果正确

### test_matrix_multiply
1. **基本矩阵乘法测试**：测试GPU和CPU矩阵乘法的正确性
2. **单位矩阵测试**：验证特殊矩阵的乘法运算
3. **Alpha/Beta值测试**：测试不同缩放因子的矩阵乘法
4. **大规模压力测试**：测试大矩阵乘法的性能和正确性

### test_cosine_distance
1. **余弦距离计算测试**：测试GPU和CPU余弦距离计算的正确性
2. **Top-K选择测试**：测试融合余弦距离Top-K选择功能
3. **性能对比测试**：比较GPU和CPU实现的性能差异
4. **边界条件测试**：测试特殊输入情况下的正确性

## 依赖要求

- CUDA 11.0+
- CMake 3.18+
- 支持CUDA的GPU

## 注意事项

- 测试工具文件`test_utils.cuh`和`test_utils.cu`现在都位于`common/`目录中
- 测试框架完全独立，不依赖原位置的test_cuda目录
- 编译选项与原Makefile保持一致
- 使用相同的CUDA架构和优化选项

## 测试结果示例

```
=== 大规模数据压力测试 ===
测试规模: 1024 lists × 1024 vectors × 1024 dimensions
总内存使用量: 4096 MB
向量处理吞吐量: 1.56504e+06 vectors/second
内存带宽: 11.9403 GB/s
大规模压力测试完成 ✓
all test passed!
```