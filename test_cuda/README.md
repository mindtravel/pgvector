# pgvector CUDA 单元测试

这个目录包含了pgvector中CUDA代码的单元测试，用于验证GPU加速功能的正确性和性能。

## 测试内容

### 1. VectorNormalizer 测试 (`test_vector_normalizer.cu`)
- **功能**: 向量L2范数归一化
- **测试用例**:
  - 基本归一化功能
  - 零向量处理
  - 单位向量处理
  - 大维度向量
  - 性能测试

### 2. CosineDistanceOp 测试 (`test_cosine_distance.cu`)
- **功能**: 余弦距离计算
- **测试用例**:
  - 基本余弦距离计算
  - 相同向量（距离为0）
  - 相反向量（距离为2）
  - 零向量处理
  - 大维度向量
  - 性能测试（与CPU版本对比）

### 3. L2DistanceOp 测试 (`test_l2_distance.cu`)
- **功能**: L2距离计算
- **测试用例**:
  - 基本L2距离计算
  - 相同向量（距离为0）
  - 单位向量（正交距离为√2）
  - 零向量处理
  - 大维度向量
  - 性能测试（与CPU版本对比）

## 编译和运行

### 环境要求
- CUDA 12.1+
- NVIDIA GPU (支持CUDA)
- GCC 或 Clang 编译器

### 编译所有测试
```bash
cd /root/pgvector/test_cuda
make all
```

### 运行所有测试
```bash
make test
```

### 运行单个测试
```bash
# 运行向量归一化测试
make test-normalizer

# 运行余弦距离测试
make test-cosine

# 运行L2距离测试
make test-l2
```

### 查看帮助
```bash
make help
```

### 清理编译文件
```bash
make clean
```

## 测试输出示例

### VectorNormalizer 测试输出
```
开始VectorNormalizer单元测试...

=== 测试1：基本归一化功能 ===
原始向量: 3 4 0 5
期望模长: 7.07107
归一化后向量: 0.424264 0.565685 0 0.707107
实际模长: 7.07107
归一化后向量模长: 1
✓ 基本归一化测试通过

=== 测试2：零向量处理 ===
零向量: 0 0 0 0
归一化后: 0 0 0 0
✓ 零向量测试通过

...

🎉 所有VectorNormalizer测试通过！
```

### CosineDistanceOp 测试输出
```
开始CosineDistanceOp单元测试...

=== 测试1：基本余弦距离计算 ===
向量A: 1 0 0 0
向量B: 0 1 0 0
GPU余弦距离: 1
CPU余弦距离: 1
✓ 基本余弦距离测试通过

=== 测试2：相同向量 ===
向量A: 1 2 3 4
向量B: 1 2 3 4
GPU余弦距离: 0
CPU余弦距离: 0
✓ 相同向量测试通过

...

🎉 所有CosineDistanceOp测试通过！
```

## 性能测试

每个测试都包含性能测试部分，会对比GPU和CPU版本的执行时间：

```
=== 测试6：性能测试 ===
GPU执行 1000 次耗时: 45ms
CPU执行 1000 次耗时: 120ms
GPU加速比: 2.67x
✓ 性能测试完成
```

## 注意事项

1. **GPU架构**: 默认使用`sm_60`架构，如果您的GPU不同，请修改Makefile中的`CUDA_FLAGS`
2. **内存管理**: 测试代码会自动管理GPU内存的分配和释放
3. **错误处理**: 如果CUDA操作失败，测试会抛出异常并显示错误信息
4. **精度**: 大维度向量测试允许稍大的数值误差（1e-4）

## 故障排除

### 编译错误
- 确保CUDA路径正确：`export CUDA_PATH=/usr/local/cuda`
- 检查GPU架构兼容性
- 确保已安装CUDA开发工具包

### 运行时错误
- 检查GPU是否可用：`nvidia-smi`
- 确保GPU内存足够
- 检查CUDA驱动版本

### 性能问题
- 调整测试中的向量维度和迭代次数
- 检查GPU利用率：`nvidia-smi -l 1`
- 考虑使用更优化的CUDA配置
