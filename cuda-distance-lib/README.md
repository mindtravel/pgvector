# CUDA Distance Library

## 说明

本库实现了基于 CUDA 的批量余弦距离和 L2 距离搜索。

### 文件结构

- `include/distance.h`：接口声明
- `src/distance_cuda.cu`：CUDA 实现（已合并余弦和 L2 距离）
- `src/distance.cpp`：接口声明
- `test/test_distance.cpp`：测试代码

### 编译

```bash
make
```

### 运行

```bash
./test_distance
```

### 备注

原 `cosine_distance.cu` 和 `l2_distance.cu` 已合并为 `distance_cuda.cu`，方便维护和扩展。