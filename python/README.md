# IVFTensor Python Bindings

Python 绑定用于将 CUDA 实现的 IVF-Flat 搜索功能暴露给 Python。

## 构建

### 前置要求

1. **CUDA Toolkit** (>= 11.0)
2. **CMake** (>= 3.18)
3. **pybind11**: `pip install pybind11` 或通过 CMake 查找
4. **Python 开发头文件**

### 构建步骤

```bash
cd /home/diy/lzx/pgvector
mkdir -p build
cd build
cmake ..
make IVFTensor  # 构建 Python 模块
make ivf_search_shared  # 构建共享库（供 ctypes 使用）
```

构建完成后：
- Python 模块: `python/build/IVFTensor.so` (Linux) 或 `IVFTensor.pyd` (Windows)
- 共享库: `build/libivf_search.so` (Linux) 或 `libivf_search.dll` (Windows)

## 使用

### 方式 1: 使用 pybind11 模块（推荐）

```python
import sys
sys.path.insert(0, '/home/diy/lzx/pgvector/python/build')
import IVFTensor
import numpy as np

# 创建数据集并执行 K-means 聚类
dataset = IVFTensor.ClusterDataset()
data = np.random.rand(10000, 96).astype(np.float32)
dataset.init_with_kmeans(
    data,
    n_clusters=100,
    kmeans_iters=20,
    use_minibatch=False,
    distance_mode=IVFTensor.DISTANCE_COSINE
)

# 获取聚类结果
reordered_data, reordered_indices, centroids, cluster_offsets, cluster_counts, n_clusters = dataset.get_data()

# 创建搜索器
searcher = IVFTensor.IVFSearcher()

# 准备查询
queries = np.random.rand(10, 96).astype(np.float32)

# 执行搜索
distances, indices = searcher.search(
    queries,
    cluster_counts,  # cluster sizes
    reordered_data.flatten(),  # cluster vectors (flattened)
    centroids.flatten(),  # cluster centers (flattened)
    n_probes=10,
    k=10,
    distance_mode=IVFTensor.DISTANCE_COSINE
)

print(f"Search results: {indices.shape}")
```

### 方式 2: 使用 ctypes（通过共享库）

```python
import ctypes
import numpy as np

# 加载共享库
lib = ctypes.CDLL('/home/diy/lzx/pgvector/build/libivf_search.so')

# 定义函数签名
lib.batch_search_pipeline_wrapper.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_query_batch
    ctypes.POINTER(ctypes.c_int),    # d_cluster_size
    ctypes.POINTER(ctypes.c_float),  # d_cluster_vectors
    ctypes.POINTER(ctypes.c_float),  # d_cluster_centers
    ctypes.POINTER(ctypes.c_int),    # d_initial_indices
    ctypes.POINTER(ctypes.c_float),  # d_topk_dist
    ctypes.POINTER(ctypes.c_int),    # d_topk_index
    ctypes.c_int,  # n_query
    ctypes.c_int,  # n_dim
    ctypes.c_int,  # n_total_cluster
    ctypes.c_int,  # n_total_vectors
    ctypes.c_int,  # n_probes
    ctypes.c_int,  # k
    ctypes.c_int   # distance_mode
]

# 使用 CUDA 内存管理函数分配 GPU 内存并调用
# ... (需要额外的 CUDA 内存管理代码)
```

## 在 ann-benchmarks 中使用

Python 模块会自动被 `ann-benchmarks/ann_benchmarks/algorithms/ivf_tensor/module.py` 加载。

确保构建的 `.so` 文件在以下路径之一：
- `/home/diy/lzx/pgvector/python/build/IVFTensor.so`
- `/home/diy/lzx/pgvector/build/libivf_search.so`
- 当前工作目录

## 文件说明

- `ivf_tensor_bindings.cpp`: pybind11 Python 绑定代码
- `dataset_wrapper.cpp`: ClusterDataset 的 C 包装函数
- `ivf_search_wrapper.cpp`: IVF 搜索的 C 包装函数（已存在）

## 注意事项

1. **GPU 内存管理**: Python 绑定会自动管理 GPU 内存，但需要确保有足够的显存
2. **数据格式**: 所有输入数据必须是 `float32` 类型的 numpy 数组
3. **距离模式**: `0` = L2距离, `1` = Cosine距离
4. **线程安全**: 当前实现不是线程安全的，不要在多个线程中同时使用同一个对象




