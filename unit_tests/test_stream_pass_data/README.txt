# Stream Pass Data Test

## 测试目的
对比流式处理与直接处理在GPU数据传输方面的性能差异。

## 测试内容
- **流式处理**：使用CUDA Streams分批上传数据到GPU，处理后再分批下载
- **直接处理**：一次性上传所有数据到GPU，处理后再一次性下载
- **空核函数**：使用简单的数据传输核函数作为占位符
- **数据一致性验证**：确保生成的数据和回传的数据完全一致

## 测试规模
- **小规模**：3-50组 × 10-100个向量 × 8-128维
- **中等规模**：100-500组 × 100-500个向量 × 256-512维  
- **大规模**：1000组 × 1000个向量 × 1024维

## 输出指标
- pass rate: 测试通过率
- n_groups: 数据集组数
- n_vectors: 每组向量数量
- n_dim: 向量维度
- avg_stream_ms: 平均流式处理时间(ms)
- avg_direct_ms: 平均直接处理时间(ms)
- avg_time_ratio: 平均时间比例(直接/流式)
- memory_mb: 内存使用量(MB)

## 编译运行
```bash
cd unit_tests/test_stream_pass_data
mkdir build && cd build
cmake ..
make
./test_stream_pass_data
```

## 注意事项
- 当前使用空核函数进行测试，主要测量数据传输性能
- 后续可以替换`empty_kernel`为实际的业务逻辑
- 流式处理使用4个并发CUDA Streams
- **数据一致性检查**：使用`compare_1D`函数验证原始数据和回传数据的一致性
- 测试结果会导出为CSV文件供进一步分析
- 如果数据一致性检查失败，测试将被标记为FAIL
