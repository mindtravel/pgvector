# CUDA测试Makefile
CUDA_PATH ?= /usr/local/cuda
RAFT_ROOT ?= /root/miniconda3/envs/ann-benchmarks
NVCC = ccache $(CUDA_PATH)/bin/nvcc

CUDA_FLAGS = -arch=sm_70 -O3 -std=c++17 -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE --expt-relaxed-constexpr
# CUDA_FLAGS += -Xlinker -fuse-ld=lld# 使用新的链接器

CUDA_INCLUDE = -I$(RAFT_ROOT)/include -I$(RAFT_ROOT)/include/raft -I$(RAFT_ROOT)/include/cuvs -I$(RAFT_ROOT)/include/rmm -I$(RAFT_ROOT)/targets/x86_64-linux/include -I$(RAFT_ROOT)/include/rapids -I$(CUDA_PATH)/include -I../cuda -I.
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -L$(RAFT_ROOT)/lib -lraft -lcuvs -lrmm -lcudart -lcurand -lcublas -lcusparse
CXXFLAGS := -Wall -Wextra -std=c++17 -O2

# 设置环境变量
export LD_LIBRARY_PATH := $(RAFT_ROOT)/lib:$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH)

# 通用头文件包含 - 使用统一的头文件来减少重复解析
COMMON_HEADERS = -include ../cuda/pch.h

# 测试目标
TESTS = test_vector_normalizer test_cosine_distance test_cosine_distance_topk test_l2_distance test_matrix_multiply

# 默认目标
all: $(TESTS)

# 编译Normalizer测试
test_vector_normalizer: test_vector_normalizer.cu ../cuda/kernels.cu ../cuda/normalize.cu test_utils.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(COMMON_HEADERS) $(CUDA_LIBS) test_vector_normalizer.cu ../cuda/kernels.cu ../cuda/normalize.cu test_utils.cu -o $@

# 编译Cosine Distance测试
test_cosine_distance: test_cosine_distance.cu ../cuda/distances.cu ../cuda/kernels.cu test_utils.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(COMMON_HEADERS) $(CUDA_LIBS) test_cosine_distance.cu ../cuda/distances.cu ../cuda/kernels.cu test_utils.cu -o $@

# 编译Cosine Distance Top-K测试
test_cosine_distance_topk: test_cosine_distance_topk.cu ../cuda/distances.cu ../cuda/kernels.cu ../cuda/select_topk.cu test_utils.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(COMMON_HEADERS) $(CUDA_LIBS) test_cosine_distance_topk.cu ../cuda/distances.cu ../cuda/kernels.cu ../cuda/select_topk.cu test_utils.cu -o $@

# 编译L2 Distance测试
test_l2_distance: test_l2_distance.cu ../cuda/kernels.cu test_utils.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(COMMON_HEADERS) $(CUDA_LIBS) test_l2_distance.cu ../cuda/kernels.cu test_utils.cu -o $@

# 编译Matrix Multiply测试
test_matrix_multiply: test_matrix_multiply.cu ../cuda/matrix-multiply.cu ../cuda/kernels.cu test_utils.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(COMMON_HEADERS) $(CUDA_LIBS) test_matrix_multiply.cu ../cuda/matrix-multiply.cu ../cuda/kernels.cu test_utils.cu -o $@

# 运行所有测试
test: all
	@echo "运行所有CUDA测试..."
	@echo "=================================="
	@LD_LIBRARY_PATH=$(RAFT_ROOT)/lib:$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH) ./test_vector_normalizer
	@echo "=================================="
	@LD_LIBRARY_PATH=$(RAFT_ROOT)/lib:$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH) ./test_cosine_distance
	@echo "=================================="
	@LD_LIBRARY_PATH=$(RAFT_ROOT)/lib:$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH) ./test_cosine_distance_topk
	@echo "=================================="
	@LD_LIBRARY_PATH=$(RAFT_ROOT)/lib:$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH) ./test_l2_distance
	@echo "=================================="
	@LD_LIBRARY_PATH=$(RAFT_ROOT)/lib:$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH) ./test_matrix_multiply
	@echo "=================================="
	@echo "所有测试完成！"

# 运行单个测试
test-normalizer: test_vector_normalizer
	LD_LIBRARY_PATH=$(RAFT_ROOT)/lib:$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH) ./test_vector_normalizer

test-cosine: test_cosine_distance
	LD_LIBRARY_PATH=$(RAFT_ROOT)/lib:$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH) ./test_cosine_distance

test-cosine-topk: test_cosine_distance_topk
	LD_LIBRARY_PATH=$(RAFT_ROOT)/lib:$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH) ./test_cosine_distance_topk

test-l2: test_l2_distance
	LD_LIBRARY_PATH=$(RAFT_ROOT)/lib:$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH) ./test_l2_distance

test-matrix: test_matrix_multiply
	LD_LIBRARY_PATH=$(RAFT_ROOT)/lib:$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH) ./test_matrix_multiply

# 清理
clean:
	rm -f $(TESTS)

# 帮助信息
help:
	@echo "可用的目标："
	@echo "  all              - 编译所有测试"
	@echo "  test             - 编译并运行所有测试"
	@echo "  test-normalizer  - 运行向量归一化测试"
	@echo "  test-cosine      - 运行余弦距离测试"
	@echo "  test-cosine-topk      - 运行余弦最近邻测试"
	@echo "  test-l2          - 运行L2距离测试"
	@echo "  test-matrix      - 运行矩阵乘法测试"
	@echo "  clean            - 清理编译文件"
	@echo "  help             - 显示此帮助信息"

.PHONY: all test test-normalizer test-cosine test-cosine-topk test-l2 test-matrix clean help
