# CUDA测试Makefile
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_FLAGS = -arch=sm_60 -O3 -std=c++14
CUDA_INCLUDE = -I$(CUDA_PATH)/include -I../cuda -I.
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas

# 测试目标
TESTS = test_vector_normalizer test_cosine_distance test_l2_distance test_matrix_multiply

# 默认目标
all: $(TESTS)

# 编译VectorNormalizer测试
test_vector_normalizer: test_vector_normalizer.cu ../cuda/kernels.cu ../cuda/normalize.cu test_utils.cpp
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(CUDA_LIBS) $^ -o $@

# 编译CosineDistanceOp测试
test_cosine_distance: test_cosine_distance.cu ../cuda/matrix-multiply.cu ../cuda/kernels.cu test_utils.cpp
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(CUDA_LIBS) $^ -o $@

# 编译L2DistanceOp测试
test_l2_distance: test_l2_distance.cu ../cuda/kernels.cu test_utils.cpp
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(CUDA_LIBS) $^ -o $@

# 编译MatrixMultiply测试
test_matrix_multiply: test_matrix_multiply.cu ../cuda/matrix-multiply.cu ../cuda/kernels.cu test_utils.cpp
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(CUDA_LIBS) $^ -o $@

# 运行所有测试
test: all
	@echo "运行所有CUDA测试..."
	@echo "=================================="
	@./test_vector_normalizer
	@echo "=================================="
	@./test_cosine_distance
	@echo "=================================="
	@./test_l2_distance
	@echo "=================================="
	@./test_matrix_multiply
	@echo "=================================="
	@echo "所有测试完成！"

# 运行单个测试
test-normalizer: test_vector_normalizer
	./test_vector_normalizer

test-cosine: test_cosine_distance
	./test_cosine_distance

test-l2: test_l2_distance
	./test_l2_distance

test-matrix: test_matrix_multiply
	./test_matrix_multiply

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
	@echo "  test-l2          - 运行L2距离测试"
	@echo "  test-matrix      - 运行矩阵乘法测试"
	@echo "  clean            - 清理编译文件"
	@echo "  help             - 显示此帮助信息"

.PHONY: all test test-normalizer test-cosine test-l2 test-matrix clean help
