# CUDA测试Makefile
CUDA_PATH ?= /usr/local/cuda
RAFT_ROOT ?= /root/miniconda3/envs/ann-benchmarks
NVCC = ccache $(CUDA_PATH)/bin/nvcc

# CUDA编译选项
CUDA_ARCH = -arch=sm_70
STD_CXX = -std=c++17
OPTIMIZATION = -O3
DEFINES = -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
EXPERIMENTAL_FLAGS = --expt-relaxed-constexpr
# CUDA_FLAGS += -Xlinker -fuse-ld=lld# 使用新的链接器
CUDA_FLAGS = $(CUDA_ARCH) $(STD_CXX) $(OPTIMIZATION) $(DEFINES) $(EXPERIMENTAL_FLAGS)

CUDA_INCLUDE = \
	-I$(CUDA_PATH)/include \
	-I../cuda -I.

CUDA_LIBS = \
	-L$(CUDA_PATH)/lib64 \
	-lcudart \
	-lcurand \
	-lcublas \
	-lcusparse

# CXXFLAGS := -Wall -Wextra -std=c++17 -O2

# 设置环境变量
# export LD_LIBRARY_PATH := $(RAFT_ROOT)/lib:$(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH)

# 通用头文件包含 - 使用统一的头文件来减少重复解析
COMMON_HEADERS = -include ../cuda/pch.h

# 测试目标
TESTS = \
	test_cosine_distance \
	test_cosine_distance_topk \

# 默认目标
all: $(TESTS)

# 编译Cosine Distance Top-K测试
test_cosine_distance_topk: test_cosine_distance_topk.cu ../cuda/distances.cu ../cuda/kernels.cu ../cuda/fusion_cos_topk.cu test_utils.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(COMMON_HEADERS) $(CUDA_LIBS) test_cosine_distance_topk.cu ../cuda/distances.cu ../cuda/kernels.cu ../cuda/fusion_cos_topk.cu test_utils.cu -o $@


# 运行所有测试
test: all
	@echo "运行所有CUDA测试..."
	@echo "=================================="
	@./test_cosine_distance_topk
	@echo "=================================="
	@echo "所有测试完成！"

# 运行单个测试
test-cosine-topk: test_cosine_distance_topk
	./test_cosine_distance_topk

# 清理
clean:
	rm -f $(TESTS)

# 帮助信息
help:
	@echo "可用的目标："
	@echo "  all              - 编译所有测试"
	@echo "  test             - 编译并运行所有测试"
	@echo "  test-cosine-topk      - 运行余弦最近邻测试"
	@echo "  clean            - 清理编译文件"
	@echo "  help             - 显示此帮助信息"

.PHONY: all test test-cosine-topk clean help
