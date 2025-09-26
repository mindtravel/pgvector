# CUDA Select TopK 测试 Makefile

# 环境变量设置
CUDA_HOME ?= /usr/local/cuda
RAFT_ROOT ?= /root/miniconda3/envs/ann-benchmarks

# 编译器设置
NVCC = $(CUDA_HOME)/bin/nvcc

# 包含路径
INCLUDE_PATHS = \
    -I$(RAFT_ROOT)/include \
    -I$(RAFT_ROOT)/include/raft \
    -I$(RAFT_ROOT)/include/cuvs \
    -I$(RAFT_ROOT)/include/rmm \
    -I$(RAFT_ROOT)/targets/x86_64-linux/include \
    -I$(RAFT_ROOT)/include/rapids \
    -I$(CUDA_HOME)/include \
    -I/root/pgvector/cuda

# 库路径
LIB_PATHS = \
    -L$(CUDA_HOME)/lib64 \
    -L$(RAFT_ROOT)/lib

# 链接库
LIBS = \
    -lraft \
    -lcuvs \
    -lrmm \
    -lcudart \
    -lcurand \
    -lcublas \
    -lcusparse

# CUDA编译选项
CUDA_ARCH = -arch=sm_70
STD_CXX = -std=c++17
OPTIMIZATION = -O3
DEFINES = -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
EXPERIMENTAL_FLAGS = --expt-relaxed-constexpr

# 源文件
TEST_SOURCE = test.cu
CUDA_SOURCE = ../../cuda/select_topk.cu
TARGET = test

# 设置环境变量
export LD_LIBRARY_PATH := $(RAFT_ROOT)/lib:$(CUDA_HOME)/lib64:$(LD_LIBRARY_PATH)

# 默认目标
all: $(TARGET)

# 编译目标
$(TARGET): $(TEST_SOURCE) $(CUDA_SOURCE)
	@echo "=== test_cuda_select_topk ==="
	@echo "包含路径: $(INCLUDE_PATHS)"
	@echo "库路径: $(LIB_PATHS)"
	@echo "链接库: $(LIBS)"
	$(NVCC) $(CUDA_ARCH) $(STD_CXX) $(OPTIMIZATION) $(DEFINES) $(EXPERIMENTAL_FLAGS) \
		$(INCLUDE_PATHS) \
		$(LIB_PATHS) \
		$(LIBS) \
		$(TEST_SOURCE) $(CUDA_SOURCE) \
		-o $(TARGET) \
		--cudart=shared

# 运行测试
run: $(TARGET)
	@echo "运行测试程序..."
	./$(TARGET)

# 清理
clean:
	rm -f $(TARGET)

# 帮助信息
help:
	@echo "可用的目标："
	@echo "  all     - 编译测试程序"
	@echo "  run     - 编译并运行测试程序"
	@echo "  clean   - 清理编译文件"
	@echo "  help    - 显示此帮助信息"

.PHONY: all run clean help