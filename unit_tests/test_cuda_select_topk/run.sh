#!/bin/bash
echo "=== test_cuda_select_topk ==="

export CUDA_HOME=/usr/local/cuda
export RAFT_ROOT=/root/miniconda3/envs/ann-benchmarks
export LD_LIBRARY_PATH=/root/miniconda3/envs/ann-benchmarks/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

INCLUDE_PATHS=(
    "-I${RAFT_ROOT}/include"
    "-I${RAFT_ROOT}/include/raft"
    "-I${RAFT_ROOT}/include/cuvs"
    "-I${RAFT_ROOT}/include/rmm"
    "-I${RAFT_ROOT}/targets/x86_64-linux/include"
    "-I${RAFT_ROOT}/include/rapids"
    "-I${CUDA_HOME}/include"
    "-I/root/pgvector/cuda"
)

LIB_PATHS=(
    "-L/usr/local/cuda/lib64"
    "-L${RAFT_ROOT}/lib"
)

LIBS=(
    "-lraft"
    "-lcuvs"
    "-lrmm"
    "-lcudart"
    "-lcurand"
    "-lcublas"
    "-lcusparse"
)

CUDA_ARCH="-arch=sm_70"
STD_CXX="-std=c++17"
OPTIMIZATION="-O3"
DEFINES="-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
EXPERIMENTAL_FLAGS="--expt-relaxed-constexpr"

echo "包含路径: ${INCLUDE_PATHS[*]}"
echo "库路径: ${LIB_PATHS[*]}"
echo "链接库: ${LIBS[*]}"

nvcc ${CUDA_ARCH} ${STD_CXX} ${OPTIMIZATION} ${DEFINES} ${EXPERIMENTAL_FLAGS} \
    ${INCLUDE_PATHS[*]} \
    ${LIB_PATHS[*]} \
    ${LIBS[*]} \
    test_cuda_select_topk/test.cu ../cuda/select_topk.cu \
    -o test_cuda_select_topk/test \
    --cudart=shared

if [ $? -eq 0 ]; then
    echo "编译成功!"
    echo "运行测试程序..."
    ./test_cuda_select_topk/test
else
    echo "编译失败!"
    exit 1
fi
