#!/bin/bash
cd /home/diy/lzx/ivftensor
source /home/diy/miniconda3/etc/profile.d/conda.sh
conda activate ann-benchmarks
# rm -rf build python/build
cd build
PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
PYTHON_EXEC=$(which python3)
cmake -Dpybind11_DIR="$PYBIND11_DIR" -DPython3_EXECUTABLE="$PYTHON_EXEC" ..
make PyIVFTensor -j$(nproc)
