# Makefile

CUDA_PATH   ?= /usr/local/cuda
NVCC        := $(CUDA_PATH)/bin/nvcc
CXX         := g++
CXXFLAGS    := -Iinclude -std=c++11
NVCCFLAGS   := -Iinclude -arch=sm_60

SRC_DIR     := src
TEST_DIR    := test

SRC_CU      := $(SRC_DIR)/distance_cuda.cu
SRC_CPP     := $(SRC_DIR)/distance.cpp
TEST_CPP    := $(TEST_DIR)/test_distance.cpp

TARGET      := test_distance

all: $(TARGET)

$(TARGET): $(SRC_CU) $(SRC_CPP) $(TEST_CPP)
	$(NVCC) $(NVCCFLAGS) $(SRC_CU) $(SRC_CPP) $(TEST_CPP) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o $(SRC_DIR)/*.o

test: $(TARGET)
	./$(TARGET)

.PHONY: all clean test