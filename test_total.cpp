#include "total.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

int main() {
    // 测试 VectorNormalizer
    std::vector<float> vec = {1, 2, 3, 4, 5, 6, 7, 8};
    VectorNormalizer normalizer;
    normalizer.normalize(vec.data(), vec.size());
    std::cout << "L2 Norm: " << normalizer.last_norm() << std::endl;
    std::cout << "Normalized vector: ";
    for (float v : vec) std::cout << v << " ";
    std::cout << std::endl;

    // 测试 CosineDistanceOp
    std::vector<float> a = {1, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> b = {0, 1, 0, 0, 0, 0, 0, 0};
    CosineDistanceOp cosine_op(a.size());
    float cosine_dist = cosine_op.compute(a.data(), b.data());
    std::cout << "Cosine distance: " << cosine_dist << std::endl;

    // 测试 L2DistanceOp
    // 先将a和b拷贝到设备端
    float *d_a, *d_b;
    cudaMalloc(&d_a, a.size() * sizeof(float));
    cudaMalloc(&d_b, b.size() * sizeof(float));
    cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

    L2DistanceOp l2_op;
    float l2_dist = l2_op(d_a, d_b, a.size());
    std::cout << "L2 distance: " << l2_dist << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}