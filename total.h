#pragma once

class VectorNormalizer {
public:
    // 构造与析构
    VectorNormalizer();
    ~VectorNormalizer();

    // 向量归一化（L2范数归一化，原地修改输入数组）
    void normalize(float* data, int n);

    // 获取最近一次归一化的模长
    float last_norm() const;

private:
    float norm_;
};

class CosineDistanceOp {
public:
    CosineDistanceOp(int n);
    ~CosineDistanceOp();
    float compute(const float* a, const float* b);

private:
    int n_, blockSize_, gridSize_;
    float *d_a_, *d_b_, *d_dot_, *d_sq_a_, *d_sq_b_;
    float *d_dot_sum_, *d_sq_a_sum_, *d_sq_b_sum_;
    float *h_dot_sum_, *h_sq_a_sum_, *h_sq_b_sum_;

    static __global__ void dotProductKernel(const float* a, const float* b, float* result, int n);
    static __global__ void squareKernel(const float* vec, float* result, int n);
    static __global__ void reduceSumKernel(const float* input, float* output, int n);
};

class L2DistanceOp {
public:
    // 计算 L2 距离（输入为设备端指针）
    float operator()(const float* d_A, const float* d_B, int n);

private:
    // CUDA Kernel 计算平方差
    static __global__ void l2_distance_kernel(const float* A, const float* B, float* result, int n);
};