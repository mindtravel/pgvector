#pragma once

class VectorNormalizer {
public:
    // 构造与析构
    VectorNormalizer();
    ~VectorNormalizer();

    // 向量归一化（L2范数归一化，原地修改输入数组）
    void normalize(float** vector_list, int n_batch, int n_dim);
    void normalize_async(float*** vector_list, int n_lists, int n_batch, int n_dim);
    // 获取最近一次归一化的模长
    float last_norm() const;

private:
    float norm_;
};

class CosineDistanceOp {
public:
    CosineDistanceOp(int n);
    ~CosineDistanceOp();
    float compute(const float* a, const float* b, const int n) ;

private:
    int n_, blockSize_, gridSize_;
    float *d_a_, *d_b_, *d_dot_, *d_sq_a_, *d_sq_b_;
    float *d_dot_sum_, *d_sq_a_sum_, *d_sq_b_sum_;
    float *h_dot_sum_, *h_sq_a_sum_, *h_sq_b_sum_;
};

// 全局CUDA核函数声明
__global__ void dotProductKernel(const float* a, const float* b, float* result, int n);
__global__ void squareKernel(const float* vec, float* result, int n);
__global__ void reduceSumKernel(const float* input, float* output, int n);

class L2DistanceOp {
public:
    // 计算 L2 距离（输入为设备端指针）
    float operator()(const float* d_A, const float* d_B, int n);

private:
    // 不需要在这里声明，因为已经在全局声明了
};

// 全局CUDA核函数声明
__global__ void l2_distance_kernel(const float* A, const float* B, float* result, int n);