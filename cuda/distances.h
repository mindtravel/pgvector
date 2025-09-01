#pragma once
void cuda_cosine_dist(
    float** query_vector_group_cpu, float** data_vector_group_cpu, float** h_cos_dist,
    int n_query, int n_batch, int n_dim, 
    float alpha, float beta);

class L2DistanceOp {
    public:
        // 计算 L2 距离（输入为设备端指针）
        float operator()(const float* d_A, const float* d_B, int n);
    
    private:
        // 不需要在这里声明，因为已经在全局声明了
    };
    
    // 全局CUDA核函数声明
    __global__ void l2_distance_kernel(const float* A, const float* B, float* result, int n);