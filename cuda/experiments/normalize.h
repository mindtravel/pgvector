#ifndef NORMALIZE_H
#define NORMALIZE_H

// 向量归一化（L2范数归一化，原地修改输入数组）
void normalize(float** vector_list, int n_batch, int n_dim);
void normalize_async(float*** vector_list, int n_lists, int n_batch, int n_dim);

__global__ void normalize_kernel(float *vector_data, float *vector_square_sum, int n_batch, int n_dim);

#endif
    