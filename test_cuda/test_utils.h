#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#define DEBUG false
bool float_equal(float a, float b, float epsilon = 1e-5f);
bool float_equal_relative(float a, float b, float epsilon = 1e-5f);
bool matrix_equal(float* a, float* b, int rows, int cols, float epsilon = 1e-5f);
bool matrix_equal_2D(float** a, float** b, int rows, int cols, float epsilon = 1e-5f);
float** malloc_vector_list(int n_batch, int n_dim);
float** generate_vector_list(int n_batch, int n_dim);
float*** generate_large_scale_vectors(int n_lists, int n_batch, int n_dim);
void free_vector_list(float** vector_list);

#endif