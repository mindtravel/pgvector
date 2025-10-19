#include "../common/test_utils.cuh"
int main(){
    int x = 3, y = 4;
    float* test = (float*)malloc(x*y*sizeof(float));
    memset(test, 1, x*y*sizeof(float));
    float* d_test;
    cudaMalloc(&d_test, x*y*sizeof(float));
    // cudaMemset(&d_test, 0, x*y*sizeof(float));
    cudaMemcpy(d_test, test, x*y*sizeof(float), cudaMemcpyHostToDevice);

    COUT_ENDL("test print cuda 2d", d_test, x, y);
}
