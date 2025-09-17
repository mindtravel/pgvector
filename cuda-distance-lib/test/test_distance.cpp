#include <iostream>
#include <cmath>
#include "distance.h"
#include <cuda_runtime.h>

void print_results(const int* results, int N1, int k) {
    for (int i = 0; i < N1; ++i) {
        std::cout << "Query " << i << " top " << k << " results: ";
        for (int j = 0; j < k; ++j) {
            std::cout << results[i * k + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    const int N1 = 2; // Number of queries
    const int N2 = 5; // Number of data points
    const int D = 3;  // Dimension of each vector
    const int k = 2;  // Number of top results to retrieve

    float h_query[N1 * D] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}; // Sample query vectors
    float h_data[N2 * D] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 0.5f, 0.5f}; // Sample data vectors
    int h_result[N1 * k]; // To store the results

    // Compute cosine distances
    search_batch_cosine_distance_cuda(h_query, h_data, N1, N2, D, k, h_result);
    print_results(h_result, N1, k);

    // Compute L2 distances
    search_batch_l2_distance_cuda(h_query, h_data, N1, N2, D, k, h_result);
    print_results(h_result, N1, k);

    return 0;
}