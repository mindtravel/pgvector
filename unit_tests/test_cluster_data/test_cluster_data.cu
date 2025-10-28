// 测试修复后的函数
#include "../unit_tests/common/test_utils.cuh"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Testing generate_cluster_query_data function...\n");
    
    // 准备测试数据
    int n_query = 10;
    int k = 3;
    int batch_size = 2;
    
    int* query_cluster_group = (int*)malloc(n_query * k * sizeof(int));
    
    // 填充测试数据
    for (int i = 0; i < n_query * k; i++) {
        query_cluster_group[i] = i % 5; // 假设有5个cluster (0-4)
    }
    
    printf("Input data:\n");
    for (int i = 0; i < n_query; i++) {
        printf("Query %d: ", i);
        for (int j = 0; j < k; j++) {
            printf("%d ", query_cluster_group[i * k + j]);
        }
        printf("\n");
    }
    
    // 调用函数
    void* result = generate_cluster_query_data(query_cluster_group, n_query, k, batch_size);
    
    if (result != NULL) {
        printf("Function executed successfully!\n");
        printf("Result pointer: %p\n", result);
    } else {
        printf("Function failed!\n");
    }
    
    // 清理内存
    free(query_cluster_group);
    
    printf("Test completed.\n");
    return 0;
}
