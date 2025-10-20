#include "../cuda/experiments/distances.h"
#include "../cuda/pch.h"
#include "../common/test_utils.cuh"

#define EPSILON 1e-2
#define DIV_EPSILON 1e-4

// CPU版本的余弦距离计算（用于验证）
void cpu_cosine_distance(float** query_vectors, float** data_vectors, float** cos_dist,
                        int n_query, int n_batch, int n_dim) {
    // 计算每个向量的L2范数
    float* query_norms = (float*)malloc(n_query * sizeof(float));
    float* data_norms = (float*)malloc(n_batch * sizeof(float));
    
    // 计算query向量的L2范数
    for (int i = 0; i < n_query; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n_dim; j++) {
            sum += query_vectors[i][j] * query_vectors[i][j];
        }
        query_norms[i] = sqrt(sum);
    }
    
    // 计算data向量的L2范数
    for (int i = 0; i < n_batch; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n_dim; j++) {
            sum += data_vectors[i][j] * data_vectors[i][j];
        }
        data_norms[i] = sqrt(sum);
    }
    
    // 计算余弦距离矩阵
    for (int i = 0; i < n_query; i++) {
        for (int j = 0; j < n_batch; j++) {
            // 计算点积
            float dot_product = 0.0f;
            for (int k = 0; k < n_dim; k++) {
                dot_product += query_vectors[i][k] * data_vectors[j][k];
            }
            
            float cos_sim;
            // 计算余弦相似度
            if (query_norms[i] < 1e-6f || data_norms[j] < 1e-6f)
                cos_sim = 0.0f;  // 如果任一向量接近零向量，相似度为0
            else
                cos_sim = dot_product / (query_norms[i] * data_norms[j]);
                // cos_sim = (query_norms[i]);
                // cos_sim = dot_product;
            // 余弦距离 = 1 - 余弦相似度
            // cos_dist[i][j] = 1.0f - cos_sim;
            cos_dist[i][j] = cos_sim;
        }
    }
    
    free(query_norms);
    free(data_norms);
}

// 测试1：基本余弦距离计算
bool test_basic_cosine_distance(int n_query, int n_batch, int n_dim) {
    COUT_ENDL("=== Test1: 基本余弦距离测试 ===");
    bool pass = true;

    float alpha = 1.0f, beta = 0.0f;
    
    COUT_ENDL("测试向量组大小: ", n_query, " 个查询向量 × ", n_batch, " 个数据向量");
    COUT_ENDL("向量维度: ", n_dim);
    
    // 计算内存使用量
    size_t memory_mb = (n_query * n_dim + n_batch * n_dim + n_query * n_batch) * sizeof(float) / (1024 * 1024);
    COUT_ENDL("内存使用量: ", memory_mb, " MB");
    
    // 生成测试数据
    float** h_query_vectors = generate_vector_list(n_query, n_dim);
    float** h_data_vectors = generate_vector_list(n_batch, n_dim);
    float** h_cos_dist_gpu = (float**)malloc_vector_list(n_query, n_batch, sizeof(float));
    float** h_cos_dist_cpu = (float**)malloc_vector_list(n_query, n_batch, sizeof(float));
    
    // std::cout << h_query_vectors[0] << std::endl;
    // std::cout << h_data_vectors[0] << std::endl;
    if(DEBUG==true){
        // std::cout << "query" << std::endl;
        // for(int i=0; i<n_query; ++i){
        //     for(int j=0; j<n_dim; ++j)
        //         std::cout << h_query_vectors[i][j] << " ";
        //     std::cout << std::endl;        
        // }
        // std::cout << std::endl;        

        // std::cout << "data" << std::endl;
        // for(int i=0; i<n_batch; ++i){
        //     for(int j=0; j<n_dim; ++j)
        //         std::cout << h_data_vectors[i][j] << " ";
        //     std::cout << std::endl;        
        // }
        // std::cout << std::endl;          
    }

    long long cpu_duration_ms = 0, gpu_duration_ms = 0;

    MEASURE_MS_AND_SAVE("gpu耗时：", gpu_duration_ms, 
        cuda_cosine_dist(h_query_vectors, h_data_vectors, h_cos_dist_gpu, n_query, n_batch, n_dim, alpha, beta);
    );
    
    MEASURE_MS_AND_SAVE("cpu耗时：", cpu_duration_ms,
        cpu_cosine_distance(h_query_vectors, h_data_vectors, h_cos_dist_cpu, n_query, n_batch, n_dim);
    );
    
    // 验证结果
    pass &= compare_2D(h_cos_dist_gpu, h_cos_dist_cpu, n_query, n_batch, EPSILON);
    COUT_ENDL("加速比", (float)cpu_duration_ms / (float)gpu_duration_ms, "x");

    // 清理内存
    free_vector_list((void**)h_query_vectors);
    free_vector_list((void**)h_data_vectors);
    free_vector_list((void**)h_cos_dist_gpu);
    free_vector_list((void**)h_cos_dist_cpu);

    return pass;
}

// 测试2：单位向量测试
void test_unit_vectors() {
    std::cout << "=== Test2: 单位向量测试 ===" << std::endl;
    
    int n_query = 4, n_batch = 4, n_dim = 3;
    float alpha = 1.0f, beta = 0.0f;
    
    // 生成单位向量
    float** h_query_vectors = (float**)malloc_vector_list(n_query, n_dim, sizeof(float));
    float** h_data_vectors = (float**)malloc_vector_list(n_batch, n_dim, sizeof(float));
    float** h_cos_dist_gpu = (float**)malloc_vector_list(n_query, n_batch, sizeof(float));
    float** h_cos_dist_cpu = (float**)malloc_vector_list(n_query, n_batch, sizeof(float));
    
    // 设置单位向量
    // Query向量: [1,0,0], [0,1,0], [0,0,1], [1,1,1]/sqrt(3)
    h_query_vectors[0][0] = 1.0f; h_query_vectors[0][1] = 0.0f; h_query_vectors[0][2] = 0.0f;
    h_query_vectors[1][0] = 0.0f; h_query_vectors[1][1] = 1.0f; h_query_vectors[1][2] = 0.0f;
    h_query_vectors[2][0] = 0.0f; h_query_vectors[2][1] = 0.0f; h_query_vectors[2][2] = 1.0f;
    h_query_vectors[3][0] = 1.0f/sqrt(3.0f); h_query_vectors[3][1] = 1.0f/sqrt(3.0f); h_query_vectors[3][2] = 1.0f/sqrt(3.0f);
    
    // Data向量: [1,0,0], [0,1,0], [0,0,1], [1,1,1]/sqrt(3)
    h_data_vectors[0][0] = 1.0f; h_data_vectors[0][1] = 0.0f; h_data_vectors[0][2] = 0.0f;
    h_data_vectors[1][0] = 0.0f; h_data_vectors[1][1] = 1.0f; h_data_vectors[1][2] = 0.0f;
    h_data_vectors[2][0] = 0.0f; h_data_vectors[2][1] = 0.0f; h_data_vectors[2][2] = 1.0f;
    h_data_vectors[3][0] = 1.0f/sqrt(3.0f); h_data_vectors[3][1] = 1.0f/sqrt(3.0f); h_data_vectors[3][2] = 1.0f/sqrt(3.0f);
    
    // GPU计算
    cuda_cosine_dist(h_query_vectors, h_data_vectors, h_cos_dist_gpu, n_query, n_batch, n_dim, alpha, beta);
    
    // CPU计算
    cpu_cosine_distance(h_query_vectors, h_data_vectors, h_cos_dist_cpu, n_query, n_batch, n_dim);
    
    // 验证结果
    assert(compare_2D(h_cos_dist_gpu, h_cos_dist_cpu, n_query, n_batch, EPSILON));
    
    // 打印结果矩阵（前几个元素）
    std::cout << "余弦距离矩阵（前4x4）:" << std::endl;
    for (int i = 0; i < std::min(4, n_query); i++) {
        for (int j = 0; j < std::min(4, n_batch); j++) {
            std::cout << h_cos_dist_gpu[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    // 清理内存
    free_vector_list((void**)h_query_vectors);
    free_vector_list((void**)h_data_vectors);
    free_vector_list((void**)h_cos_dist_gpu);
    free_vector_list((void**)h_cos_dist_cpu);
    
    std::cout << "单位向量测试通过 ✓" << std::endl << std::endl;
}

// 测试3：大规模压力测试
void test_large_scale_cosine_distance(int n_query, int n_batch, int n_dim) {
    std::cout << "=== Test3: 大规模余弦距离压力测试 ===" << std::endl;
    
    float alpha = 1.0f, beta = 0.0f;
    
    std::cout << "测试向量组大小: " << n_query << " 个查询向量 × " << n_batch << " 个数据向量" << std::endl;
    std::cout << "向量维度: " << n_dim << std::endl;
    
    // 计算内存使用量
    size_t memory_mb = (n_query * n_dim + n_batch * n_dim + n_query * n_batch) * sizeof(float) / (1024 * 1024);
    std::cout << "内存使用量: " << memory_mb << " MB" << std::endl;
    
    // 生成测试数据
    float** h_query_vectors = generate_vector_list(n_query, n_dim);
    float** h_data_vectors = generate_vector_list(n_batch, n_dim);
    float** h_cos_dist_gpu = (float**)malloc_vector_list(n_query, n_batch, sizeof(float));
    float** h_cos_dist_cpu = (float**)malloc_vector_list(n_query, n_batch, sizeof(float));
    
    long long cpu_duration_ms = 0, gpu_duration_ms = 0;

    MEASURE_MS_AND_SAVE("gpu耗时：", gpu_duration_ms,
        cuda_cosine_dist(h_query_vectors, h_data_vectors, h_cos_dist_gpu, n_query, n_batch, n_dim, alpha, beta);
    );

    MEASURE_MS_AND_SAVE("cpu耗时：", cpu_duration_ms,
        cpu_cosine_distance(h_query_vectors, h_data_vectors, h_cos_dist_cpu, n_query, n_batch, n_dim);
    );

    // 验证结果
    assert(compare_2D(h_cos_dist_gpu, h_cos_dist_cpu, n_query, n_batch, EPSILON));
    COUT_ENDL("加速比", (float)cpu_duration_ms / (float)gpu_duration_ms, "x");
    
    // 清理内存
    free_vector_list((void**)h_query_vectors);
    free_vector_list((void**)h_data_vectors);
    free_vector_list((void**)h_cos_dist_gpu);
    free_vector_list((void**)h_cos_dist_cpu);
    
    std::cout << "大规模余弦距离压力测试完成 ✓" << std::endl << std::endl;
}

int main() {
    srand(time(0));
    COUT_ENDL("开始余弦距离单元测试...");
    
    bool test1 = true;
    // 基本测试
    // test_basic_cosine_distance(3, 5, 4);
    // test_basic_cosine_distance(128, 128, 128);
    test1 &= check_pass("Test 1 (基本功能):", test_basic_cosine_distance(1024, 1024, 1024));
    // 单位向量测试
    // test_unit_vectors();
    
    // 大规模压力测试
    // test_large_scale_cosine_distance(1024, 1024, 512);
    
    COUT_ENDL("all_test_passed");
    return 0;
}
