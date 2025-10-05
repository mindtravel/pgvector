#include <stdlib.h>
#include <limits>

#include "../cuda/distances.h"
#include "../cuda/pch.h"
#include "../common/test_utils.cuh"

#define EPSILON 1e-2
#define DIV_EPSILON 1e-4

// CPU版本余弦版本k近邻计算（用于验证）
void cpu_cosine_distance_topk(float** query_vectors, float** data_vectors, 
                        int** data_index, 
                        int** topk_index, float** topk_dist,
                        int n_query, int n_batch, int n_dim, int k) {
    // 计算每个向量的L2范数
    float* query_norms = (float*)malloc(n_query * sizeof(float));
    float* data_norms = (float*)malloc(n_batch * sizeof(float));
    float** cos_dist = (float**)malloc_vector_list(n_query, n_batch, sizeof(float));
    
    // 计算query向量的L2范数
    for (int i = 0; i < n_query; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n_dim; j++) {
            sum += query_vectors[i][j] * query_vectors[i][j];
        }
        query_norms[i] = sum;
    }
    
    // 计算data向量的L2范数
    for (int i = 0; i < n_batch; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n_dim; j++) {
            sum += data_vectors[i][j] * data_vectors[i][j];
        }
        data_norms[i] = sum;
    }
    
    // 创建索引-距离对的数组用于排序
    std::vector<std::pair<float, int>> cos_pairs(n_batch);
    
    // 计算余弦距离矩阵并进行topk选择
    for (int i = 0; i < n_query; i++) {
        for (int j = 0; j < n_batch; j++) {
            // 计算点积
            float dot_product = 0.0f;
            for (int d = 0; d < n_dim; d++) {
                dot_product += query_vectors[i][d] * data_vectors[j][d];
            }
            float cos_sim;
            // 计算余弦相似度
            if (query_norms[i] < 1e-6f || data_norms[j] < 1e-6f)
                cos_sim = 0.0f;  // 如果任一向量接近零向量，相似度为0
            else
                cos_sim = 1.0f - (dot_product / sqrt(query_norms[i] * data_norms[j]));
            
            // 存储余弦相似度和对应的数据索引
            cos_pairs[j] = std::make_pair(cos_sim, data_index[i][j]);
            // COUT_TABLE(i,j,dot_product,query_norms[i],data_norms[j], cos_sim);
        }
        
        // 按余弦相似度降序排序（按相似度从大到小排序，相似度一样则按索引排序）
        std::sort(cos_pairs.begin(), cos_pairs.end(), 
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) 
                {
                    if(a.first != b.first){
                        return a.first < b.first;  // 降序排序
                    }
                    else{
                        return a.second > b.second;
                    }
                });
        
        // 提取前k个最相似的索引
        for (int j = 0; j < k && j < n_batch; ++j) {
            topk_index[i][j] = cos_pairs[j].second;
            topk_dist[i][j] = cos_pairs[j].first;
        }
    }
    
    free(query_norms);
    free(data_norms);
}

int** generate_data_index(int size_x, int size_y){
    int** data_index = (int**)malloc_vector_list(size_x, size_y, sizeof(int));
    for(int i = 0; i < size_x; i++){
        for(int j = 0; j < size_y; j++){
            data_index[i][j] = j;
        }
    }
    return data_index;
}

// 测试1：基本余弦k近邻计算
void test_basic_cosine_distance_topk(int n_query, int n_batch, int n_dim, int k) {
    COUT_VAL("=== Test1: 基本余弦k近邻测试 ===");    
    COUT_VAL("测试向量组大小: ", n_query, " 个查询向量 × ", n_batch, " 个数据向量");    
    COUT_VAL("向量维度: ", n_dim);    
    
    // 计算内存使用量
    size_t memory_mb = (n_query * n_dim + n_batch * n_dim + n_query * n_batch) * sizeof(float) / (1024 * 1024);
    COUT_VAL("内存使用量: ", memory_mb, " MB");    

    
    // 生成测试数据
    float** h_query_vectors = generate_vector_list(n_query, n_dim);
    float** h_data_vectors = generate_vector_list(n_batch, n_dim);
    // float** h_cos_dist_cpu = (float**)ector_list(n_batch, n_dim);
    // float** h_cos_dist_gpu = (float*malloc_vector_list(n_query, n_batch, sizeof(float));
    int** data_index = generate_data_index(n_query, n_batch);
    int** topk_index_cpu = (int**)malloc_vector_list(n_query, k, sizeof(int));
    int** topk_index_gpu = (int**)malloc_vector_list(n_query, k, sizeof(int));

    /**
     * 将topk距离设为浮点数最小值
     */
    float** topk_dist_cpu = (float**)malloc_vector_list(n_query, k, sizeof(float));
    float** topk_dist_gpu = (float**)malloc_vector_list(n_query, k, sizeof(float));
    for(int i = 0; i < n_query; ++i){
        for(int j = 0; j < k; ++j){
            topk_dist_cpu[i][j] = -std::numeric_limits<float>::max();
            topk_dist_gpu[i][j] = -std::numeric_limits<float>::max();
        }
    }
    // for(int i = 0; i < n_query; ++i){
    //     for(int j = 0; j < k; ++j){
    //         COUT_ENDL(topk_dist_cpu[i][j], topk_dist_gpu[i][j]);
    //     }
    // }
    
    // std::cout << h_query_vectors[0] << std::endl;
    // std::cout << h_data_vectors[0] << std::endl;
    if(DEBUG==true){
        // print_2D("query: ", h_query_vectors, n_query, n_dim);    
        // print_2D("data: ", h_data_vectors, n_batch, n_dim);  
        // print_1D("index", data_index, n_batch);  
    }


    // GPU计算
    auto start = std::chrono::high_resolution_clock::now();
    cuda_cosine_dist_topk(
        h_query_vectors, h_data_vectors, 
        data_index, topk_index_gpu, topk_dist_gpu,
        n_query, n_batch, n_dim,
        k
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // CPU计算
    start = std::chrono::high_resolution_clock::now();
    cpu_cosine_distance_topk(
        h_query_vectors, h_data_vectors, 
        data_index, topk_index_cpu, topk_dist_cpu,
        n_query, n_batch, n_dim, k
    );
    end = std::chrono::high_resolution_clock::now();

    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    
    // 验证结果
    // assert(equal_2D_int(topk_index_gpu, topk_index_cpu, n_query, k));
    assert(equal_2D_float(topk_dist_cpu, topk_dist_gpu, n_query, k));
    
    // 计算性能指标
    float speedup = (float)cpu_duration.count() / gpu_duration.count();
    
    COUT_VAL("GPU耗时: ", gpu_duration.count(), " ms");
    COUT_VAL("CPU耗时: ", cpu_duration.count(), " ms");
    COUT_VAL( "加速比: ", speedup, "x");
    

    // 清理内存
    free(data_index);

    free_vector_list((void**)h_query_vectors);    
    free_vector_list((void**)h_data_vectors);

    free_vector_list((void**)topk_index_gpu);
    free_vector_list((void**)topk_index_cpu);
    free_vector_list((void**)topk_dist_cpu);
    free_vector_list((void**)topk_dist_gpu);

    COUT_ENDL( "基本余弦k近邻测试完成");
}

// 测试3：大规模压力测试
void test_large_scale_cosine_distance_topk(int n_query, int n_batch, int n_dim, int k) {
    std::cout << "=== Test3: 大规模余弦距离压力测试 ===" << std::endl;    
    std::cout << "测试向量组大小: " << n_query << " 个查询向量 × " << n_batch << " 个数据向量" << std::endl;
    std::cout << "向量维度: " << n_dim << std::endl;
    
    // 计算内存使用量
    size_t memory_mb = (n_query * n_dim + n_batch * n_dim + n_query * n_batch) * sizeof(float) / (1024 * 1024);
    std::cout << "内存使用量: " << memory_mb << " MB" << std::endl;
    
    // 生成测试数据
    float** h_query_vectors = generate_vector_list(n_query, n_dim);
    float** h_data_vectors = generate_vector_list(n_batch, n_dim);
    // float** h_cos_dist_gpu = (float**)malloc_vector_list(n_query, n_batch, sizeof(float));
    // float** h_cos_dist_cpu = (float**)malloc_vector_list(n_query, n_batch, sizeof(float));
    int** data_index = generate_data_index(n_query, n_batch);
    int** topk_index_cpu = (int**)malloc_vector_list(n_query, k, sizeof(int));
    int** topk_index_gpu = (int**)malloc_vector_list(n_query, k, sizeof(int));
    float** topk_dist_cpu = (float**)malloc_vector_list(n_query, k, sizeof(float));
    float** topk_dist_gpu = (float**)malloc_vector_list(n_query, k, sizeof(float));

    // GPU计算
    auto start = std::chrono::high_resolution_clock::now();
    cuda_cosine_dist_topk(
        h_query_vectors, h_data_vectors, 
        data_index, topk_index_gpu, topk_dist_gpu,
        n_query, n_batch, n_dim, k
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // CPU计算
    start = std::chrono::high_resolution_clock::now();
    cpu_cosine_distance_topk(
        h_query_vectors, h_data_vectors, 
        data_index, topk_index_cpu, topk_dist_cpu,
        n_query, n_batch, n_dim, k
    );
    end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 验证结果
    // assert(equal_2D_int(topk_index_cpu, topk_index_gpu, n_query, k));
    assert(equal_2D_float(topk_dist_cpu, topk_dist_gpu, n_query, k));
    
    // 计算性能指标
    float speedup = (float)cpu_duration.count() / gpu_duration.count();
    
    std::cout << "GPU耗时: " << gpu_duration.count() << " ms" << std::endl;
    std::cout << "CPU耗时: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "加速比: " << speedup << "x" << std::endl;
    
    // 清理内存
    free_vector_list((void**)h_query_vectors);
    free_vector_list((void**)h_data_vectors);
    free_vector_list((void**)topk_index_gpu);
    free_vector_list((void**)topk_index_cpu);
    free_vector_list((void**)topk_dist_cpu);
    free_vector_list((void**)topk_dist_gpu);
    
    std::cout << "大规模余弦距离压力测试完成 ✓" << std::endl << std::endl;
}

int main() {
    srand(time(0));
    std::cout << "开始余弦距离单元测试..." << std::endl << std::endl;
    
    try {
        // 基本测试
        test_basic_cosine_distance_topk(3, 5, 4, 2);
        // test_basic_cosine_distance_topk(1024, 1024, 1024, 100);
        // test_basic_cosine_distance(128, 128, 128);
        
        // 单位向量测试
        // test_unit_vectors();
        
        // 大规模压力测试
        // test_large_scale_cosine_distance(1024, 1024, 512);
        
        std::cout << "all_test_passed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
}
