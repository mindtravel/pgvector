#include "../pch.h"
#include "integrate_screen.cuh"

#include "../fusion_cos_topk/fusion_cos_topk.cuh"
#include "../indexed_gemm/indexed_gemm.cuh"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort_topk.cu"
#include "../cudatimer.h"
#include "../../unit_tests/common/test_utils.cuh"
#include "../l2norm/l2norm.cuh"
#include "../utils.cuh"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <cstring>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define ENABLE_CUDA_TIMING 0

using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort_topk;

void batch_search_pipeline(float** query_batch,
                           int* cluster_size,
                           float*** cluster_vectors,
                           float** cluster_center_data,
                           
                           float** topk_dist,
                           int** topk_index,
                           int* n_isnull,


                           int n_query,
                           int n_dim,
                           int n_total_clusters,
                           int n_total_vectors,
                           int n_probes,
                           int k) {

    if (n_query <= 0 || n_dim <= 0 || n_total_clusters <= 0 || k <= 0) {
        printf("[ERROR] Invalid parameters: n_query=%d, n_dim=%d, n_total_clusters=%d, k=%d\n",
               n_query, n_dim, n_total_clusters, k);
        throw std::invalid_argument("invalid batch_search_pipeline configuration");
    }
    if (!cluster_size || !cluster_vectors) {
        throw std::invalid_argument("cluster metadata is null");
    }

    if (!cluster_center_data) {
        throw std::invalid_argument("cluster_center_data must not be null for coarse search");
    }
    if (n_probes <= 0 || n_probes > n_total_clusters) {
        throw std::invalid_argument("invalid n_probes");
    }


    float* d_queries = nullptr;

    float* d_cluster_vectors = nullptr; // 指向所有cluster开头
    float** d_cluster_vector_ptr = nullptr; // 指向每个cluster的开头

    float *d_cluster_centers = nullptr;
    float *d_cluster_centers_norm = nullptr;
    float *d_query_norm = nullptr;
    float *d_cluster_vector_norm = nullptr;

    int *d_topk_index = nullptr;
    float *d_topk_dist = nullptr;

    int *d_top_nprobe_index = nullptr;
    float *d_top_nprobe_dist = nullptr;
    float *d_inner_product = nullptr;
    int* d_probe_vector_offset = nullptr;
    int* d_probe_vector_count = nullptr;


    {
        CUDATimer timer("Step 0: Data Preparation");
        cudaMalloc(&d_queries, n_query * n_dim * sizeof(float));
        cudaMemcpy(d_queries, query_batch[0], n_query * n_dim * sizeof(float), cudaMemcpyHostToDevice);
        CHECK_CUDA_ERRORS

        d_cluster_vector_ptr = (float**)malloc(n_total_clusters * sizeof(float*));
        cudaMalloc(&d_cluster_vectors, n_total_vectors * n_dim * sizeof(float));

        // 先在GPU上计算probe_vector_offset（使用前缀和）
        cudaMalloc(&d_probe_vector_offset, (n_total_clusters + 1) * sizeof(int));
        cudaMalloc(&d_probe_vector_count, n_total_clusters * sizeof(int));
        cudaMemcpy(d_probe_vector_count, cluster_size, 
                   n_total_clusters * sizeof(int), cudaMemcpyHostToDevice);
        CHECK_CUDA_ERRORS;
        
        // 使用GPU前缀和计算offset数组
        compute_prefix_sum(d_probe_vector_count, d_probe_vector_offset, n_total_clusters, 0);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
        
        // 从GPU读取offset数组（用于计算d_cluster_vector_ptr）
        int* probe_vector_offset_host = (int*)malloc(n_total_clusters * sizeof(int));
        cudaMemcpy(probe_vector_offset_host, d_probe_vector_offset, 
                   n_total_clusters * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();  // 确保数据复制完成
        CHECK_CUDA_ERRORS;
        
        // 复制cluster向量到GPU，并使用GPU计算的offset
        // 确保所有复制操作完成后再继续
        for (int i = 0; i < n_total_clusters; ++i) {
            if (cluster_size[i] > 0) {  // 只复制非空的cluster
                // cluster_vectors 是 float***，所以 cluster_vectors[i] 是 float**，cluster_vectors[i][0] 是 float*
                float* cluster_start = d_cluster_vectors + probe_vector_offset_host[i] * n_dim;
                cudaMemcpy(cluster_start, cluster_vectors[i][0], cluster_size[i] * n_dim * sizeof(float), cudaMemcpyHostToDevice);
                d_cluster_vector_ptr[i] = cluster_start;
            }
        }
        cudaDeviceSynchronize();  // 确保所有复制完成
        CHECK_CUDA_ERRORS;
        free(probe_vector_offset_host);
    
        cudaMalloc(&d_cluster_centers, n_total_clusters * n_dim * sizeof(float));
        cudaMemcpy(d_cluster_centers, cluster_center_data[0], n_total_clusters * n_dim * sizeof(float), cudaMemcpyHostToDevice);
        CHECK_CUDA_ERRORS;

        cudaMalloc(&d_cluster_vector_norm, n_total_vectors * sizeof(float));
        cudaMalloc(&d_query_norm, n_query * sizeof(float)); /*存储query的l2 Norm*/
        cudaMalloc(&d_cluster_centers_norm, n_total_clusters * sizeof(float)); /*存储data的l2 Norm*/
        CHECK_CUDA_ERRORS;

        cudaMalloc(&d_topk_dist, n_query * k * sizeof(float));/*存储topk距离*/
        cudaMalloc(&d_topk_index, n_query * k * sizeof(int));/*存储topk索引*/
        cudaMalloc(&d_top_nprobe_index, n_query * n_probes * sizeof(int));/*存储top n_probes索引*/
        CHECK_CUDA_ERRORS;
        cudaDeviceSynchronize();

        compute_l2_norm_gpu(d_cluster_vectors, d_cluster_vector_norm, n_total_vectors, n_dim);
        compute_l2_norm_gpu(d_queries, d_query_norm, n_query, n_dim);
        compute_l2_norm_gpu(d_cluster_centers, d_cluster_centers_norm, n_total_clusters, n_dim);
        
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    }

        // ------------------------------------------------------------------
        // Step 1. 粗筛：调用 warpsort 融合算子，得到 query -> cluster mapping
        // ------------------------------------------------------------------
    // 注意：data_index 在 cuda_cos_topk_warpsort 内部使用 CUDA kernel 自动生成顺序索引 [0, 1, 2, ..., n_total_clusters-1]
    
    {
        CUDATimer timer("Step 1: Coarse Search (cuda_cos_topk_warpsort)");
        float alpha = 1.0f; 
        float beta = 0.0f;
    
        dim3 queryDim(n_query);
        dim3 dataDim(n_total_clusters);
        dim3 vectorDim(n_dim);
        
        // cuBLAS句柄
        cublasHandle_t handle;
    
        // 分配设备内存
        int *d_index;
        {
            CUDATimer timer_manage("Step 1: GPU Memory Allocation");
    
            cudaMalloc(&d_inner_product, n_query * n_total_clusters * sizeof(float));/*存储各个query需要查找的data向量的距离*/
            cudaMalloc(&d_index, n_query * n_total_clusters * sizeof(int));/*存储各个query需要查找的data向量的索引*/
            cudaMalloc(&d_top_nprobe_dist, n_query * n_probes * sizeof(float));/*存储topk距离*/
    
            cublasCreate(&handle);
        }
    
        // 复制数据到设备
        {
            // COUT_ENDL("begin data transfer");
    
            CUDATimer timer_trans1("Step 1: H2D Data Transfer");
    
            /* 使用 CUDA kernel 并行生成顺序索引 [0, 1, 2, ..., n_total_clusters-1] */
            // 线程模型：每个block处理一个query，每个block使用256个线程（或更少）
            const int threads_per_block = 256;
            dim3 block_dim((n_total_clusters < threads_per_block) ? n_total_clusters : threads_per_block);
            dim3 grid_dim(n_query);
            
            generate_sequence_indices_kernel<<<grid_dim, block_dim>>>(
                d_index, n_query, n_total_clusters);
            CHECK_CUDA_ERRORS;
    
            /* 初始化距离数组（为一个小于-1的负数） */
            thrust::fill(
                thrust::device_pointer_cast(d_top_nprobe_dist),/*使用pointer_cast不用创建临时对象*/
                thrust::device_pointer_cast(d_top_nprobe_dist) + (n_query * n_probes),  /* 使用元素数量而非字节数 */
                FLT_MAX
            );
            // cudaMemset((void*)d_top_nprobe_dist, (int)0xEF, n_query * k * sizeof(float)) /*也可以投机取巧用memset，正好将数组为一个非常大的负数*/
            // table_cuda_2D("topk cos distance", d_top_nprobe_dist, n_query, k);    
            // COUT_ENDL("finish data transfer");
        }
    
        /* 核函数执行 */
        {
            // COUT_ENDL("begin_kernel");
    
            CUDATimer timer_compute("Step 1: Kernel Execution: matrix multiply");

            /**
            * 使用cuBLAS进行矩阵乘法
            * cuBLAS默认使用列主序，leading dimension是行数
            * */ 
            cublasSgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N, 
                n_total_clusters, n_query, n_dim,                   
                &alpha, 
                d_cluster_centers, n_dim,            
                d_queries, n_dim,               
                &beta, 
                d_inner_product, n_total_clusters
            );    
            
            cudaDeviceSynchronize(); 
    
            // print_cuda_2D("inner product", d_inner_product, n_query, n_total_clusters);    
            // table_cuda_2D("topk index", d_topk_index, n_query, k);
            // table_cuda_2D("topk cos distance", d_top_nprobe_dist, n_query, k);
        }
    
        {
            CUDATimer timer_compute("Step 1: Kernel Execution: cos + topk");
    
            pgvector::fusion_cos_topk_warpsort::fusion_cos_topk_warpsort<float, int>(
                d_query_norm, d_cluster_centers_norm, d_inner_product, d_index,
                n_query, n_total_clusters, n_probes,  // 粗筛选择 n_probes 个 cluster
                d_top_nprobe_dist, d_top_nprobe_index,
                true /* select min */
            );

            cudaDeviceSynchronize(); 
            CHECK_CUDA_ERRORS;
            
            // 调试输出：检查粗筛结果
            if(false){
                int* h_top_nprobe_index = (int*)malloc(n_query * n_probes * sizeof(int));
                float* h_top_nprobe_dist = (float*)malloc(n_query * n_probes * sizeof(float));
                cudaMemcpy(h_top_nprobe_index, d_top_nprobe_index, 
                           n_query * n_probes * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_top_nprobe_dist, d_top_nprobe_dist, 
                           n_query * n_probes * sizeof(float), cudaMemcpyDeviceToHost);
                CHECK_CUDA_ERRORS;
                
                printf("[DEBUG GPU Coarse] Query 0 coarse results (top %d clusters):\n", n_probes);
                for (int p = 0; p < n_probes; ++p) {
                    printf("  Probe %d: cluster_id=%d, dist=%.6f\n", 
                           p, h_top_nprobe_index[p], h_top_nprobe_dist[p]);
                }
                if (n_query > 1) {
                    printf("[DEBUG GPU Coarse] Query 1 coarse results (top %d clusters):\n", n_probes);
                    for (int p = 0; p < n_probes; ++p) {
                        printf("  Probe %d: cluster_id=%d, dist=%.6f\n", 
                               p, h_top_nprobe_index[n_probes + p], h_top_nprobe_dist[n_probes + p]);
                    }
                }
                
                free(h_top_nprobe_index);
                free(h_top_nprobe_dist);
            }
        }
    
    
        {
            CUDATimer timer_manage2("Step 1: GPU Memory Free", false);
            cublasDestroy(handle);
            cudaFree(d_cluster_centers);
            cudaFree(d_inner_product);
            cudaFree(d_cluster_centers_norm);
            cudaFree(d_index);
            cudaFree(d_top_nprobe_dist);
        }
    
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS
        
        // 确保 d_top_nprobe_index 的数据已经写入完成
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS
    }

    // ------------------------------------------------------------------
    // Step 2. 将 query→cluster 粗筛结果转成 cluster 序列
    // ------------------------------------------------------------------
    // 确保 Step 1 完全完成后再开始 Step 2
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS
    int* d_cluster_query_offset = nullptr;
    int* d_cluster_query_data = nullptr;
    int* d_cluster_query_probe_indices = nullptr;
    int total_entries = 0;  // 总条目数

    {
        CUDATimer timer("Step 2: Convert query→cluster to cluster sequence (GPU)");
        
        // 第一步：在GPU上统计每个cluster有多少个query使用它
        int* d_cluster_query_count = nullptr;
        cudaMalloc(&d_cluster_query_count, n_total_clusters * sizeof(int));
        cudaMemset(d_cluster_query_count, 0, n_total_clusters * sizeof(int));
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
        
        dim3 count_block_dim(n_probes);  // 每个block处理一个query的n_probes个probe
        dim3 count_grid_dim(n_query);    // 每个block处理一个query
        count_cluster_queries_kernel<<<count_grid_dim, count_block_dim>>>(
            d_top_nprobe_index,
            d_cluster_query_count,
            n_query,
            n_probes,
            n_total_clusters
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
        
        // 调试：检查 d_cluster_query_count 的值
        if(false){
            int* h_cluster_query_count_debug = (int*)malloc(n_total_clusters * sizeof(int));
            cudaMemcpy(h_cluster_query_count_debug, d_cluster_query_count, 
                       n_total_clusters * sizeof(int), cudaMemcpyDeviceToHost);
            printf("[DEBUG GPU Step 2] d_cluster_query_count (first 10): ");
            for (int i = 0; i < std::min(10, n_total_clusters); ++i) {
                printf("c%d=%d ", i, h_cluster_query_count_debug[i]);
            }
            printf("\n");
            free(h_cluster_query_count_debug);
        }
        
        // 第二步：在GPU上构建CSR格式的offset数组（使用前缀和）
        cudaMalloc(&d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int));
        CHECK_CUDA_ERRORS;
        
        // 使用GPU前缀和计算offset数组
        compute_prefix_sum(d_cluster_query_count, d_cluster_query_offset, n_total_clusters, 0);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
        
        // 调试：检查前缀和计算后的 offset 数组
        if(false){
            int* h_cluster_query_offset_debug = (int*)malloc((n_total_clusters + 1) * sizeof(int));
            cudaMemcpy(h_cluster_query_offset_debug, d_cluster_query_offset, 
                       (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            printf("[DEBUG GPU Step 2] d_cluster_query_offset after prefix_sum (first 11): ");
            for (int i = 0; i < std::min(11, n_total_clusters + 1); ++i) {
                printf("offset[%d]=%d ", i, h_cluster_query_offset_debug[i]);
            }
            printf("\n");
            free(h_cluster_query_offset_debug);
        }
        
        // 获取总条目数（需要从GPU读取最后一个元素）
        int total_entries_host = 0;
        cudaMemcpy(&total_entries_host, d_cluster_query_offset + n_total_clusters, 
                   sizeof(int), cudaMemcpyDeviceToHost);
        total_entries = total_entries_host;
        CHECK_CUDA_ERRORS;
        
        // 初始化写入位置数组（从offset复制，跳过最后一个元素）
        int* d_cluster_write_pos = nullptr;
        cudaMalloc(&d_cluster_write_pos, n_total_clusters * sizeof(int));
        cudaMemcpy(d_cluster_write_pos, d_cluster_query_offset, 
                   n_total_clusters * sizeof(int), cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERRORS;
        
        // 调试：检查 d_cluster_write_pos 的初始值
        if(false){
            int* h_cluster_write_pos_debug = (int*)malloc(n_total_clusters * sizeof(int));
            cudaMemcpy(h_cluster_write_pos_debug, d_cluster_write_pos, 
                       n_total_clusters * sizeof(int), cudaMemcpyDeviceToHost);
            printf("[DEBUG GPU Step 2] d_cluster_write_pos initial values (first 10): ");
            for (int i = 0; i < std::min(10, n_total_clusters); ++i) {
                printf("c%d=%d ", i, h_cluster_write_pos_debug[i]);
            }
            printf("\n");
            free(h_cluster_write_pos_debug);
        }
        
        // 第三步：在GPU上分配输出数组
        cudaMalloc(&d_cluster_query_data, total_entries * sizeof(int));
        cudaMalloc(&d_cluster_query_probe_indices, total_entries * sizeof(int));
        // 初始化输出数组为无效值（用于调试）
        cudaMemset(d_cluster_query_data, 0xFF, total_entries * sizeof(int));
        cudaMemset(d_cluster_query_probe_indices, 0xFF, total_entries * sizeof(int));
        CHECK_CUDA_ERRORS;
        
        // 调试：检查 d_top_nprobe_index 的值
        if(false){
            int* h_top_nprobe_index_debug = (int*)malloc(n_query * n_probes * sizeof(int));
            cudaMemcpy(h_top_nprobe_index_debug, d_top_nprobe_index, 
                       n_query * n_probes * sizeof(int), cudaMemcpyDeviceToHost);
            printf("[DEBUG GPU Step 2] d_top_nprobe_index for Query 0: ");
            for (int p = 0; p < n_probes; ++p) {
                printf("p%d=cluster%d ", p, h_top_nprobe_index_debug[p]);
            }
            printf("\n");
            free(h_top_nprobe_index_debug);
        }
        
        // 第四步：在GPU上构建CSR格式的数据
        build_cluster_query_mapping_kernel<<<count_grid_dim, count_block_dim>>>(
            d_top_nprobe_index,
            d_cluster_query_offset,
            d_cluster_query_data,
            d_cluster_query_probe_indices,
            d_cluster_write_pos,
            n_query,
            n_probes,
            n_total_clusters
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
        
        // 调试输出：检查Step 2的cluster-query映射
        if(false){
            int* h_cluster_query_offset = (int*)malloc((n_total_clusters + 1) * sizeof(int));
            cudaMemcpy(h_cluster_query_offset, d_cluster_query_offset, 
                       (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            CHECK_CUDA_ERRORS;
            
            printf("[DEBUG GPU Step 2] Cluster-query mapping (first 10 clusters, total_entries=%d):\n", total_entries);
            for (int c = 0; c < std::min(10, n_total_clusters); ++c) {
                int start = h_cluster_query_offset[c];
                int end = h_cluster_query_offset[c + 1];
                printf("  Cluster %d: offset=[%d, %d), count=%d\n", c, start, end, end - start);
                if (end > start && end <= total_entries) {
                    int* h_cluster_query_data = (int*)malloc((end - start) * sizeof(int));
                    int* h_cluster_query_probe_indices = (int*)malloc((end - start) * sizeof(int));
                    cudaMemcpy(h_cluster_query_data, d_cluster_query_data + start, 
                               (end - start) * sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_cluster_query_probe_indices, d_cluster_query_probe_indices + start, 
                               (end - start) * sizeof(int), cudaMemcpyDeviceToHost);
                    printf("    Data: [");
                    for (int i = 0; i < std::min(5, end - start); ++i) {
                        printf("(q=%d,p=%d) ", h_cluster_query_data[i], h_cluster_query_probe_indices[i]);
                    }
                    if (end - start > 5) printf("...");
                    printf("]\n");
                    free(h_cluster_query_data);
                    free(h_cluster_query_probe_indices);
                } else if (end > start) {
                    printf("    [ERROR] end=%d > total_entries=%d\n", end, total_entries);
                }
            }
            free(h_cluster_query_offset);
        }
        
        cudaFree(d_cluster_query_count);
        cudaFree(d_cluster_write_pos);        
        cudaFree(d_top_nprobe_index);
        CHECK_CUDA_ERRORS;
    }


        // ------------------------------------------------------------------
    // Step 3. 精筛：上传 cluster 向量 + cluster 查询映射，调用 GPU kernel
        // ------------------------------------------------------------------
    {
        CUDATimer timer("Step 3: Fine Search (fine_screen_top_n)");

        int capacity = 32;
        int* probe_query_offsets_host = nullptr;
        constexpr int kQueriesPerBlock = 8;
        float* d_topk_dist_candidate = nullptr;
        int* d_topk_index_candidate = nullptr;
        
        // 配置kernel launch
        // 固定probe版本：每个block处理一个probe的多个query
        dim3 block(kQueriesPerBlock * 32);  // 8个warp，每个warp 32个线程
    
        int max_queries_per_probe = 0;
    
        {
            CUDATimer timer_init("Init Invalid Values Kernel", ENABLE_CUDA_TIMING);
    
            // 选择合适的Capacity（必须是2的幂，且 > k）
            while (capacity < k) capacity <<= 1;
            capacity = std::min(capacity, kMaxCapacity);
            
            CHECK_CUDA_ERRORS;
            
            // 计算max_queries_per_probe用于launch函数（用于计算grid配置）
            // d_cluster_query_offset 的大小是 n_total_clusters + 1，因为每个 cluster 都是一个 probe
            probe_query_offsets_host = (int*)malloc((n_total_clusters + 1) * sizeof(int));
            cudaMemcpy(probe_query_offsets_host, d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    
            for (int i = 0; i < n_total_clusters; ++i) {
                int n_queries = probe_query_offsets_host[i + 1] - probe_query_offsets_host[i];
                max_queries_per_probe = std::max(max_queries_per_probe, n_queries);
            }
            CHECK_CUDA_ERRORS;
    
            // 按query组织的结果 [n_query][n_probes][k]
            cudaMalloc(&d_topk_dist_candidate, n_query * n_probes * k * sizeof(float));
            cudaMalloc(&d_topk_index_candidate, n_query * n_probes * k * sizeof(int));
        
            // 初始化输出内存为无效值（FLT_MAX 和 -1）
            dim3 init_block(512);
            int init_grid_size = (n_query * n_probes * k + init_block.x - 1) / init_block.x;
            
            // // 检查 grid 大小是否超过 CUDA 限制（65535）
            // if (init_grid_size > 65535) {
            //     printf("[ERROR] Grid size too large: %d (max 65535), total_size=%d\n", 
            //            init_grid_size, total_size);
            //     cudaFree(d_topk_dist_candidate);
            //     cudaFree(d_topk_index_candidate);
            //     return;
            // }
            
            dim3 init_grid(init_grid_size);
            init_invalid_values_kernel<<<init_grid, init_block>>>(
                d_topk_dist_candidate,
                d_topk_index_candidate,
                n_query * n_probes * k
            );
        }
        int max_query_batches = 0;
        {
            CUDATimer timer_kernel("Indexed Inner Product with TopK Kernel (v3 fixed probe)", ENABLE_CUDA_TIMING);
            
            // 验证 grid 配置
            // grid.x 使用 n_total_clusters，因为每个 cluster 都是一个 probe
            max_query_batches = (max_queries_per_probe + kQueriesPerBlock - 1) / kQueriesPerBlock;
            dim3 grid(n_total_clusters, max_query_batches, 1);
            
            // 根据capacity选择kernel实例
            if (capacity <= 32) {
                launch_indexed_inner_product_with_topk_kernel_v3_fixed_probe<64, true, kQueriesPerBlock>(
                    block,
                    n_dim,
                    d_queries,
                    d_cluster_vectors,
                    d_probe_vector_offset,
                    d_probe_vector_count,
                    d_cluster_query_data,
                    d_cluster_query_offset,
                    d_cluster_query_probe_indices,
                    d_query_norm,
                    d_cluster_vector_norm,
                    n_total_clusters,  // 传递总的 cluster 数量（用于grid.x和检查probe_id）
                    n_probes,  // 传递每个query的probe数量（用于检查probe_index_in_query和计算输出位置）
                    max_queries_per_probe,
                    k,
                    d_topk_dist_candidate,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                    d_topk_index_candidate,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                    0
                );
            } else if (capacity <= 64) {
                launch_indexed_inner_product_with_topk_kernel_v3_fixed_probe<128, true, kQueriesPerBlock>(
                    block,
                    n_dim,
                    d_queries,  // 修正：使用 d_queries 而不是 d_query_group
                    d_cluster_vectors,  // 修正：使用 d_cluster_vectors 而不是 d_cluster_vector
                    d_probe_vector_offset,
                    d_probe_vector_count,
                    d_cluster_query_data,
                    d_cluster_query_offset,
                    d_cluster_query_probe_indices,
                    d_query_norm,
                    d_cluster_vector_norm,
                    n_total_clusters,  // 传递总的 cluster 数量（用于grid.x和检查probe_id）
                    n_probes,  // 传递每个query的probe数量（用于检查probe_index_in_query和计算输出位置）
                    max_queries_per_probe,
                    k,
                    d_topk_dist_candidate,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                    d_topk_index_candidate,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                    0
                );
            } else {
                launch_indexed_inner_product_with_topk_kernel_v3_fixed_probe<256, true, kQueriesPerBlock>(
                    block,
                    n_dim,
                    d_queries, 
                    d_cluster_vectors, 
                    d_probe_vector_offset,
                    d_probe_vector_count,
                    d_cluster_query_data,
                    d_cluster_query_offset,
                    d_cluster_query_probe_indices,
                    d_query_norm,
                    d_cluster_vector_norm,
                    n_total_clusters,  // 传递总的 cluster 数量（用于grid.x和检查probe_id）
                    n_probes,  // 传递每个query的probe数量（用于检查probe_index_in_query和计算输出位置）
                    max_queries_per_probe,
                    k,
                    d_topk_dist_candidate,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                    d_topk_index_candidate,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                    0
                );
            }
            cudaDeviceSynchronize();
            CHECK_CUDA_ERRORS;
        }
            
        // 规约：将 [n_query][n_probes][k] 归并为 [n_query][k]
        // 在GPU上完成，避免CPU-GPU数据复制
        {
            CUDATimer timer_reduce("Reduce probe results to query top-k", ENABLE_CUDA_TIMING);
            
            select_k<float, int>(
                d_topk_dist_candidate, n_query, n_probes * k, k,
                d_topk_dist, d_topk_index, true, 0
            );
            cudaDeviceSynchronize();
            CHECK_CUDA_ERRORS;
            
            // 2. 映射回原始向量索引
            // select_k返回的索引是候选数组中的位置，需要映射回原始向量索引
            dim3 map_block(256);
            dim3 map_grid((n_query * k + map_block.x - 1) / map_block.x);
            map_candidate_indices_kernel<<<map_grid, map_block>>>(
                d_topk_index_candidate,  // 使用原数组作为候选索引
                d_topk_index,
                n_query,
                n_probes,
                k
            );
            CHECK_CUDA_ERRORS;
            
            // 清理临时内存
            cudaFree(d_topk_dist_candidate);
            cudaFree(d_topk_index_candidate);
        }
        
        // 确保kernel执行完成后再复制结果
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;

        cudaMemcpy(topk_dist[0], d_topk_dist, 
                   n_query * k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(topk_index[0], d_topk_index, 
                   n_query * k * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;

        cudaFree(d_cluster_vectors);
        cudaFree(d_cluster_vector_norm);
        free(probe_query_offsets_host);
        CHECK_CUDA_ERRORS;
    }
    
    // 释放Step 2中分配的GPU内存
    cudaFree(d_cluster_query_offset);
    cudaFree(d_cluster_query_data);
    cudaFree(d_cluster_query_probe_indices);

    // 释放Step 0中分配的GPU内存
    cudaFree(d_probe_vector_offset);
    cudaFree(d_probe_vector_count);

    cudaFree(d_queries);
    cudaFree(d_query_norm);
    
    CHECK_CUDA_ERRORS;
}

void run_integrate_pipeline() {
    // TODO: 后续补充粗筛 + 精筛整体调度
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS
}
