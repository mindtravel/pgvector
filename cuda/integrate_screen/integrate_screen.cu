/* 必须在任何头文件之前包含limits.h，以便Thrust可以使用CHAR_MIN等宏 */
#ifndef _LIMITS_H_
#define _LIMITS_H_
#endif
#include <limits.h>
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

    dim3 queryDim(n_query);
    dim3 dataDim(n_total_clusters);
    dim3 vectorDim(n_dim);
    dim3 probeDim(n_probes);

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
        {
            CUDATimer timer("Step 0: Copy Cluster Data");
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
        }
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
            dim3 block_dim((n_total_clusters < 256) ? n_total_clusters : 256);
            generate_sequence_indices_kernel<<<queryDim, block_dim>>>(
                d_index, n_query, n_total_clusters);
            CHECK_CUDA_ERRORS;
    
            /* 初始化距离数组（使用fill kernel替代thrust::fill） */
            dim3 fill_block(256);
            int fill_grid_size = (n_query * n_probes + fill_block.x - 1) / fill_block.x;
            dim3 fill_grid(fill_grid_size);
            fill_kernel<<<fill_grid, fill_block>>>(
                d_top_nprobe_dist,
                FLT_MAX,
                n_query * n_probes
            );
            // cudaMemset((void*)d_top_nprobe_dist, (int)0xEF, n_query * k * sizeof(float)) /*也可以投机取巧用memset，正好将数组为一个非常大的负数*/
            // table_cuda_2D("topk cos distance", d_top_nprobe_dist, n_query, k);    
            // COUT_ENDL("finish data transfer");
        }
    
        /* 核函数执行 */
        {    
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
    }

    // ------------------------------------------------------------------
    // Step 2. 将 query→cluster 粗筛结果转成 entry 数据（v5 entry-based）
    // ------------------------------------------------------------------
    // 确保 Step 1 完全完成后再开始 Step 2
    // 首先构建 cluster-query 映射（CSR格式），然后转换为 entry 数据
    int* d_cluster_query_offset = nullptr;
    int* d_cluster_query_data = nullptr;
    int* d_cluster_query_probe_indices = nullptr;
    
    // Entry数据结构（GPU内存）
    int* d_entry_cluster_id = nullptr;
    int* d_entry_query_start = nullptr;
    int* d_entry_query_count = nullptr;
    int* d_entry_queries = nullptr;
    int* d_entry_probe_indices = nullptr;
    int n_entry = 0;
    constexpr int kQueriesPerBlock = 8;

    {
        CUDATimer timer("Step 2: Build entry data (GPU)");
        
        // 第一步：在GPU上统计每个cluster有多少个query使用它
        int* d_cluster_query_count = nullptr;
        cudaMalloc(&d_cluster_query_count, n_total_clusters * sizeof(int));
        cudaMemset(d_cluster_query_count, 0, n_total_clusters * sizeof(int));
        CHECK_CUDA_ERRORS;
        
        count_cluster_queries_kernel<<<queryDim, probeDim>>>(
            d_top_nprobe_index,
            d_cluster_query_count,
            n_query,
            n_probes,
            n_total_clusters
        );
        CHECK_CUDA_ERRORS;
        
        // 第二步：在GPU上构建CSR格式的offset数组（使用前缀和）
        cudaMalloc(&d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int));
        CHECK_CUDA_ERRORS;
        
        compute_prefix_sum(d_cluster_query_count, d_cluster_query_offset, n_total_clusters, 0);
        CHECK_CUDA_ERRORS;
        
        // 获取总条目数（需要从GPU读取最后一个元素）
        int total_entries = 0;
        cudaMemcpy(&total_entries, d_cluster_query_offset + n_total_clusters, 
                   sizeof(int), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;
        
        // 初始化写入位置数组（从offset复制，跳过最后一个元素）
        int* d_cluster_write_pos = nullptr;
        cudaMalloc(&d_cluster_write_pos, n_total_clusters * sizeof(int));
        cudaMemcpy(d_cluster_write_pos, d_cluster_query_offset, 
                   n_total_clusters * sizeof(int), cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERRORS;
        
        // 第三步：在GPU上分配cluster-query映射数组
        cudaMalloc(&d_cluster_query_data, total_entries * sizeof(int));
        cudaMalloc(&d_cluster_query_probe_indices, total_entries * sizeof(int));
        CHECK_CUDA_ERRORS;
        
        // 第四步：在GPU上构建CSR格式的cluster-query映射
        build_cluster_query_mapping_kernel<<<queryDim, probeDim>>>(
            d_top_nprobe_index,
            d_cluster_query_offset,
            d_cluster_query_data,
            d_cluster_query_probe_indices,
            d_cluster_write_pos,
            n_query,
            n_probes,
            n_total_clusters
        );
        CHECK_CUDA_ERRORS;
        
        // 第五步：在GPU上计算每个cluster会产生多少个entry
        int* d_entry_count_per_cluster = nullptr;
        cudaMalloc(&d_entry_count_per_cluster, n_total_clusters * sizeof(int));
        CHECK_CUDA_ERRORS;
        
        dim3 clusterDim(n_total_clusters);
        dim3 blockDim_entry(1);
        count_entries_per_cluster_kernel<<<clusterDim, blockDim_entry>>>(
            d_cluster_query_offset,
            d_entry_count_per_cluster,
            n_total_clusters,
            kQueriesPerBlock
        );
        CHECK_CUDA_ERRORS;
        
        // 第六步：计算entry的offset数组（使用前缀和）
        int* d_entry_offset = nullptr;
        cudaMalloc(&d_entry_offset, (n_total_clusters + 1) * sizeof(int));
        CHECK_CUDA_ERRORS;
        
        compute_prefix_sum(d_entry_count_per_cluster, d_entry_offset, n_total_clusters, 0);
        CHECK_CUDA_ERRORS;
        
        // 获取entry总数（从GPU读取最后一个元素）
        cudaMemcpy(&n_entry, d_entry_offset + n_total_clusters, 
                   sizeof(int), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;
        
        // 第七步：计算每个cluster在entry queries数组中的起始位置
        // 这等于 d_cluster_query_offset（因为每个cluster的query数量不变，只是被分组为entry）
        int* d_entry_query_offset = nullptr;
        cudaMalloc(&d_entry_query_offset, (n_total_clusters + 1) * sizeof(int));
        cudaMemcpy(d_entry_query_offset, d_cluster_query_offset, 
                   (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERRORS;
        
        // 第八步：分配entry数据数组
        if (n_entry > 0) {
            cudaMalloc(&d_entry_cluster_id, n_entry * sizeof(int));
            cudaMalloc(&d_entry_query_start, n_entry * sizeof(int));
            cudaMalloc(&d_entry_query_count, n_entry * sizeof(int));
            
            // 计算总的query数量（等于total_entries，因为每个entry-query对对应一个query）
            cudaMalloc(&d_entry_queries, total_entries * sizeof(int));
            cudaMalloc(&d_entry_probe_indices, total_entries * sizeof(int));
            CHECK_CUDA_ERRORS;
            
            // 第九步：在GPU上构建entry数据
            build_entry_data_kernel<<<clusterDim, blockDim_entry>>>(
                d_cluster_query_offset,
                d_cluster_query_data,
                d_cluster_query_probe_indices,
                d_entry_offset,
                d_entry_query_offset,
                d_entry_cluster_id,
                d_entry_query_start,
                d_entry_query_count,
                d_entry_queries,
                d_entry_probe_indices,
                n_total_clusters,
                kQueriesPerBlock
            );
            CHECK_CUDA_ERRORS;
        }
        
        // 清理临时内存
        cudaFree(d_entry_query_offset);
        
        // 清理临时内存
        cudaFree(d_cluster_query_count);
        cudaFree(d_cluster_write_pos);        
        cudaFree(d_entry_count_per_cluster);
        cudaFree(d_entry_offset);
        cudaFree(d_top_nprobe_index);
        CHECK_CUDA_ERRORS;
    }


        // ------------------------------------------------------------------
    // Step 3. 精筛：使用 v5 entry-based 版本
        // ------------------------------------------------------------------
    {
        CUDATimer timer("Step 3: Fine Search (v5 entry-based)");

        int capacity = 32;
        float* d_topk_dist_candidate = nullptr;
        int* d_topk_index_candidate = nullptr;
        
        // 配置kernel launch
        // v5 entry-based版本：每个block处理一个entry（一个cluster + 一组query）
        dim3 block(kQueriesPerBlock * 32);  // 8个warp，每个warp 32个线程
    
        {
            CUDATimer timer_init("Init Invalid Values Kernel", ENABLE_CUDA_TIMING);
    
            // 选择合适的Capacity（必须是2的幂，且 > k）
            while (capacity < k) capacity <<= 1;
            capacity = std::min(capacity, kMaxCapacity);
            
            CHECK_CUDA_ERRORS;
    
            // 按query组织的结果 [n_query][n_probes][k]
            cudaMalloc(&d_topk_dist_candidate, n_query * n_probes * k * sizeof(float));
            cudaMalloc(&d_topk_index_candidate, n_query * n_probes * k * sizeof(int));
        
            // 初始化输出内存为无效值（FLT_MAX 和 -1）
            dim3 init_block(512);
            int init_grid_size = (n_query * n_probes * k + init_block.x - 1) / init_block.x;
            dim3 init_grid(init_grid_size);
            init_invalid_values_kernel<<<init_grid, init_block>>>(
                d_topk_dist_candidate,
                d_topk_index_candidate,
                n_query * n_probes * k
            );
            CHECK_CUDA_ERRORS;
        }
        
        {
            CUDATimer timer_kernel("Indexed Inner Product with TopK Kernel (v5 entry-based)", ENABLE_CUDA_TIMING);
            
            if (n_entry == 0) {
                // 没有entry，直接跳过（所有结果已经是FLT_MAX和-1）
            } else {
            // 根据capacity选择kernel实例
            if (capacity <= 32) {
                    launch_indexed_inner_product_with_topk_kernel_v5_entry_based<64, true, kQueriesPerBlock>(
                    block,
                    n_dim,
                    d_queries,
                    d_cluster_vectors,
                    d_probe_vector_offset,
                    d_probe_vector_count,
                        d_entry_cluster_id,
                        d_entry_query_start,
                        d_entry_query_count,
                        d_entry_queries,
                        d_entry_probe_indices,
                    d_query_norm,
                    d_cluster_vector_norm,
                        n_entry,
                        n_probes,
                    k,
                    d_topk_dist_candidate,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                    d_topk_index_candidate,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                    0
                );
            } else if (capacity <= 64) {
                    launch_indexed_inner_product_with_topk_kernel_v5_entry_based<128, true, kQueriesPerBlock>(
                    block,
                    n_dim,
                        d_queries,
                        d_cluster_vectors,
                    d_probe_vector_offset,
                    d_probe_vector_count,
                        d_entry_cluster_id,
                        d_entry_query_start,
                        d_entry_query_count,
                        d_entry_queries,
                        d_entry_probe_indices,
                    d_query_norm,
                    d_cluster_vector_norm,
                        n_entry,
                        n_probes,
                    k,
                    d_topk_dist_candidate,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                    d_topk_index_candidate,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                    0
                );
            } else {
                    launch_indexed_inner_product_with_topk_kernel_v5_entry_based<256, true, kQueriesPerBlock>(
                    block,
                    n_dim,
                    d_queries, 
                    d_cluster_vectors, 
                    d_probe_vector_offset,
                    d_probe_vector_count,
                        d_entry_cluster_id,
                        d_entry_query_start,
                        d_entry_query_count,
                        d_entry_queries,
                        d_entry_probe_indices,
                    d_query_norm,
                    d_cluster_vector_norm,
                        n_entry,
                        n_probes,
                    k,
                    d_topk_dist_candidate,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                    d_topk_index_candidate,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                    0
                );
            }
            }
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
        CHECK_CUDA_ERRORS;
    }
    
    // 释放Step 2中分配的GPU内存
    cudaFree(d_cluster_query_offset);
    cudaFree(d_cluster_query_data);
    cudaFree(d_cluster_query_probe_indices);
    
    // 释放entry数据
    if (d_entry_cluster_id != nullptr) {
        cudaFree(d_entry_cluster_id);
    }
    if (d_entry_query_start != nullptr) {
        cudaFree(d_entry_query_start);
    }
    if (d_entry_query_count != nullptr) {
        cudaFree(d_entry_query_count);
    }
    if (d_entry_queries != nullptr) {
        cudaFree(d_entry_queries);
    }
    if (d_entry_probe_indices != nullptr) {
        cudaFree(d_entry_probe_indices);
    }

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
