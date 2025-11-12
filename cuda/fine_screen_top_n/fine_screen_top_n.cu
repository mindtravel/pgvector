#include "fine_screen_top_n.cuh"
#include "../l2norm.cuh"
#include "../unit_tests/common/test_utils.cuh"
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <limits.h>
#include <float.h>
#include <vector>

#define ENABLE_CUDA_TIMING 0

/**
 * 棰勭暀鐨剋arpsort鎺ュ彛锛岀敤浜庡湪瀵勫瓨鍣ㄤ腑缁存姢灞€閮╰op-k
 * 鍙傛暟寰呭畾锛屽疄鐜板緟瀹?
 */
__device__ void cluster_warpsort_topk(
    float* local_distances,    // 褰撳墠cluster鐨勮窛绂绘暟缁?
    int* local_indices,        // 瀵瑰簲鐨勭储寮曟暟缁?
    int cluster_vector_count,   // 褰撳墠cluster鐨勫悜閲忔暟閲?
    int k,                     // top-k鏁伴噺
    float* output_distances,   // 杈撳嚭璺濈
    int* output_indices        // 杈撳嚭绱㈠紩
) {
    // 瀹炵幇寰呭畾
    // 杩欓噷鏆傛椂鐢ㄧ畝鍗曠殑鎺掑簭瀹炵幇
    for (int i = 0; i < cluster_vector_count - 1; i++) {
        for (int j = i + 1; j < cluster_vector_count; j++) {
            if (local_distances[i] > local_distances[j]) {
                // 浜ゆ崲璺濈
                float temp_dist = local_distances[i];
                local_distances[i] = local_distances[j];
                local_distances[j] = temp_dist;
                
                // 浜ゆ崲绱㈠紩
                int temp_idx = local_indices[i];
                local_indices[i] = local_indices[j];
                local_indices[j] = temp_idx;
            }
        }
    }
    
    // 澶嶅埗鍓峩涓粨鏋?
    for (int i = 0; i < k && i < cluster_vector_count; i++) {
        output_distances[i] = local_distances[i];
        output_indices[i] = local_indices[i];
    }
}

/**
 * 璁＄畻cluster涓悜閲忎笌query鐨凩2璺濈骞堕€夋嫨top-k
 * 姣忎釜block澶勭悊涓€涓猚luster
 */
__global__ void cluster_l2_distance_kernel(
    const float* __restrict__ d_query_group,
    const float* __restrict__ d_query_norm,
    const float* __restrict__ d_cluster_vector,
    const float* __restrict__ d_cluster_vector_norm,
    const int* __restrict__ d_query_cluster_group,
    const int* __restrict__ d_cluster_query_offset,
    const int* __restrict__ d_cluster_query_data,
    const int* __restrict__ d_cluster_map,
    const int* __restrict__ d_cluster_vector_index,
    const int* __restrict__ d_cluster_vector_num,
    int n_query, int n_cluster, int n_dim, int n_topn,
    int max_cluster_vector_count, int distinct_cluster_count, int tol_vector,
    int* __restrict__ d_query_mutex,
    int* __restrict__ d_topn_index,
    float* __restrict__ d_topn_dist
) {
    int cluster_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    if (cluster_idx >= distinct_cluster_count || thread_idx >= d_cluster_vector_num[cluster_idx]) return;
    if (thread_idx >= blockDim.x) return;
    // 鍏变韩鍐呭瓨锛氱紦瀛楲2鑼冩暟鍜宑luster鍚戦噺鏁版嵁
    extern __shared__ float shared_mem[];
    float* s_query_norm = shared_mem;
    float* s_cluster_norm = s_query_norm + n_query;
    
    // 鍙湁绗竴涓嚎绋嬭绠梣uery鑼冨洿锛岄伩鍏嶈秺鐣?
    int query_start, query_count;
    if (thread_idx == 0) {
        // 杈圭晫妫€鏌ワ細纭繚涓嶈秺鐣岃闂?
        if (cluster_idx >= distinct_cluster_count) {
            query_count = 0;
        } else {
            query_start = d_cluster_query_offset[cluster_idx];
            
            // 瀵逛簬鏈€鍚庝竴涓猚luster锛屼娇鐢ㄦ€绘暟浣滀负缁撴潫浣嶇疆
            if (cluster_idx + 1 >= distinct_cluster_count) {
                // 鏈€鍚庝竴涓猚luster锛歲uery_count = 鎬籷uery鏁?- query_start
                query_count = n_query - query_start;
            } else {
                query_count = d_cluster_query_offset[cluster_idx + 1] - query_start;
            }
            
            // 棰濆鐨勮秺鐣屾鏌?
            if (query_start >= n_query || query_start + query_count > n_query || query_count < 0) {
                query_count = 0;
            }
        }
    }
    if (query_count == 0) return;

    // 鑾峰彇褰撳墠cluster鐨勫悜閲忎俊鎭?
    int vector_start_idx = d_cluster_vector_index[cluster_idx];
    int vector_count = d_cluster_vector_num[cluster_idx];
    
    // 淇锛氭坊鍔犺竟鐣屾鏌ワ紝纭繚鍚戦噺绱㈠紩鏈夋晥
    if (vector_start_idx < 0 || vector_count <= 0 || vector_start_idx + vector_count > tol_vector) {
        return;
    }
    __syncthreads();
    
    
    // 鍔犺浇L2鑼冩暟鍒板叡浜唴瀛?
    if (thread_idx < n_query) {
        s_query_norm[thread_idx] = d_query_norm[thread_idx];
    }
    // 淇锛氬姞杞藉綋鍓峜luster鐨勫悜閲廘2鑼冩暟锛屾坊鍔犺竟鐣屾鏌?
    if (thread_idx < vector_count && thread_idx < max_cluster_vector_count) {
        int global_vec_idx = vector_start_idx + thread_idx;
        if (global_vec_idx < tol_vector) {
            s_cluster_norm[thread_idx] = d_cluster_vector_norm[global_vec_idx];
        }
    }
    __syncthreads();
    
    // 姣忎釜绾跨▼澶勭悊cluster涓殑閮ㄥ垎鍚戦噺
    int vectors_per_thread = (vector_count + blockDim.x - 1) / blockDim.x;
    int start_vec = thread_idx * vectors_per_thread;
    int end_vec = min(start_vec + vectors_per_thread, vector_count);
    
    // 涓烘瘡涓猶uery璁＄畻L2璺濈骞剁淮鎶ゅ眬閮╰opk
    for (int q = 0; q < query_count; q++) {
        int query_idx = query_start + q;
        
        
        
        // 璁＄畻褰撳墠query涓巆luster涓悜閲忕殑L2璺濈
        for (int vec_idx = start_vec; vec_idx < end_vec; vec_idx++) {
            int global_vec_idx = vector_start_idx + vec_idx;
            
            // 淇锛氭坊鍔犺竟鐣屾鏌ワ紝纭繚鍏ㄥ眬鍚戦噺绱㈠紩鏈夋晥
            if (global_vec_idx < 0 || global_vec_idx >= tol_vector) {
                continue;
            }
            
            // 璁＄畻L2璺濈鐨勫钩鏂癸紙浣跨敤L2鑼冩暟浼樺寲锛?   todo 鍏跺疄杩欓噷涔熷彲浠ユ彁鍓嶈绠楀嚭鏉?鍚庣画鐪嬪摢涓€ц兘鏇村ソ涓€鐐瑰惂
            float dot_product = 0.0f;
            for (int dim = 0; dim < n_dim; dim++) {
                dot_product += d_query_group[query_idx * n_dim + dim] * 
                              d_cluster_vector[global_vec_idx * n_dim + dim];
            }
            
            // L2璺濈骞虫柟 = ||q||^2 + ||v||^2 - 2*q路v
            float distance_squared = s_query_norm[query_idx] + s_cluster_norm[vec_idx] - 2.0f * dot_product;
            
            // 鍙栧钩鏂规牴寰楀埌瀹為檯璺濈
            float distance = sqrtf(fmaxf(0.0f, distance_squared));
            
            // // 鎻掑叆鍒板綋鍓峲uery鐨勫眬閮╰opk涓?
            // for (int k = 0; k < n_topn; k++) {
            //     if (distance < query_local_topk_dist[k]) {
            //         // 鍚戝悗绉诲姩鍏冪礌
            //         for (int m = n_topn - 1; m > k; m--) {
            //             query_local_topk_dist[m] = query_local_topk_dist[m-1];
            //             query_local_topk_idx[m] = query_local_topk_idx[m-1];
            //         }
            //         // 鎻掑叆鏂板厓绱?
            //         query_local_topk_dist[k] = distance;
            //         query_local_topk_idx[k] = global_vec_idx;
            //         break;
            //     }
            // }
        }
        
    }
    
    __syncthreads();
    
    // 鍐欏叆鏄惧瓨瀵瑰簲浣嶇疆 - 浣跨敤鍘熷瓙鎿嶄綔鍔犻攣
    // 姣忎釜绾跨▼澶勭悊鑷繁璐熻矗鐨剄uery鑼冨洿
    
    int queries_per_thread = (query_count + blockDim.x - 1) / blockDim.x;
    int start_query = thread_idx * queries_per_thread;
    int end_query = min(start_query + queries_per_thread, query_count);
    
    for (int q = start_query; q < end_query; q++) {
        int query_idx = query_start + q;
        if (query_idx >= n_query) continue;
        // 浣跨敤鍘熷瓙鎿嶄綔鑾峰彇閿?
        while (atomicCAS(&d_query_mutex[query_idx], 0, 1) != 0) {
            // 鑷棆绛夊緟
        }
        
        // 鍚堝苟灞€閮╰opk鍒板叏灞€topk
        // 淇锛氭坊鍔犺竟鐣屾鏌ワ紝纭繚绱㈠紩涓嶈秺鐣?
        for (int k = 0; k < n_topn && k < vector_count; k++) {
            // 纭繚鍏ㄥ眬鍚戦噺绱㈠紩鍦ㄦ湁鏁堣寖鍥村唴
            if (query_idx * n_topn + k >= n_query * n_topn) continue;
            d_topn_index[query_idx * n_topn + k] = vector_start_idx + k;
            // TODO: 杩欓噷搴旇浣跨敤瀹為檯璁＄畻鐨勮窛绂诲€硷紝鑰屼笉鏄复鏃跺€?
            // 闇€瑕佸疄鐜扮湡姝ｇ殑top-k閫夋嫨閫昏緫鏉ヨ幏鍙栨纭殑璺濈
            d_topn_dist[query_idx * n_topn + k] = 0.0f; // 涓存椂鍊硷紝闇€瑕佹浛鎹负瀹為檯璺濈
            
        }
        
        // 閲婃斁閿?
        atomicExch(&d_query_mutex[query_idx], 0);
    }

}

void fine_screen_top_n_old(
    float* h_query_group, int* h_query_cluster_group, int* h_cluster_query_offset, int* h_cluster_query_data,
    int* cluster_map,
    int* h_cluster_vector_index, int* h_cluster_vector_num, float** h_cluster_vector,
    int n_query, int n_cluster, int distinct_cluster_count, int n_dim, int n_topn, int max_cluster_id, int tol_vector,
    int max_cluster_vector_count,  // 鏂板锛氭渶澶ц仛绫诲悜閲忔暟閲?
    int* h_query_topn_index, float* h_query_topn_dist
) {
    (void)h_query_cluster_group;
    // 璁＄畻鍐呭瓨澶у皬
    size_t size_query_group = n_query * n_dim * sizeof(float);
    size_t size_query_cluster_group = n_query * n_cluster * sizeof(int); //姣忎釜query瀵瑰簲n涓猚luster
    size_t size_cluster_query_offset = distinct_cluster_count * sizeof(int);  // distinct cluster鏁伴噺
    size_t size_cluster_query_data = n_query * n_cluster * sizeof(int);  // 姣忎釜query瀵瑰簲n涓猚luster
    size_t size_cluster_map = distinct_cluster_count * sizeof(int);  // distinct cluster鏁伴噺
    size_t size_cluster_vector_index = distinct_cluster_count * sizeof(int);  // distinct cluster鏁伴噺
    size_t size_cluster_vector_num = distinct_cluster_count * sizeof(int);  // distinct cluster鏁伴噺
    size_t size_cluster_vector = tol_vector * n_dim * sizeof(float);  // 鎬诲悜閲忔暟閲?
    size_t size_topn_index = n_query * n_topn * sizeof(int);
    size_t size_topn_dist = n_query * n_topn * sizeof(float);
    
    // 鍒嗛厤璁惧鍐呭瓨
    float *d_query_group, *d_cluster_vector, *d_topn_dist, *d_query_norm, *d_cluster_vector_norm;
    int *d_query_cluster_group, *d_cluster_query_offset, *d_cluster_query_data;
    int *d_cluster_vector_index, *d_cluster_vector_num, *d_topn_index, *d_cluster_map, *d_query_mutex;
    
    dim3 clusterDim(tol_vector);
    dim3 vectorDim(n_dim);
    dim3 queryDim(n_query);
    // GPU鍐呭瓨鍒嗛厤
    cudaMalloc(&d_query_group, size_query_group);
    cudaMalloc(&d_query_cluster_group, size_query_cluster_group);
    cudaMalloc(&d_cluster_query_offset, size_cluster_query_offset);
    cudaMalloc(&d_cluster_query_data, size_cluster_query_data);
    cudaMalloc(&d_cluster_map, size_cluster_map);
    cudaMalloc(&d_cluster_vector_index, size_cluster_vector_index);
    cudaMalloc(&d_cluster_vector_num, size_cluster_vector_num);
    cudaMalloc(&d_cluster_vector, size_cluster_vector);
    cudaMalloc(&d_query_norm, n_query * sizeof(float));  // 瀛樺偍query鐨凩2鑼冩暟
    cudaMalloc(&d_cluster_vector_norm, tol_vector * sizeof(float));  // 瀛樺偍cluster鍚戦噺鐨凩2鑼冩暟
    cudaMalloc(&d_query_mutex, n_query * sizeof(int));  // 姣忎釜query涓€涓攣
    cudaMalloc(&d_topn_index, size_topn_index);
    cudaMalloc(&d_topn_dist, size_topn_dist);
    
    // 澶嶅埗鏁版嵁鍒拌澶囧唴瀛?
    cudaMemcpy(d_query_group, h_query_group, size_query_group, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_cluster_group, h_query_cluster_group, size_query_cluster_group, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_query_offset, h_cluster_query_offset, size_cluster_query_offset, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_query_data, h_cluster_query_data, size_cluster_query_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_map, cluster_map, size_cluster_map, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector_index, h_cluster_vector_index, size_cluster_vector_index, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector_num, h_cluster_vector_num, size_cluster_vector_num, cudaMemcpyHostToDevice);
    // 浣跨敤cudaMemcpy2D浠庝簩缁存寚閽堝鍒禼luster鍚戦噺鏁版嵁鍒拌澶囧唴瀛?
    // h_cluster_vector[i] 鎸囧悜绗琲涓猚luster鐨勫悜閲忔暟鎹?
    cudaMemcpy2D(
        d_cluster_vector,                    // 鐩爣璁惧鍐呭瓨
        n_dim * sizeof(float),              // 鐩爣琛岄棿璺?
        h_cluster_vector[0],                // 婧愪富鏈哄唴瀛橈紙绗竴涓猚luster鐨勫悜閲忥級
        n_dim * sizeof(float),              // 婧愯闂磋窛
        n_dim * sizeof(float),              // 姣忚瀛楄妭鏁?
        tol_vector,                          // 琛屾暟锛坈luster鏁伴噺锛?
        cudaMemcpyHostToDevice
    );
    
    // 鍒濆鍖栭攣鏁扮粍鍜宼op-k鏁扮粍
    cudaMemset(d_query_mutex, 0, n_query * sizeof(int)); // 閿佸垵濮嬪寲涓?锛堟湭閿佸畾锛?
    thrust::fill(
        thrust::device_pointer_cast(d_topn_dist),
        thrust::device_pointer_cast(d_topn_dist) + (n_query * n_topn),
        FLT_MAX
    );
    
    // TODO: 鍦ㄨ繖閲屾坊鍔犲疄闄呯殑kernel璁＄畻閫昏緫
    // 璁＄畻cluster鍚戦噺鐨凩2鑼冩暟
    {
        CUDATimer timer_compute("Kernel Execution: l2 Norm", ENABLE_CUDA_TIMING);
        // 璁＄畻鏌ヨ鍚戦噺鐨凩2鑼冩暟
        l2_norm_kernel<<<queryDim, vectorDim, n_dim * sizeof(float)>>>(
            d_query_group, d_query_norm, 
            n_query, n_dim
        );
        l2_norm_kernel<<<clusterDim, vectorDim, n_dim * sizeof(float)>>>(
            d_cluster_vector, d_cluster_vector_norm, 
            tol_vector, n_dim
        );
        cudaDeviceSynchronize();
    }

    {
        CUDATimer timer_compute("Kernel Execution: L2 Distance + Top-K", ENABLE_CUDA_TIMING);
        
        // 璁＄畻鍏变韩鍐呭瓨澶у皬
        size_t shared_mem_size = (n_query + max_cluster_vector_count) * sizeof(float);
        
        // 璋冪敤涓昏鐨凩2璺濈璁＄畻kernel
        dim3 grid(distinct_cluster_count);
        dim3 block(max_cluster_vector_count);
        
        cluster_l2_distance_kernel<<<grid, block, shared_mem_size>>>(
            d_query_group, d_query_norm, d_cluster_vector, d_cluster_vector_norm,
            d_query_cluster_group, d_cluster_query_offset, d_cluster_query_data,
            d_cluster_map, d_cluster_vector_index, d_cluster_vector_num,
            n_query, n_cluster, n_dim, n_topn, max_cluster_vector_count, distinct_cluster_count, tol_vector,
            d_query_mutex, d_topn_index, d_topn_dist
        );
        
        cudaDeviceSynchronize();
    }

    // 澶嶅埗缁撴灉鍥炰富鏈哄唴瀛?
    cudaMemcpy(h_query_topn_index, d_topn_index, size_topn_index, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_query_topn_dist, d_topn_dist, size_topn_dist, cudaMemcpyDeviceToHost);
    
    // 閲婃斁璁惧鍐呭瓨
    cudaFree(d_query_group);
    cudaFree(d_query_cluster_group);
    cudaFree(d_cluster_query_offset);
    cudaFree(d_cluster_query_data);
    cudaFree(d_cluster_map);
    cudaFree(d_cluster_vector_index);
    cudaFree(d_cluster_vector_num);
    cudaFree(d_cluster_vector);
    cudaFree(d_query_norm);
    cudaFree(d_cluster_vector_norm);
    cudaFree(d_query_mutex);
    cudaFree(d_topn_index);
    cudaFree(d_topn_dist);
}

void fine_screen_top_n_blocks(
    float* h_query_group,
    int n_query,
    int n_dim,
    int n_topn,
    float** h_block_vectors,
    int* h_block_vector_counts,
    int block_count,
    int* h_block_query_offset,
    int* h_block_query_data,
    int* h_query_topn_index,
    float* h_query_topn_dist
) {
    if (n_query <= 0 || n_dim <= 0 || n_topn <= 0 || block_count <= 0) {
        return;
    }

    float* d_query_group = nullptr;
    size_t query_bytes = static_cast<size_t>(n_query) * n_dim * sizeof(float);
    cudaMalloc(&d_query_group, query_bytes);
    cudaMemcpy(d_query_group, h_query_group, query_bytes, cudaMemcpyHostToDevice);

    std::vector<float*> d_block_buffers(block_count, nullptr);
    for (int block_id = 0; block_id < block_count; ++block_id) {
        int vec_count = h_block_vector_counts[block_id];
        if (vec_count <= 0 || !h_block_vectors[block_id]) {
            continue;
        }
        size_t bytes = static_cast<size_t>(vec_count) * n_dim * sizeof(float);
        cudaMalloc(&d_block_buffers[block_id], bytes);
        cudaMemcpy(d_block_buffers[block_id],
                   h_block_vectors[block_id],
                   bytes,
                   cudaMemcpyHostToDevice);
    }

    float** d_block_ptrs = nullptr;
    cudaMalloc(&d_block_ptrs, block_count * sizeof(float*));
    cudaMemcpy(d_block_ptrs, d_block_buffers.data(),
               block_count * sizeof(float*), cudaMemcpyHostToDevice);

    int* d_block_vector_counts = nullptr;
    cudaMalloc(&d_block_vector_counts, block_count * sizeof(int));
    cudaMemcpy(d_block_vector_counts, h_block_vector_counts,
               block_count * sizeof(int), cudaMemcpyHostToDevice);

    int* d_block_query_offset = nullptr;
    cudaMalloc(&d_block_query_offset, (block_count + 1) * sizeof(int));
    cudaMemcpy(d_block_query_offset, h_block_query_offset,
               (block_count + 1) * sizeof(int), cudaMemcpyHostToDevice);

    size_t block_query_entries = static_cast<size_t>(h_block_query_offset[block_count]);
    int* d_block_query_data = nullptr;
    if (block_query_entries > 0) {
        cudaMalloc(&d_block_query_data, block_query_entries * sizeof(int));
        cudaMemcpy(d_block_query_data, h_block_query_data,
                   block_query_entries * sizeof(int), cudaMemcpyHostToDevice);
    }

    int* d_topn_index = nullptr;
    float* d_topn_dist = nullptr;
    cudaMalloc(&d_topn_index, static_cast<size_t>(n_query) * n_topn * sizeof(int));
    cudaMalloc(&d_topn_dist, static_cast<size_t>(n_query) * n_topn * sizeof(float));
    cudaMemset(d_topn_index, 0xff, static_cast<size_t>(n_query) * n_topn * sizeof(int));
    cudaMemset(d_topn_dist, 0, static_cast<size_t>(n_query) * n_topn * sizeof(float));

    // TODO: 绮剧瓫 kernel 寰呭疄鐜?
    cudaMemcpy(h_query_topn_index, d_topn_index,
               static_cast<size_t>(n_query) * n_topn * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_query_topn_dist, d_topn_dist,
               static_cast<size_t>(n_query) * n_topn * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_topn_dist);
    cudaFree(d_topn_index);
    if (d_block_query_data) cudaFree(d_block_query_data);
    cudaFree(d_block_query_offset);
    cudaFree(d_block_vector_counts);
    cudaFree(d_block_ptrs);
    for (float*& ptr : d_block_buffers) {
        if (ptr) cudaFree(ptr);
    }
    cudaFree(d_query_group);
}








