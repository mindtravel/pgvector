#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include "../cuda/utils.cuh"
#include "../cuda/ivf_search/ivf_search.cuh"
#include "../cuda/pch.h"
#include "../cuda/dataset/dataset.cuh"

namespace py = pybind11;

// Python 包装类，直接使用 C++ ClusterDataset
class PyClusterDataset {
public:
    PyClusterDataset() : dataset_() {}
    
    ~PyClusterDataset() {
        if (dataset_.is_valid()) {
            dataset_.release();
        }
    }
    
    void init_with_kmeans(
        py::array_t<float> data,
        int n_clusters,
        int kmeans_iters = 20,
        bool use_minibatch = false,
        int distance_mode = 1,  // 1=COSINE, 0=L2
        unsigned int seed = 1234,
        int batch_size = 1 << 20,
        int device_id = 0
    ) {
        // 释放之前的dataset（如果存在）
        if (dataset_.is_valid()) {
            dataset_.release();
        }
        
        auto buf = data.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Data must be 2D array");
        }
        
        // Python接口要求必须传入外部数据
        if (!buf.ptr) {
            throw std::runtime_error("Data array pointer is NULL. External data is required.");
        }
        
        int n_total_vectors = buf.shape[0];
        int vector_dim = buf.shape[1];
        
        if (n_total_vectors <= 0 || vector_dim <= 0) {
            throw std::runtime_error("Data array must have positive dimensions. External data is required.");
        }
        
        float* h_data = static_cast<float*>(buf.ptr);
        float objective = 0.0f;
        
        // 转换距离类型
        DistanceType dist_type = (distance_mode == 0) ? L2_DISTANCE : COSINE_DISTANCE;
        
        // 直接调用 C++ 方法
        try {
            dataset_.init_with_kmeans(
            n_total_vectors,
            vector_dim,
            n_clusters,
            &objective,
                h_data,
            kmeans_iters,
                use_minibatch,
                dist_type,
            seed,
            batch_size,
            device_id
        );
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to initialize ClusterDataset: ") + e.what());
        }
    }
    
    std::tuple<py::array_t<float>, py::array_t<int>, py::array_t<float>, 
               py::array_t<long long>, py::array_t<int>, int> get_data() {
        if (!dataset_.is_valid()) {
            throw std::runtime_error("ClusterDataset not initialized");
        }
        
        // 直接访问 C++ 对象的成员
        int n_total_vectors = dataset_.n_total_vectors;
        int vector_dim = dataset_.vector_dim;
        int n_clusters = dataset_.get_n_clusters();
        
        // 创建 numpy 数组（不复制数据，只是包装）
        py::array_t<float> reordered_data_array(
            {n_total_vectors, vector_dim},
            {vector_dim * sizeof(float), sizeof(float)},
            dataset_.reordered_data
        );
        
        py::array_t<int> reordered_indices_array(
            {n_total_vectors},
            {sizeof(int)},
            dataset_.reordered_indices
        );
        
        py::array_t<float> centroids_array(
            {n_clusters, vector_dim},
            {vector_dim * sizeof(float), sizeof(float)},
            dataset_.centroids
        );
        
        py::array_t<long long> cluster_offsets_array(
            {n_clusters},
            {sizeof(long long)},
            dataset_.cluster_info.offsets
        );
        
        py::array_t<int> cluster_counts_array(
            {n_clusters},
            {sizeof(int)},
            dataset_.cluster_info.counts
        );
        
        return std::make_tuple(
            reordered_data_array,
            reordered_indices_array,
            centroids_array,
            cluster_offsets_array,
            cluster_counts_array,
            n_clusters
        );
    }
    
    // 便捷方法：获取重排后的数据
    py::array_t<float> get_reordered_data() {
        if (!dataset_.is_valid()) {
            throw std::runtime_error("ClusterDataset not initialized");
        }
        return py::array_t<float>(
            {dataset_.n_total_vectors, dataset_.vector_dim},
            {dataset_.vector_dim * sizeof(float), sizeof(float)},
            dataset_.reordered_data
        );
    }
    
    // 便捷方法：获取重排后的索引
    py::array_t<int> get_reordered_indices() {
        if (!dataset_.is_valid()) {
            throw std::runtime_error("ClusterDataset not initialized");
        }
        return py::array_t<int>(
            {dataset_.n_total_vectors},
            {sizeof(int)},
            dataset_.reordered_indices
        );
    }
    
    // 便捷方法：获取聚类中心
    py::array_t<float> get_centroids() {
        if (!dataset_.is_valid()) {
            throw std::runtime_error("ClusterDataset not initialized");
        }
        int n_clusters = dataset_.get_n_clusters();
        return py::array_t<float>(
            {n_clusters, dataset_.vector_dim},
            {dataset_.vector_dim * sizeof(float), sizeof(float)},
            dataset_.centroids
        );
    }
    
    // 便捷方法：获取 cluster 信息
    std::tuple<py::array_t<long long>, py::array_t<int>> get_cluster_info() {
        if (!dataset_.is_valid()) {
            throw std::runtime_error("ClusterDataset not initialized");
        }
        int n_clusters = dataset_.get_n_clusters();
        return std::make_tuple(
            py::array_t<long long>(
                {n_clusters},
                {sizeof(long long)},
                dataset_.cluster_info.offsets
            ),
            py::array_t<int>(
                {n_clusters},
                {sizeof(int)},
                dataset_.cluster_info.counts
            )
        );
    }
    
    // 获取基本信息
    int get_n_clusters() const {
        return dataset_.get_n_clusters();
    }
    
    int get_n_total_vectors() const {
        return dataset_.n_total_vectors;
    }
    
    int get_vector_dim() const {
        return dataset_.vector_dim;
    }
    
    bool is_valid() const {
        return dataset_.is_valid();
    }
    
    // 获取底层 C++ 对象的指针（用于搜索等需要原始指针的场景）
    ClusterDataset* get_cpp_ptr() {
        return &dataset_;
    }
    
private:
    ClusterDataset dataset_;
};

class IVFSearcher {
public:
    IVFSearcher() {}
    
    std::tuple<py::array_t<int>, py::array_t<float>> search(
        py::array_t<float> queries,
        py::array_t<int> cluster_sizes,
        py::array_t<float> cluster_vectors,
        py::array_t<float> cluster_centers,
        int n_probes,
        int k,
        int distance_mode = 1,  // 1=COSINE, 0=L2
        py::array_t<int> reordered_indices = py::array_t<int>()  // 可选：重排后的索引映射数组
    ) {
        auto queries_buf = queries.request();
        auto cluster_sizes_buf = cluster_sizes.request();
        auto cluster_vectors_buf = cluster_vectors.request();
        auto cluster_centers_buf = cluster_centers.request();
        
        if (queries_buf.ndim != 2) {
            throw std::runtime_error("Queries must be 2D array");
        }
        
        int n_query = queries_buf.shape[0];
        int n_dim = queries_buf.shape[1];
        int n_total_cluster = cluster_sizes_buf.shape[0];
        int n_total_vectors = cluster_vectors_buf.shape[0] / n_dim;
        
        // 分配 GPU 内存
        float* d_query_batch = nullptr;
        int* d_cluster_size = nullptr;
        float* d_cluster_vectors = nullptr;
        float* d_cluster_centers = nullptr;
        int* d_initial_indices = nullptr;
        float* d_topk_dist = nullptr;
        int* d_topk_index = nullptr;
        
        cudaError_t err;
        
        err = cudaMalloc(&d_query_batch, n_query * n_dim * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate GPU memory for queries");
        
        err = cudaMalloc(&d_cluster_size, n_total_cluster * sizeof(int));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate GPU memory for cluster_size");
        
        err = cudaMalloc(&d_cluster_vectors, n_total_vectors * n_dim * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate GPU memory for cluster_vectors");
        
        err = cudaMalloc(&d_cluster_centers, n_total_cluster * n_dim * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate GPU memory for cluster_centers");
        
        err = cudaMalloc(&d_topk_dist, n_query * k * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate GPU memory for topk_dist");
        
        err = cudaMalloc(&d_topk_index, n_query * k * sizeof(int));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate GPU memory for topk_index");
        
        // d_initial_indices 设为 nullptr，由 ivf_search_lookup 内部自动生成
        d_initial_indices = nullptr;
        
        // 复制数据到 GPU
        cudaMemcpy(d_query_batch, queries.request().ptr, n_query * n_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster_size, cluster_sizes.request().ptr, n_total_cluster * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster_vectors, cluster_vectors.request().ptr, n_total_vectors * n_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster_centers, cluster_centers.request().ptr, n_total_cluster * n_dim * sizeof(float), cudaMemcpyHostToDevice);
        
        // 处理 reordered_indices（如果提供）
        int* d_reordered_indices = nullptr;
        if (reordered_indices.size() > 0) {
        cudaMalloc(&d_reordered_indices, n_total_vectors * sizeof(int));
        cudaMemcpy(d_reordered_indices, reordered_indices.request().ptr, n_total_vectors * sizeof(int), cudaMemcpyHostToDevice);
        }
        
        DistanceType dist_type = (distance_mode == 0) ? L2_DISTANCE : COSINE_DISTANCE;
        
        // 直接调用带 lookup 的搜索函数
        try {
            ivf_search_lookup(
            d_query_batch,
            d_cluster_size,
            d_cluster_vectors,
            d_cluster_centers,
            d_initial_indices,
            d_topk_dist,
            d_topk_index,
            n_query,
            n_dim,
            n_total_cluster,
            n_total_vectors,
            n_probes,
            k,
            dist_type,
            nullptr,  // h_coarse_index (不需要)
            nullptr,  // h_coarse_dist (不需要)
            d_reordered_indices  // 回表映射数组（可选）
        );
        } catch (const std::exception& e) {
            // 清理内存
            cudaFree(d_query_batch);
            cudaFree(d_cluster_size);
            cudaFree(d_cluster_vectors);
            cudaFree(d_cluster_centers);
            cudaFree(d_initial_indices);
            cudaFree(d_topk_dist);
            cudaFree(d_topk_index);
            if (d_reordered_indices) cudaFree(d_reordered_indices);
            throw std::runtime_error(std::string("ivf_search_lookup failed: ") + e.what());
        }
        
        // 分配结果数组
        py::array_t<int> indices({n_query, k});
        py::array_t<float> distances({n_query, k});
        
        // 复制结果回 CPU
        cudaMemcpy(indices.mutable_data(), d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(distances.mutable_data(), d_topk_dist, n_query * k * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 清理 GPU 内存
        cudaFree(d_query_batch);
        cudaFree(d_cluster_size);
        cudaFree(d_cluster_vectors);
        cudaFree(d_cluster_centers);
        // d_initial_indices 由 ivf_search_lookup 内部管理，不需要释放
        cudaFree(d_topk_dist);
        cudaFree(d_topk_index);
        if (d_reordered_indices) cudaFree(d_reordered_indices);
        
        return std::make_tuple(indices, distances);
    }
};

PYBIND11_MODULE(PyIVFTensor, m) {
    m.doc() = "IVF-Tensor: GPU-accelerated IVF-Flat search";
    
    py::class_<PyClusterDataset>(m, "ClusterDataset")
        .def(py::init<>())
        .def("init_with_kmeans", &PyClusterDataset::init_with_kmeans,
             py::arg("data"),
             py::arg("n_clusters"),
             py::arg("kmeans_iters") = 20,
             py::arg("use_minibatch") = false,
             py::arg("distance_mode") = 1,
             py::arg("seed") = 1234,
             py::arg("batch_size") = 1 << 20,
             py::arg("device_id") = 0,
             "Initialize ClusterDataset with K-means clustering")
        .def("get_data", &PyClusterDataset::get_data,
             "Get all data: reordered_data, reordered_indices, centroids, cluster_offsets, cluster_counts, n_clusters")
        .def("get_reordered_data", &PyClusterDataset::get_reordered_data,
             "Get reordered data array")
        .def("get_reordered_indices", &PyClusterDataset::get_reordered_indices,
             "Get reordered indices array")
        .def("get_centroids", &PyClusterDataset::get_centroids,
             "Get cluster centroids")
        .def("get_cluster_info", &PyClusterDataset::get_cluster_info,
             "Get cluster offsets and counts")
        .def("get_n_clusters", &PyClusterDataset::get_n_clusters,
             "Get number of clusters")
        .def("get_n_total_vectors", &PyClusterDataset::get_n_total_vectors,
             "Get total number of vectors")
        .def("get_vector_dim", &PyClusterDataset::get_vector_dim,
             "Get vector dimension")
        .def("is_valid", &PyClusterDataset::is_valid,
             "Check if ClusterDataset is initialized")
        .def("get_cpp_ptr", &PyClusterDataset::get_cpp_ptr,
             py::return_value_policy::reference_internal,
             "Get underlying C++ ClusterDataset pointer (for advanced usage)");
    
    py::class_<IVFSearcher>(m, "IVFSearcher")
        .def(py::init<>())
        .def("search", &IVFSearcher::search,
             py::arg("queries"),
             py::arg("cluster_sizes"),
             py::arg("cluster_vectors"),
             py::arg("cluster_centers"),
             py::arg("n_probes"),
             py::arg("k"),
             py::arg("distance_mode") = 1,
             py::arg("reordered_indices") = py::array_t<int>(),
             "Perform IVF search");
    
    // 导出常量
    m.attr("DISTANCE_L2") = 0;
    m.attr("DISTANCE_COSINE") = 1;
}
