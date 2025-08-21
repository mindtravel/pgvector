#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <libpq-fe.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <getopt.h>
#include <thread>
#include <random>
#include <iterator>
#include <set>
#include <iomanip>
#include <stdexcept>
#include <cstring>

double pre_process_time;

// 数据库配置
const std::string DB_CONN_STR = 
    "host=localhost port=5432 dbname=mydatabase user=postgres password=1";

// 数据结构定义
struct QueryResult {
    std::vector<int> ids;
    double latency;
};

struct BenchmarkResult {
    int nprob;
    double recall;
    double throughput;
    BenchmarkResult(int n, double r, double t) 
        : nprob(n), recall(r), throughput(t) {}
};

// 计算向量范数
float vector_norm(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// 向量标准化
std::vector<float> normalize_vector(const std::vector<float>& vec) {
    float norm = vector_norm(vec);
    if (norm < 1e-8) return vec; // 避免除零
    
    std::vector<float> normalized(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        normalized[i] = vec[i] / norm;
    }
    return normalized;
}

// 对整个数据集进行标准化
std::vector<std::vector<float>> normalize_dataset(const std::vector<std::vector<float>>& dataset) {
    std::vector<std::vector<float>> normalized(dataset.size());
    for (size_t i = 0; i < dataset.size(); ++i) {
        normalized[i] = normalize_vector(dataset[i]);
    }
    return normalized;
}

// 计算两个向量之间的欧氏距离平方
float euclidean_distance_squared(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// 数据加载函数 - 增加对ivecs格式的支持
std::vector<std::vector<float>> load_data(const std::string& path, int dim, bool is_ivecs = false) {
    std::ifstream file(path, std::ios::binary);
    std::vector<std::vector<float>> data;
    
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
        return data;
    }

    // 计算向量数量
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t element_size = is_ivecs ? sizeof(int32_t) : sizeof(float);
    size_t vector_size = (dim * element_size) + sizeof(int32_t);
    size_t num_vectors = file_size / vector_size;
    
    std::cout << "Loading " << (is_ivecs ? "ivecs" : "fvecs") 
              << " file: " << path << std::endl;
    std::cout << "File size: " << file_size << " bytes" << std::endl;
    std::cout << "Vector size: " << vector_size << " bytes" << std::endl;
    std::cout << "Number of vectors expected: " << num_vectors << std::endl;
    
    for (size_t i = 0; i < num_vectors; ++i) {
        int32_t actual_dim;
        file.read(reinterpret_cast<char*>(&actual_dim), sizeof(int32_t));
        
        if (actual_dim != dim) {
            std::cerr << "Dimension mismatch at vector " << i 
                      << ": expected " << dim << ", got " << actual_dim << std::endl;
            break;
        }
        
        std::vector<float> vec(dim);
        if (is_ivecs) {
            // 读取整型ID并转换为float
            std::vector<int32_t> int_vec(dim);
            file.read(reinterpret_cast<char*>(int_vec.data()), dim * sizeof(int32_t));
            for (int j = 0; j < dim; ++j) {
                vec[j] = static_cast<float>(int_vec[j]);
            }
        } else {
            // 正常读取float
            file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        }
        
        if (file.gcount() != static_cast<std::streamsize>(dim * element_size)) {
            std::cerr << "Incomplete read at vector " << i << std::endl;
            break;
        }
        
        data.push_back(vec);
    }
    
    std::cout << "Successfully loaded " << data.size() << " vectors" << std::endl;
    return data;
}

// 数据库连接
PGconn* connect_to_db() {
    PGconn* conn = PQconnectdb(DB_CONN_STR.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << std::endl;
        PQfinish(conn);
        return nullptr;
    }
    return conn;
}

// 执行SQL命令
void execute_sql(PGconn* conn, const std::string& sql) {
    PGresult* res = PQexec(conn, sql.c_str());
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "SQL Error: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        exit(1);
    }
    PQclear(res);
}

// 产品量化类
class ProductQuantizer {
private:
    int original_dim;     // 原始维度
    int m_subvectors;     // 子向量数量
    int k_centroids;      // 每个子空间的质心数量
    int code_size;        // 编码大小（字节）
    
    std::vector<std::vector<std::vector<float>>> centroids;  // 质心表 [m][k][d/m]
    
public:
    ProductQuantizer(int dim, int m, int k) 
        : original_dim(dim), m_subvectors(m), k_centroids(k) {
        code_size = m_subvectors;  // 每个子向量用一个字节表示
        
        // 初始化质心表
        int subvector_dim = original_dim / m_subvectors;
        centroids.resize(m_subvectors);
        for (int m = 0; m < m_subvectors; ++m) {
            centroids[m].resize(k_centroids, std::vector<float>(subvector_dim, 0.0f));
        }
    }
    
    // 训练质心
    void train(const std::vector<std::vector<float>>& data, int iterations = 10) {
        int subvector_dim = original_dim / m_subvectors;
        int n_vectors = data.size();
        
        // 分割数据并训练每个子空间的质心
        for (int m = 0; m < m_subvectors; ++m) {
            std::vector<std::vector<float>> sub_vectors(n_vectors);
            
            // 提取子向量
            for (int i = 0; i < n_vectors; ++i) {
                sub_vectors[i].resize(subvector_dim);
                for (int d = 0; d < subvector_dim; ++d) {
                    sub_vectors[i][d] = data[i][m * subvector_dim + d];
                }
            }
            
            // K-means 聚类（简化版）
            std::vector<std::vector<float>>& current_centroids = centroids[m];
            
            // 初始化质心为随机向量
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, n_vectors - 1);
            
            for (int k = 0; k < k_centroids; ++k) {
                int random_idx = dis(gen);
                current_centroids[k] = sub_vectors[random_idx];
            }
            
            // 简化的K-means迭代
            for (int iter = 0; iter < iterations; ++iter) {
                // 分配向量到最近的质心
                std::vector<std::vector<int>> clusters(k_centroids);
                
                for (int i = 0; i < n_vectors; ++i) {
                    float min_dist = std::numeric_limits<float>::max();
                    int closest_centroid = 0;
                    
                    for (int k = 0; k < k_centroids; ++k) {
                        float dist = euclidean_distance_squared(sub_vectors[i], current_centroids[k]);
                        if (dist < min_dist) {
                            min_dist = dist;
                            closest_centroid = k;
                        }
                    }
                    
                    clusters[closest_centroid].push_back(i);
                }
                
                // 更新质心
                for (int k = 0; k < k_centroids; ++k) {
                    if (clusters[k].empty()) continue;
                    
                    std::vector<float> new_centroid(subvector_dim, 0.0f);
                    for (int idx : clusters[k]) {
                        for (int d = 0; d < subvector_dim; ++d) {
                            new_centroid[d] += sub_vectors[idx][d];
                        }
                    }
                    
                    for (int d = 0; d < subvector_dim; ++d) {
                        new_centroid[d] /= clusters[k].size();
                    }
                    
                    current_centroids[k] = new_centroid;
                }
            }
        }
    }
    
    // 量化向量
    std::vector<uint8_t> encode(const std::vector<float>& vector) const {
        std::vector<uint8_t> codes(m_subvectors);
        int subvector_dim = original_dim / m_subvectors;
        
        for (int m = 0; m < m_subvectors; ++m) {
            // 提取子向量
            std::vector<float> sub_vector(subvector_dim);
            for (int d = 0; d < subvector_dim; ++d) {
                sub_vector[d] = vector[m * subvector_dim + d];
            }
            
            // 找到最近的质心
            float min_dist = std::numeric_limits<float>::max();
            int closest_centroid = 0;
            
            for (int k = 0; k < k_centroids; ++k) {
                float dist = euclidean_distance_squared(sub_vector, centroids[m][k]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = k;
                }
            }
            
            codes[m] = static_cast<uint8_t>(closest_centroid);
        }
        
        return codes;
    }
    
    // 对整个数据集进行编码
    std::vector<std::vector<uint8_t>> encode_all(const std::vector<std::vector<float>>& dataset) const {
        std::vector<std::vector<uint8_t>> encoded(dataset.size());
        for (size_t i = 0; i < dataset.size(); ++i) {
            encoded[i] = encode(dataset[i]);
        }
        return encoded;
    }
    
    // 获取编码大小（字节）
    int get_code_size() const {
        return code_size;
    }
    
    // 将编码转换为字符串格式（用于存储到数据库）
    static std::string codes_to_string(const std::vector<uint8_t>& codes) {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < codes.size(); ++i) {
            ss << static_cast<int>(codes[i]);
            if (i < codes.size() - 1) ss << ",";
        }
        ss << "]";
        return ss.str();
    }
    
    // 将字符串格式的编码转换回向量
    static std::vector<uint8_t> string_to_codes(const std::string& str) {
        std::vector<uint8_t> codes;
        std::stringstream ss(str.substr(1, str.length() - 2));
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            codes.push_back(static_cast<uint8_t>(std::stoi(token)));
        }
        
        return codes;
    }
};

void create_table_and_load_data(PGconn* conn, 
                              const std::vector<std::vector<float>>& data,
                              const std::vector<std::vector<uint8_t>>& pq_codes,
                              int dim,
                              int m_subvectors) {
    execute_sql(conn, "CREATE EXTENSION IF NOT EXISTS vector;");
    execute_sql(conn, "DROP TABLE IF EXISTS sift_data;");
    execute_sql(conn, 
        "CREATE TABLE sift_data (id serial, vector vector(" + 
        std::to_string(dim) + "), pq_code bytea);");

    if (data.empty()) {
        std::cout << "Warning: No data to insert!" << std::endl;
        return;
    }

    // 使用COPY命令高效插入
    PGresult* res = PQexec(conn, "COPY sift_data (vector, pq_code) FROM STDIN WITH (FORMAT CSV)");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return;
    }
    PQclear(res);

    const size_t batch_size = 5000;
    size_t inserted = 0;
    
    for (size_t i = 0; i < data.size(); ++i) {
        std::stringstream row;
        
        // 构建向量字符串
        row << "\"["; 
        for (size_t j = 0; j < data[i].size(); ++j) {
            row << (j > 0 ? "," : "") << std::fixed << std::setprecision(6) << data[i][j];
        }
        row << "]\",";
        
        // 正确调用PQescapeByteaConn，添加第四个参数
        size_t escaped_len;
        unsigned char* escaped_data = PQescapeByteaConn(conn,
                                                         reinterpret_cast<const unsigned char*>(pq_codes[i].data()),
                                                         pq_codes[i].size(),
                                                         &escaped_len);
        
        if (escaped_data) {
            // 将转义后的数据添加到流中
            row.write(reinterpret_cast<const char*>(escaped_data), escaped_len);
            // 释放转义后的数据
            PQfreemem(escaped_data);
        } else {
            std::cerr << "Error escaping bytea data for record " << i << std::endl;
            // 使用空字符串作为默认值
            row << "NULL";
        }
        
        row << "\n";
        
        if (PQputCopyData(conn, row.str().c_str(), row.str().size()) != 1) {
            std::cerr << "PQputCopyData failed: " << PQerrorMessage(conn) << std::endl;
            break;
        }
        
        if ((i + 1) % batch_size == 0) {
            std::cout << "Inserted " << (i + 1) << " records..." << std::endl;
        }
    }

    if (PQputCopyEnd(conn, NULL) != 1) {
        std::cerr << "PQputCopyEnd failed: " << PQerrorMessage(conn) << std::endl;
    } else {
        std::cout << "Successfully inserted " << data.size() << " records." << std::endl;
    }
}

// 创建IVFPQ索引
void create_ivfpq_index(PGconn* conn, int n_lists, int m_subvectors) {
    execute_sql(conn, "SET maintenance_work_mem = '1GB';");
    execute_sql(conn,
        "CREATE INDEX ON sift_data USING ivfflat (vector vector_l2_ops) "
        "WITH (lists = " + std::to_string(n_lists) + ");");
    
    // 为PQ编码创建辅助索引（可选）
    execute_sql(conn,
        "CREATE INDEX ON sift_data USING gin (pq_code) "
        "WITH (fastupdate = off);");
}

// 验证向量与groundtruth的距离关系
void validate_groundtruth(
    const std::vector<std::vector<float>>& database,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<float>>& groundtruth,
    int sample_size = 10) {
    
    std::cout << "\nValidating groundtruth..." << std::endl;
    
    for (int i = 0; i < sample_size; ++i) {
        const auto& query = queries[i];
        std::cout << "Query " << i << ":\n";
        
        // 计算与groundtruth中前5个ID的距离
        for (int j = 0; j < std::min(5, static_cast<int>(groundtruth[i].size())); ++j) {
            int gt_id = static_cast<int>(groundtruth[i][j]);
            if (gt_id >= 0 && gt_id < database.size()) {
                float dist = std::sqrt(euclidean_distance_squared(query, database[gt_id]));
                std::cout << "  Groundtruth #" << j << " (ID=" << gt_id << "): distance=" << dist << "\n";
            }
        }
        
        // 计算与数据库中随机选择的5个向量的距离
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, database.size() - 1);
        
        std::cout << "  Random vectors:\n";
        for (int j = 0; j < 5; ++j) {
            int random_id = dis(gen);
            float dist = std::sqrt(euclidean_distance_squared(query, database[random_id]));
            std::cout << "    Random ID=" << random_id << ": distance=" << dist << "\n";
        }
    }
}

// KNN搜索测试
std::pair<double, double> test_knn_search(
    PGconn* conn,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<float>>& groundtruth,
    int k,
    int nprob,
    int thread_count = std::thread::hardware_concurrency()
) {
    if (thread_count <= 0) thread_count = 1;
    const size_t query_count = queries.size();
    const size_t queries_per_thread = (query_count + thread_count - 1) / thread_count;
    
    std::vector<std::thread> threads;
    std::vector<double> thread_recalls(thread_count, 0.0);
    std::vector<double> thread_times(thread_count, 0.0);
    
    // 跟踪有召回的查询数量
    std::vector<int> thread_hits(thread_count, 0);
    std::vector<int> thread_tests(thread_count, 0);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    // 创建线程执行查询
    for (int t = 0; t < thread_count; ++t) {
        threads.emplace_back([&, t]() {
            size_t start_idx = t * queries_per_thread;
            size_t end_idx = std::min(start_idx + queries_per_thread, query_count);
            
            // 每个线程创建自己的数据库连接
            PGconn* thread_conn = connect_to_db();
            if (!thread_conn) {
                std::cerr << "Thread " << t << " failed to connect to database!" << std::endl;
                return;
            }
            
            // 设置每个线程的探测参数
            execute_sql(thread_conn, "SET ivfflat.probes = " + std::to_string(nprob) + ";");
            
            double local_recall = 0.0;
            double local_time = 0.0;
            int local_hits = 0;
            int local_tests = 0;
            
            for (size_t i = start_idx; i < end_idx; ++i) {
                std::string vec_str = "'[";
                for (size_t j = 0; j < queries[i].size(); ++j) {
                    vec_str += (j > 0 ? "," : "") + std::to_string(queries[i][j]);
                }
                vec_str += "]'::vector(" + std::to_string(queries[i].size()) + ")";
                
                std::string sql = "SELECT id, (vector <-> " + vec_str + ") as distance FROM sift_data "
                                  "ORDER BY vector <-> " + vec_str + " LIMIT " + std::to_string(k) + ";";
                
                auto query_start = std::chrono::high_resolution_clock::now();
                PGresult* res = PQexec(thread_conn, sql.c_str());
                auto query_end = std::chrono::high_resolution_clock::now();
                
                if (PQresultStatus(res) != PGRES_TUPLES_OK) {
                    std::cerr << "Thread " << t << ", Query " << i << " failed: " 
                              << PQerrorMessage(thread_conn) << std::endl;
                    PQclear(res);
                    continue;
                }
                
                // 收集查询结果ID和距离
                std::vector<int> results;
                std::vector<float> distances;
                int rows = PQntuples(res);
                for (int r = 0; r < rows; ++r) {
                    int db_id = atoi(PQgetvalue(res, r, 0));
                    float distance = atof(PQgetvalue(res, r, 1));
                    results.push_back(db_id);
                    distances.push_back(distance);
                }
                
                // 获取当前查询的groundtruth，并将ID加1以匹配数据库ID
                std::vector<int> gt;
                for (float val : groundtruth[i]) {
                    // 将groundtruth中的ID加1，使其与数据库ID匹配
                    gt.push_back(static_cast<int>(val) + 1);
                }
                
                // 计算交集
                std::sort(results.begin(), results.end());
                std::sort(gt.begin(), gt.end());
                
                std::vector<int> intersection;
                std::set_intersection(
                    results.begin(), results.end(),
                    gt.begin(), gt.end(),
                    std::back_inserter(intersection)
                );
                
                double query_recall = static_cast<double>(intersection.size()) / k;
                local_recall += query_recall;
                local_time += std::chrono::duration<double>(query_end - query_start).count();
                local_tests++;
                
                if (intersection.size() > 0) {
                    local_hits++;
                    if (i % 1000 == 0) {
                        std::cout << "HIT! Thread " << t << ", Query " << i 
                                  << " recall: " << query_recall 
                                  << ", intersection size: " << intersection.size() << std::endl;
                    }
                }
                
                PQclear(res);
            }
            
            thread_recalls[t] = local_recall;
            thread_times[t] = local_time;
            thread_hits[t] = local_hits;
            thread_tests[t] = local_tests;
            
            // 输出线程汇总信息
            std::cout << "Thread " << t << " completed: " 
                      << local_tests << " queries, "
                      << local_hits << " had non-zero recall" << std::endl;
            
            // 关闭线程的数据库连接
            PQfinish(thread_conn);
        });
    }
    
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count() + pre_process_time;

    // 汇总结果
    double total_recall = std::accumulate(thread_recalls.begin(), thread_recalls.end(), 0.0);
    int total_hits = std::accumulate(thread_hits.begin(), thread_hits.end(), 0);
    int total_tests = std::accumulate(thread_tests.begin(), thread_tests.end(), 0);
    
    std::cout << "Total queries: " << total_tests << std::endl;
    std::cout << "Queries with non-zero recall: " << total_hits 
              << " (" << (double)total_hits/total_tests*100.0 << "%)" << std::endl;
    
    return {
        total_recall / query_count,
        query_count / total_time
    };
}

int main(int argc, char* argv[]) {
    int max_parallel_workers = 20;
    int n_lists = 100;       // IVF索引的分区数量
    int m_subvectors = 8;    // PQ的子向量数量
    int k_centroids = 256;   // 每个子空间的质心数量
    
    // 解析命令行参数
    struct option long_options[] = {
        {"max_parallel_workers_per_gather", required_argument, 0, 'p'},
        {"n_lists", required_argument, 0, 'l'},
        {"m_subvectors", required_argument, 0, 'm'},
        {"k_centroids", required_argument, 0, 'k'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "p:l:m:k:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'p':
                max_parallel_workers = atoi(optarg);
                break;
            case 'l':
                n_lists = atoi(optarg);
                break;
            case 'm':
                m_subvectors = atoi(optarg);
                break;
            case 'k':
                k_centroids = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] 
                          << " [-p max_parallel_workers_per_gather] [-l n_lists] "
                          << "[-m m_subvectors] [-k k_centroids]" << std::endl;
                return 1;
        }
    }

    try {
        std::cout << "Loading datasets..." << std::endl;
        auto database = load_data("../../sift/sift_base.fvecs", 128);
        auto queries = load_data("../../sift/sift_query.fvecs", 128);
        // 明确指定groundtruth是ivecs格式
        auto groundtruth = load_data("../../sift/sift_groundtruth.ivecs", 100, true);
        std::cout << "Datasets loaded successfully" << std::endl;
        
        // 输出样本数据用于验证
        if (!database.empty()) {
            std::cout << "Sample database vector[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(database[0].size())); ++i) {
                std::cout << database[0][i] << " ";
            }
            std::cout << "\nNorm: " << vector_norm(database[0]) << std::endl;
        }
        
        if (!queries.empty()) {
            std::cout << "Sample query vector[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(queries[0].size())); ++i) {
                std::cout << queries[0][i] << " ";
            }
            std::cout << "\nNorm: " << vector_norm(queries[0]) << std::endl;
        }
        
        if (!groundtruth.empty()) {
            std::cout << "Sample groundtruth[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(groundtruth[0].size())); ++i) {
                std::cout << groundtruth[0][i] << " ";
            }
            std::cout << std::endl;
        }
        
        // 验证groundtruth
        validate_groundtruth(database, queries, groundtruth);
        
        // 向量标准化
        std::cout << "\nNormalizing vectors..." << std::endl;
        database = normalize_dataset(database);
        queries = normalize_dataset(queries);
        
        // 输出标准化后的样本数据用于验证
        if (!database.empty()) {
            std::cout << "Sample normalized database vector[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(database[0].size())); ++i) {
                std::cout << database[0][i] << " ";
            }
            std::cout << "\nNorm: " << vector_norm(database[0]) << std::endl;
        }
        
        if (!queries.empty()) {
            std::cout << "Sample normalized query vector[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(queries[0].size())); ++i) {
                std::cout << queries[0][i] << " ";
            }
            std::cout << "\nNorm: " << vector_norm(queries[0]) << std::endl;
        }
        
        // 训练产品量化器
        std::cout << "\nTraining Product Quantizer..." << std::endl;
        ProductQuantizer pq(128, m_subvectors, k_centroids);
        pq.train(database);
        std::cout << "Product Quantizer trained successfully" << std::endl;
        
        // 对数据进行PQ编码
        std::cout << "Encoding dataset with PQ..." << std::endl;
        auto database_pq_codes = pq.encode_all(database);
        auto queries_pq_codes = pq.encode_all(queries);
        std::cout << "PQ encoding completed" << std::endl;
        
        // 连接数据库
        PGconn* conn = connect_to_db();
        if (!conn) return 1;

        // 创建表并加载数据
        std::cout << "\nCreating table and loading data..." << std::endl;
        create_table_and_load_data(conn, database, database_pq_codes, 128, m_subvectors);

        // 创建IVFPQ索引
        std::cout << "Creating IVFPQ index with " << n_lists << " lists..." << std::endl;
        create_ivfpq_index(conn, n_lists, m_subvectors);

        // 设置并行工作线程
        std::cout << "Setting parallel workers..." << std::endl;
        execute_sql(conn, 
            "SET max_parallel_workers_per_gather = " + 
            std::to_string(max_parallel_workers) + ";");

        // 性能测试
        std::vector<BenchmarkResult> results;
        int thread_count = 20;
        std::cout << "Using " << thread_count << " threads for parallel queries" << std::endl;

        for (int nprob : {1, 5, 10, 20}) { // 测试不同nprob值
            std::cout << "\nTesting with nprob = " << nprob << std::endl;
            auto [recall, throughput] = test_knn_search(
                conn, queries, groundtruth, 10, nprob, thread_count
            );
            results.push_back(BenchmarkResult(nprob, recall, throughput));
            std::cout << "Recall: " << recall 
                      << ", Throughput: " << throughput << " qps\n";
        }

        // 关闭数据库连接
        PQfinish(conn);
        std::cout << "Database connection closed" << std::endl;
        
        // 输出JSON结果
        std::ofstream json_file("ivfpq_results.json");
        if (json_file.is_open()) {
            json_file << "[";
            for (size_t i = 0; i < results.size(); ++i) {
                json_file << "{"
                          << "\"nprob\":" << results[i].nprob << ","
                          << "\"recall\":" << results[i].recall << ","
                          << "\"throughput\":" << results[i].throughput
                          << "}" << (i < results.size()-1 ? "," : "");
            }
            json_file << "]" << std::endl;
            json_file.close();
            std::cout << "Results saved to ivfpq_results.json" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}    