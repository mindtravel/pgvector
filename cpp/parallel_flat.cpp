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
#include <numeric>

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

// 创建表并加载数据
void create_table_and_load_data(PGconn* conn, 
                              const std::vector<std::vector<float>>& data,
                              int reduced_dim) {
    execute_sql(conn, "CREATE EXTENSION IF NOT EXISTS vector;");
    execute_sql(conn, "DROP TABLE IF EXISTS sift_data;");
    execute_sql(conn, 
        "CREATE TABLE sift_data (id serial, vector vector(" + 
        std::to_string(reduced_dim) + "));");

    if (data.empty()) {
        std::cout << "Warning: No data to insert!" << std::endl;
        return;
    }

    // 使用COPY命令高效插入
    PGresult* res = PQexec(conn, "COPY sift_data (vector) FROM STDIN WITH (FORMAT CSV)");
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
        row << "\"["; 
        for (size_t j = 0; j < data[i].size(); ++j) {
            row << (j > 0 ? "," : "") << data[i][j];
        }
        row << "]\"\n";
        
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

// 创建索引
void create_ivfflat_index(PGconn* conn, int n_lists) {
    execute_sql(conn, "SET maintenance_work_mem = '1GB';");
    execute_sql(conn,
        "CREATE INDEX ON sift_data USING ivfflat (vector vector_l2_ops) "
        "WITH (lists = " + std::to_string(n_lists) + ");");
}

// KNN搜索测试 - 增加详细调试输出
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
                
                std::string sql = "SELECT id FROM sift_data ORDER BY vector <-> " + 
                                  vec_str + " LIMIT " + std::to_string(k) + ";";
                
                auto query_start = std::chrono::high_resolution_clock::now();
                PGresult* res = PQexec(thread_conn, sql.c_str());
                auto query_end = std::chrono::high_resolution_clock::now();
                
                if (PQresultStatus(res) != PGRES_TUPLES_OK) {
                    std::cerr << "Thread " << t << ", Query " << i << " failed: " 
                              << PQerrorMessage(thread_conn) << std::endl;
                    PQclear(res);
                    continue;
                }
                
                // 收集查询结果ID
                std::vector<int> results;
                int rows = PQntuples(res);
                for (int r = 0; r < rows; ++r) {
                    // 关键修改：不再对数据库ID减1
                    int db_id = atoi(PQgetvalue(res, r, 0));
                    results.push_back(db_id);
                }
                
                // 获取当前查询的groundtruth
                std::vector<int> gt;
                for (float val : groundtruth[i]) {
                    gt.push_back(static_cast<int>(val)+1);
                }
                
                // 输出样本数据用于调试
                if (i == start_idx) {
                    std::cout << "Thread " << t << ", Query " << i << " sample:" << std::endl;
                    std::cout << "  Query results (first 5):";
                    for (int j = 0; j < std::min(5, static_cast<int>(results.size())); ++j) {
                        std::cout << " " << results[j];
                    }
                    std::cout << std::endl;
                    
                    std::cout << "  Groundtruth (first 5):";
                    for (int j = 0; j < std::min(5, static_cast<int>(gt.size())); ++j) {
                        std::cout << " " << gt[j];
                    }
                    std::cout << std::endl;
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
                }
                
                // 输出部分查询的召回率
                if (i % 1000 == 0) {
                    std::cout << "Thread " << t << ", Query " << i 
                              << " recall: " << query_recall 
                              << ", intersection size: " << intersection.size() << std::endl;
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
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

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
    
    // 解析命令行参数
    struct option long_options[] = {
        {"max_parallel_workers_per_gather", required_argument, 0, 'p'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "p:", long_options, NULL)) != -1) {
        if (opt == 'p') max_parallel_workers = atoi(optarg);
    }

    try {
        std::cout << "Loading datasets..." << std::endl;

        // 修改：
        // auto database = load_data("../../sift/sift_base.fvecs", 128);
        // auto queries = load_data("../../sift/sift_query.fvecs", 128);
        // // 明确指定groundtruth是ivecs格式
        // auto groundtruth = load_data("../../sift/sift_groundtruth.ivecs", 100, true);
        auto database = load_data("../sift/sift_base.fvecs", 128);
        auto queries = load_data("../sift/sift_query.fvecs", 128);
        // 明确指定groundtruth是ivecs格式
        auto groundtruth = load_data("../sift/sift_groundtruth.ivecs", 100, true);
        
        std::cout << "Datasets loaded successfully" << std::endl;
        
        // 标准化数据集
        // database = normalize_dataset(database);
        // queries = normalize_dataset(queries);   
        
        // 输出样本数据用于验证
        if (!database.empty()) {
            std::cout << "Sample database vector[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(database[0].size())); ++i) {
                std::cout << database[0][i] << " ";
            }
            std::cout << std::endl;
        }
        
        if (!queries.empty()) {
            std::cout << "Sample query vector[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(queries[0].size())); ++i) {
                std::cout << queries[0][i] << " ";
            }
            std::cout << std::endl;
        }
        
        if (!groundtruth.empty()) {
            std::cout << "Sample groundtruth[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(groundtruth[0].size())); ++i) {
                std::cout << groundtruth[0][i] << " ";
            }
            std::cout << std::endl;
        }

        PGconn* conn = connect_to_db();
        if (!conn) return 1;

        std::cout << "Creating table..." << std::endl;
        create_table_and_load_data(conn, database, 128);

        std::cout << "Creating index..." << std::endl;
        create_ivfflat_index(conn, 100);

        std::cout << "Setting parallel workers..." << std::endl;
        execute_sql(conn, 
            "SET max_parallel_workers_per_gather = " + 
            std::to_string(max_parallel_workers) + ";");

        std::vector<BenchmarkResult> results;
        int thread_count = 1;
        std::cout << "Using " << thread_count << " threads for parallel queries" << std::endl;

        for (int nprob : {1, 5, 10, 20}) {
            std::cout << "\nTesting with nprob = " << nprob << std::endl;
            auto [recall, throughput] = test_knn_search(
                conn, queries, groundtruth, 10, nprob, thread_count
            );
            results.push_back(BenchmarkResult(nprob, recall, throughput));
            std::cout << "Recall: " << recall 
                      << ", Throughput: " << throughput << " qps\n";
        }

        PQfinish(conn);
        std::cout << "Database connection closed" << std::endl;
        
        // 输出JSON结果
        std::ofstream json_file("flat_results.json");
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
        std::cout << "Results saved to flat_results.json" << std::endl;
    }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}