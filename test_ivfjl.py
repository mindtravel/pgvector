import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import time
import io
import json
import argparse
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler

DB_PARAMS = {
    'host': 'localhost',
    'port': '5432',
    'database': 'mydatabase',
    'user': 'postgres',
    'password': '1'
}

def load_data(file_path, dim):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
        data = data.reshape(-1, dim)
        data = data[:, 1:].astype(float)
    return data

def connect_to_db():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def create_table_and_load_data(conn, data, reduced_dim):
    cur = conn.cursor()
    try:
        # 启用vector扩展
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # 删除已存在的表
        cur.execute("DROP TABLE IF EXISTS sift_data;")
        
        # 创建新表
        cur.execute(f"CREATE TABLE sift_data (id serial, vector vector({reduced_dim}));")
        
        # 分批次插入数据
        batch_size = 1000
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            for row in batch:
                # 将向量转换为字符串格式，确保数值格式正确
                vector_str = '[' + ','.join(f"{x:.6f}" for x in row) + ']'
                cur.execute(
                    "INSERT INTO sift_data (vector) VALUES (%s::vector);",
                    (vector_str,)
                )
            conn.commit()
            print(f"Inserted batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
        
        print("Data loading completed successfully")
        
    except psycopg2.Error as e:
        print(f"Error in create_table_and_load_data: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()

def create_ivfjl_index(conn, n_lists):
    cur = conn.cursor()
    try:
        # 检查表是否存在
        cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'sift_data');")
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            raise Exception("Table sift_data does not exist. Please load data first.")
        
        # 增加维护工作内存
        cur.execute("SET maintenance_work_mem = '1GB';")
        
        # 增加并行工作进程数
        cur.execute("SET max_parallel_workers_per_gather = 20;")
        
        # 创建索引
        cur.execute(f"""
            CREATE INDEX ON sift_data 
            USING ivfflat (vector vector_l2_ops) 
            WITH (lists = {n_lists});
        """)
        conn.commit()
        print("Index created successfully")
        
    except psycopg2.Error as e:
        print(f"Error in create_ivfjl_index: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()

def test_knn_search(conn, query_vectors, groundtruth, k, nprob):
    cur = conn.cursor()
    recall_sum = 0.0
    total_time = 0.0
    try:
        sum_recall = 0
        for i, query in enumerate(query_vectors):
            vector_str = '[' + ','.join(f"{x:.6f}" for x in query) + ']'
            
            start_time = time.time()
            cur.execute(f"SET ivfflat.probes = {nprob};")
            cur.execute(
                f"SELECT id FROM sift_data ORDER BY vector <-> %s::vector LIMIT {k};",
                (vector_str,)
            )
            results = [row[0]-1 for row in cur.fetchall()]
            end_time = time.time()
            total_time += end_time - start_time
            intersection = len(set(results) & set(groundtruth[i]))
            sum_recall += intersection / k

        avg_recall = sum_recall / len(query_vectors)
        throughput = len(query_vectors) / total_time
        return avg_recall, throughput
    except psycopg2.Error as e:
        print(f"Error in test_knn_search: {e}")
        raise
    finally:
        cur.close()

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_parallel_workers_per_gather', type=int, default=20,
                      help='Maximum number of parallel workers per gather')
    args = parser.parse_args()

    try:
        # 加载数据集
        print("Loading datasets...")
        database = load_data('../sift/sift_base.fvecs',129)
        queries = load_data('../sift/sift_query.fvecs',129)
        groundtruth = load_data("../sift/sift_groundtruth.ivecs",101)
        print("Datasets loaded successfully")
        
        # 计算JL降维后的维度
        n_samples = len(database)
        eps = 0.1  # 误差容忍度
        reduced_dim = min(
            johnson_lindenstrauss_min_dim(n_samples, eps=eps),
            database.shape[1] // 2
        )
        print(f"Original dimension: {database.shape[1]}")
        print(f"Reduced dimension: {reduced_dim}")
        
        # 对数据进行降维
        transformer = GaussianRandomProjection(n_components=reduced_dim)
        
        # 标准化数据
        scaler = StandardScaler()
        database_scaled = scaler.fit_transform(database)
        queries_scaled = scaler.transform(queries)
        
        # 应用JL降维
        database_reduced = transformer.fit_transform(database_scaled)
        queries_reduced = transformer.transform(queries_scaled)
        
        # 再次标准化降维后的数据
        database_reduced = scaler.fit_transform(database_reduced)
        queries_reduced = scaler.transform(queries_reduced)
        
        # 连接到数据库
        print("Connecting to database...")
        conn = connect_to_db()
        if conn is None:
            return
        
        # 创建表并加载数据
        print("Creating table and loading data...")
        create_table_and_load_data(conn, database_reduced, reduced_dim)
        
        # 创建IVF+JL索引
        print("Creating index...")
        n_lists = 100
        create_ivfjl_index(conn, n_lists)
        
        # 设置并行工作进程数
        cur = conn.cursor()
        cur.execute(f"SET max_parallel_workers_per_gather = {args.max_parallel_workers_per_gather};")
        conn.commit()
        cur.close()
        
        # 测试不同nprob参数
        print("Testing performance...")
        nprob_values = [1, 5]
        # nprob_values = [1, 5, 10, 20, 50]
        results = []
        
        for nprob in nprob_values:
            print(f"Testing with nprob = {nprob}")
            recall, throughput = test_knn_search(conn, queries_reduced, groundtruth, k=10, nprob=nprob)
            results.append({
                'nprob': nprob,
                'recall': recall,
                'throughput': throughput
            })
            print(f"Recall: {recall:.4f}, Throughput: {throughput:.2f} queries/second")
        
        # 输出JSON格式的结果
        print(json.dumps(results))
        
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            print("Database connection closed")

if __name__ == "__main__":
    main()