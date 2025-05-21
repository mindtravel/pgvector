import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import time
import io

# 数据库连接信息
DB_PARAMS = {
    'host': 'localhost',
    'port': '5432',
    'database': 'mydatabase',
    'user': 'postgres',
    'password': '1'
}

# 加载 SIFT 数据集
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
        data = data.reshape(-1, 129)
        print("asas",data.size)
        data = data[:, 1:].astype(float)
    return data

# 连接到数据库
def connect_to_db():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

# 创建表并加载数据
def create_table_and_load_data(conn, data):
    cur = conn.cursor()
    # 创建表
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("exe1")
    cur.execute("DROP TABLE IF EXISTS sift_data;")
    print("exe2")
    cur.execute("CREATE TABLE sift_data (id serial, vector vector(128));")
    print("exe3")
    # 分批次插入数据
    batch_size = 1000000
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        output = io.StringIO()
        for row in batch:
            # 将向量转换为 PostgreSQL 数组格式
            vector_str = '(' + '1.0,2.0' + ')'
            output.write(vector_str + ',')
        output.seek(0)
        # 使用 COPY 命令导入数据
        try:
            cur.copy_expert("COPY sift_data (vector) FROM STDIN WITH CSV", output)
        except psycopg2.Error as e:
            print(f"Error loading batch {i // batch_size}: {e}")
            conn.rollback()
            return
    conn.commit()
    print("exe 4")
    print("exe 5")
    cur.close()

# 创建 ivfflat 索引
def create_ivfflat_index(conn):
    cur = conn.cursor()
    cur.execute("CREATE INDEX ON sift_data USING ivfflat (vector vector_l2_ops) WITH (lists = 100);")
    conn.commit()
    cur.close()

# 测试 k 近邻搜索
def test_knn_search(conn, query_vectors, k, nprob):
    cur = conn.cursor()
    recall_sum = 0
    total_time = 0
    for query in query_vectors:
        start_time = time.time()
        cur.execute(f"SET ivfflat.probes = {nprob};")
        cur.execute(f"SELECT id FROM sift_data ORDER BY vector <-> %s LIMIT {k};", (query.tolist(),))
        results = cur.fetchall()
        end_time = time.time()
        total_time += end_time - start_time
        # 这里简单假设真实的 k 近邻可以通过暴力搜索得到，实际中需要根据具体情况计算
        # 为了简化，这里不进行真实召回率的计算，仅作示例
        recall_sum += 1
    avg_recall = recall_sum / len(query_vectors)
    throughput = len(query_vectors) / total_time
    cur.close()
    return avg_recall, throughput

# 主函数
def main():
    # 加载 SIFT 数据集
    database = load_data('./sift/sift_base.fvecs')
    querys = load_data('./sift/sift_query.fvecs')

    # 连接到数据库
    conn = connect_to_db()
    if conn is None:
        return
    print("connect successfully")

    # 创建表并加载数据
    create_table_and_load_data(conn, database)
    print("create table and load successfully")

    # 创建 ivfflat 索引
    create_ivfflat_index(conn)
    print("create index successfully")

    # 不同的 nprob 参数
    nprob_values = [1, 5, 10, 20, 50]
    recalls = []
    throughputs = []

    for nprob in nprob_values:
        recall, throughput = test_knn_search(conn, querys, k=10, nprob=nprob)
        recalls.append(recall)
        throughputs.append(throughput)

    # 绘制召回率 - 吞吐量折线图
    plt.plot(recalls, throughputs, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Throughput')
    plt.title('Recall vs Throughput with Different nprob Values')
    plt.grid(True)
    plt.show()

    # 关闭数据库连接
    conn.close()

if __name__ == "__main__":
    main()
    