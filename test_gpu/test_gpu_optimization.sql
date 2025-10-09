-- 测试GPU优化功能的SQL脚本
-- 验证聚类中心数据只上传一次的优化效果

-- sudo -u postgres psql -f test_gpu_optimization.sql postgres

-- 创建测试表
DROP TABLE IF EXISTS test_vectors;
CREATE TABLE test_vectors (id int, embedding vector(128));

-- 插入测试数据（使用固定模式，确保有匹配结果）
INSERT INTO test_vectors (id, embedding) 
SELECT 
    i,
    (SELECT array_agg(sin(i + j)) FROM generate_series(1, 128) j)::vector(128)
FROM generate_series(1, 1000) i;

-- 创建IVFFlat索引
DROP INDEX IF EXISTS test_vectors_ivfflat_idx;
CREATE INDEX test_vectors_ivfflat_idx ON test_vectors 
USING ivfflat (embedding vector_l2_ops) WITH (lists = 50);

-- 生成查询向量（使用与数据相似的模式）
-- 第一次查询 - 初始化GPU上下文并上传聚类中心
\timing on
SELECT id FROM test_vectors 
ORDER BY embedding <-> (SELECT array_agg(sin(100 + j)) FROM generate_series(1, 128) j)::vector(128) 
LIMIT 5;

-- 后续查询 - 应该更快，因为聚类中心已在GPU中
SELECT id FROM test_vectors 
ORDER BY embedding <-> (SELECT array_agg(sin(200 + j)) FROM generate_series(1, 128) j)::vector(128) 
LIMIT 5;

SELECT id FROM test_vectors 
ORDER BY embedding <-> (SELECT array_agg(sin(300 + j)) FROM generate_series(1, 128) j)::vector(128) 
LIMIT 5;
\timing off

-- 清理测试数据
DROP TABLE test_vectors;
