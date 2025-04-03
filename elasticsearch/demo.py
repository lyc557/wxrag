from elasticsearch import Elasticsearch
import time

# 连接 ES 实例
es = Elasticsearch("http://10.45.9.130:9200")

# 定义索引名称和映射（这里示例使用 3 维向量，可根据实际情况调整 dims 参数）
index_name = "vector_index"
mapping = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "vector": {"type": "dense_vector", "dims": 3}
        }
    }
}

# 如果索引不存在，则创建索引
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)
    # 创建索引后稍微等待一下，确保索引已经建立
    time.sleep(1)

# 存储示例文档，包含文本和向量数据
docs = [
    {"text": "这是文档1", "vector": [0.1, 0.2, 0.3]},
    {"text": "这是文档2", "vector": [0.2, 0.1, 0.4]},
    {"text": "这是文档3", "vector": [0.4, 0.4, 0.2]},
]

# 批量索引数据
for doc in docs:
    es.index(index=index_name, body=doc)

# 刷新索引，确保文档可搜索
es.indices.refresh(index=index_name)

# 定义查询向量，准备查找相似向量
query_vector = [0.2, 0.2, 0.3]

# 构造基于 script_score 的向量搜索查询
# 此处使用 cosineSimilarity 计算余弦相似度，加上 1.0 保证分数为正值
search_query = {
    "size": 3,
    "query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
}

# 执行搜索查询
res = es.search(index=index_name, body=search_query)
print("搜索结果：")
for hit in res['hits']['hits']:
    print("文档内容：", hit['_source'], "分数：", hit['_score'])
