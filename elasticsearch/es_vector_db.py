import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from elasticsearch import Elasticsearch
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings

from docops.logger_config import get_logger

# 初始化日志记录器
logger = get_logger(__name__)

class ESVectorDB:
    """Elasticsearch 向量数据库类，用于文档向量化和检索"""
    
    def __init__(self, es_url: str = "http://localhost:9200"):
        """初始化 Elasticsearch 向量数据库
        
        Args:
            es_url: Elasticsearch 服务器 URL
        """
        self.es_url = es_url
        try:
            self.es_client = Elasticsearch(es_url)
            logger.info(f"成功连接到 Elasticsearch: {es_url}")
        except Exception as e:
            logger.error(f"连接 Elasticsearch 失败: {e}")
            self.es_client = None
    
    def create_index(self, index_name: str, dims: int = 1024, force_recreate: bool = False):
        """创建向量索引
        
        Args:
            index_name: 索引名称
            dims: 向量维度
            force_recreate: 是否强制重建索引
        
        Returns:
            创建是否成功
        """
        if self.es_client is None:
            logger.error("Elasticsearch 客户端未初始化")
            return False
            
        # 检查索引是否存在
        index_exists = self.es_client.indices.exists(index=index_name)
        
        # 如果索引存在且需要重建，则删除
        if index_exists and force_recreate:
            self.es_client.indices.delete(index=index_name)
            logger.info(f"已删除现有索引: {index_name}")
            index_exists = False
        
        # 如果索引不存在，则创建
        if not index_exists:
            # 定义索引映射
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "vector": {"type": "dense_vector", "dims": dims},
                        "metadata": {"type": "object"}
                    }
                }
            }
            
            try:
                self.es_client.indices.create(index=index_name, body=mapping)
                logger.info(f"成功创建索引: {index_name}")
                return True
            except Exception as e:
                logger.error(f"创建索引失败: {e}")
                return False
        else:
            logger.info(f"索引已存在: {index_name}")
            return True
    
    def index_documents(self, docs: List[Document], embedding_model: Embeddings, 
                       index_name: str, batch_size: int = 100):
        """索引文档
        
        Args:
            docs: 文档列表
            embedding_model: 嵌入模型
            index_name: 索引名称
            batch_size: 批处理大小
        
        Returns:
            索引是否成功
        """
        if self.es_client is None:
            logger.error("Elasticsearch 客户端未初始化")
            return False
            
        if not docs:
            logger.warning("没有文档需要索引")
            return False
            
        # 获取第一个文档的向量维度
        try:
            sample_embedding = embedding_model.embed_documents([docs[0].page_content])[0]
            dims = len(sample_embedding)
            logger.info(f"向量维度: {dims}")
            
            # 创建索引
            if not self.create_index(index_name, dims):
                return False
                
        except Exception as e:
            logger.error(f"获取向量维度失败: {e}")
            return False
        
        # 批量索引文档
        total_batches = (len(docs) + batch_size - 1) // batch_size
        success_count = 0
        
        for i in tqdm(range(0, len(docs), batch_size), total=total_batches, desc="索引文档"):
            batch_docs = docs[i:i+batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]
            
            try:
                # 获取嵌入向量
                batch_embeddings = embedding_model.embed_documents(batch_texts)
                
                # 准备批量索引请求
                bulk_data = []
                for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
                    doc_id = doc.metadata.get('uuid', f"{i+j}")
                    
                    # 索引操作
                    bulk_data.append({"index": {"_index": index_name, "_id": doc_id}})
                    
                    # 文档内容
                    doc_data = {
                        "content": doc.page_content,
                        "vector": embedding,
                        "metadata": doc.metadata
                    }
                    bulk_data.append(doc_data)
                
                # 执行批量索引
                if bulk_data:
                    response = self.es_client.bulk(body=bulk_data, refresh=True)
                    if not response.get("errors", True):
                        success_count += len(batch_docs)
                    else:
                        logger.warning(f"批量索引部分失败: {response}")
                
            except Exception as e:
                logger.error(f"批量索引失败: {e}")
        
        logger.info(f"成功索引 {success_count}/{len(docs)} 个文档")
        return success_count > 0
    
    def search(self, query: str, embedding_model: Embeddings, index_name: str, 
              k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None):
        """向量搜索
        
        Args:
            query: 查询文本
            embedding_model: 嵌入模型
            index_name: 索引名称
            k: 返回结果数量
            metadata_filter: 元数据过滤条件
            
        Returns:
            检索到的文档列表
        """
        if self.es_client is None:
            logger.error("Elasticsearch 客户端未初始化")
            return []
            
        try:
            # 获取查询向量
            query_vector = embedding_model.embed_query(query)
            
            # 构建查询
            search_query = {
                "size": k,
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
            
            # 添加元数据过滤
            if metadata_filter:
                filter_conditions = []
                for key, value in metadata_filter.items():
                    filter_conditions.append({"term": {f"metadata.{key}": value}})
                
                search_query["query"] = {
                    "bool": {
                        "must": search_query["query"],
                        "filter": filter_conditions
                    }
                }
            
            # 执行搜索
            response = self.es_client.search(index=index_name, body=search_query)
            
            # 处理结果
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                content = source.get('content', '')
                metadata = source.get('metadata', {})
                score = hit['_score']
                
                doc = Document(page_content=content, metadata={**metadata, "score": score})
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []