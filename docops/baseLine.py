import langchain, langchain_community, pypdf, sentence_transformers, chromadb
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

# 从 .env 读取代理设置
http_proxy = os.getenv('HTTP_PROXY')
https_proxy = os.getenv('HTTPS_PROXY')

os.environ['HTTPS_PROXY'] = https_proxy
os.environ['HTTP_PROXY'] = http_proxy

# 加载 bge-large-zh-v1.5 模型
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# 需要对输入文本进行 mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # 提取所有 token 的 embedding
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# 计算文本向量
sentences = ["人工智能正在改变世界", "机器学习是一种强大的工具"]
# 直接使用 model.encode() 方法获取嵌入向量
sentence_embeddings = model.encode(sentences)

print(f"shape: {sentence_embeddings.shape}")  # (2, 1024) -> 每个文本对应 1024 维向量
# 打印每个句子的嵌入向量
for i, embedding in enumerate(sentence_embeddings):
    print(f"句子 {i+1} 的嵌入向量:")
    print(embedding)
    print(f"向量维度: {len(embedding)}\n")

# 在程序结束前添加
import gc
# 强制垃圾回收
gc.collect()
