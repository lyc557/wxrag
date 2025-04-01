# ===== 基础库 =====
import os                      # 操作系统接口，用于文件和路径操作
import gc                      # 垃圾回收，用于内存管理
import pickle                  # 序列化和反序列化Python对象
import shutil                  # 高级文件操作，如复制和删除目录
from uuid import uuid4         # 生成唯一标识符
from typing import List        # 类型注解
from datetime import datetime  # 日期和时间处理

# ===== 数据处理 =====
import numpy as np             # 数值计算库
import pandas as pd            # 数据分析和处理库
from tqdm import tqdm          # 进度条显示

# ===== 机器学习 =====
import torch                   # PyTorch深度学习框架
from sklearn.metrics.pairwise import cosine_similarity  # 计算余弦相似度

# ===== 文档处理 =====
import pypdf                   # PDF文件处理
from langchain_community.document_loaders import PyPDFLoader  # 加载PDF文档
from langchain.docstore.document import Document              # 文档对象
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 文本分割

# ===== 向量数据库 =====
import chromadb                # 向量数据库底层库
from langchain_chroma import Chroma  # LangChain的Chroma向量数据库接口

# ===== 文本嵌入 =====
import sentence_transformers   # 文本向量化基础库
from sentence_transformers import SentenceTransformer  # 文本向量化模型
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace嵌入模型接口

# ===== LangChain组件 =====
import langchain, langchain_community  # LangChain框架及社区组件
from langchain_core.prompts import PromptTemplate  # 提示词模板
from langchain_core.output_parsers import StrOutputParser  # 输出解析器
from langchain_core.runnables import RunnablePassthrough  # 可运行链组件

# ===== 配置和工具 =====
from dotenv import load_dotenv  # 环境变量管理
from logger_config import get_logger  # 日志配置
from config import (  # 项目配置
    DOC_INPUT_DIR, 
    DOC_OUTPUT_DIR, 
    DOC_SPLITTER_SEPARATORS, 
    QA_GEN_PROMPT_TMPL, 
    QA_GEN_PROMPT_TMPL_LARGE_CONTEXT, 
    QA_CHECK_PROMPT_TMPL
)

# ===== LLM模型 =====
from deepseek_chat import DeepSeekChat  # DeepSeek聊天模型接口

logger = get_logger(__name__)
class BaseLine:
    def __init__(self, input_file: str, output_dir: str, version: str = 'v1_1'):
            """初始化文档处理器"""
            self.input_file = input_file
            self.version = version
            self.output_dir = os.path.join(output_dir, f"{version}_{self._get_date()}")
            self.data_dir = os.path.join(output_dir, "data")
            logger.info(f"output_dir: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d%H%M%S")
    def _get_date(self) -> str:
        """获取日期字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d")

    def setup_environment(self):
        """设置环境变量和代理"""
        load_dotenv()
        # 从 .env 读取代理设置
        http_proxy = os.getenv('HTTP_PROXY')
        https_proxy = os.getenv('HTTPS_PROXY')

        if http_proxy:
            os.environ['HTTP_PROXY'] = http_proxy
        if https_proxy:
            os.environ['HTTPS_PROXY'] = https_proxy
        
        print("环境设置完成")


    def load_embedding_model(self,model_name="BAAI/bge-large-zh-v1.5"):
        """加载文本嵌入模型"""
        try:
            model = SentenceTransformer(model_name)
            print(f"成功加载模型: {model_name}")
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None


    def calculate_embeddings(self,model, sentences):
        """计算文本的嵌入向量"""
        if not model or not sentences:
            return None
        
        try:
            embeddings = model.encode(sentences)
            print(f"嵌入向量计算完成，形状: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"嵌入向量计算失败: {e}")
            return None


    def calculate_similarity(self,embeddings):
        """计算嵌入向量之间的相似度"""
        if embeddings is None:
            return None
        
        # 确保嵌入向量是二维数组
        embeddings_2d = np.array(embeddings).reshape(len(embeddings), -1)
        
        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(embeddings_2d)
        return similarity_matrix

    def load_document(self, input_file) -> List[Document]:
        """加载PDF文档"""
        loader = PyPDFLoader(input_file)
        return loader.load()

    def split_docs(self, documents, filepath, chunk_size=400, chunk_overlap=40, seperators=['\n\n\n', '\n\n'], force_split=False):
        # 文档切分工具
        if os.path.exists(filepath) and not force_split:
            print('found cache, restoring...')
            return pickle.load(open(filepath, 'rb'))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=seperators
        )
        split_docs = splitter.split_documents(documents)
        for chunk in split_docs:
            chunk.metadata['uuid'] = str(uuid4())

        pickle.dump(split_docs, open(filepath, 'wb'))

        return split_docs
        
    from tqdm.auto import tqdm

    def get_vector_db(self, embedding_model, docs, store_path, force_rebuild=False):
        if not os.path.exists(store_path):
            force_rebuild = True

        if force_rebuild:
            vector_db = Chroma.from_documents(
                docs,
                embedding=embedding_model,
                persist_directory=store_path
            )
        else:
            vector_db = Chroma(
                persist_directory=store_path,
                embedding_function=embedding_model
            )
        return vector_db

    def retrieve(self, vector_db, query: str, k=5):
        return vector_db.similarity_search(query, k=k)

def main():
    """主函数"""
    input_file = os.path.join(DOC_INPUT_DIR, '600519_20240403_W0YD.pdf')
    bl = BaseLine(
        input_file=input_file,
        output_dir=DOC_OUTPUT_DIR
    )
    logger.info("=== 1.处理文档 ===")
    # bl.setup_environment()
    model = bl.load_embedding_model()
    qa_file = os.path.join(bl.output_dir, 'question_answer.xlsx')
    if os.path.exists(qa_file):
        qa_df = pd.read_excel(qa_file)
        print(f"\n成功读取问答数据，共 {len(qa_df)} 条记录")
    else:
        print(f"\n问答文件不存在: {qa_file}")
    # 加载文档
    documents = bl.load_document(input_file)
    # 定义序列化文件路径
    splitted_docs_pkl = os.path.join(bl.output_dir, "splitted_docs.pkl")
    # 检查缓存是否存在
    if os.path.exists(splitted_docs_pkl) :
        logger.info("\n=== 从缓存加载QA数据 ===")
        with open(splitted_docs_pkl, 'rb') as f:
            splitted_docs = pickle.load(f)
    else:
        logger.info("\n=== 切分文档 ===")
        splitted_docs = bl.split_docs(documents, os.path.join(bl.output_dir, 'split_docs.pkl'), chunk_size=500, chunk_overlap=50)

    qa_df = pd.read_excel(os.path.join(bl.output_dir, 'question_answer.xlsx'))

    # 使用GPU加速
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 使用 MPS 加速")
    else:
        device = torch.device("cpu")
        print("⚠️ 只能使用 CPU")
    
    # 使用新的 HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={'device': str(device)},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 清空向量数据库
    store_path = os.path.join(bl.data_dir, 'chromadb', 'bge_large_v1.5')
    # if os.path.exists(store_path):
    #     shutil.rmtree(store_path)
    #     print(f"已清空向量数据库: {store_path}")
    
    # 重新构建向量数据库
    vector_db = bl.get_vector_db(embedding_model, splitted_docs, store_path=store_path)

    # 计算检索准确率
    test_df = qa_df[(qa_df['qa_type'] == 'detailed')]

    #表示要评估的不同 top-k 检索结果
    top_k_arr = list(range(1, 9))
    hit_stat_data = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = row['question']
        true_uuid = row['uuid']
        chunks = bl.retrieve(vector_db, question, k=max(top_k_arr))
        
        # 获取所有文档的UUID
        collection = vector_db._collection
        all_uuids = collection.get()['metadatas']
        all_uuids = [metadata.get('uuid') for metadata in all_uuids if metadata]

        if true_uuid not in all_uuids:
            print(f"警告: uuid {true_uuid} 不在向量数据库中")
        else:
            print(f"uuid {true_uuid} 存在于向量数据库中")

        retrieved_uuids = [doc.metadata['uuid'] for doc in chunks]
        if true_uuid not in retrieved_uuids:
            print(f"未找到正确的uuid: {true_uuid}")
            print(f"检索到的uuid: {retrieved_uuids}")
            print(f"问题: {question}")
        else:
            print(f"找到正确的uuid: {true_uuid}")

        for k in top_k_arr:
            hit_stat_data.append({
                'question': question,
                'top_k': k,
                'hit': int(true_uuid in retrieved_uuids[:k])
            })

    hit_stat_df = pd.DataFrame(hit_stat_data)
    print(hit_stat_df.groupby('top_k')['hit'].mean().reset_index())

    # 与LLM进行对话
    prompt_tmpl = """
    你是一位专业的金融分析师，专门负责解读上市公司财报和相关文件。请你：

    1. 基于以下文档片段进行分析（位于<<<<context>>>和<<<</context>>>之间）
    2. 只关注与问题直接相关的信息
    3. 保持客观专业，只陈述事实
    4. 如果信息不足，明确指出无法回答的部分

    文档内容：
    <<<<context>>>
    {context}
    <<<</context>>>

    问题：{question}

    分析回答：
    """

    retriever = vector_db.as_retriever(search_kwargs={'k': 4})

    # 构建对话链
    
    # 初始化DeepSeek模型
    model = DeepSeekChat()
    
    
    # 测试对话
    test_questions = [
        "公司的营业收入情况如何？",
        "公司的研发投入有多少？",
        "公司未来的发展战略是什么？"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        response = model.chat(prompt_tmpl.format(question=question, context=retriever.invoke(question)))
        print(f"回答: {response}")

    print("\n程序执行完成")


if __name__ == "__main__":
    main()
