from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
import os
import pickle
import re
from typing import List, Dict, Tuple
from deepseek_chat import DeepSeekChat
from config import DOC_INPUT_DIR, DOC_OUTPUT_DIR, DOC_SPLITTER_SEPARATORS, QA_GEN_PROMPT_TMPL, QA_GEN_PROMPT_TMPL_LARGE_CONTEXT
from jsonutil import  build_qa_df
import pandas as pd

class DocumentProcessor:
    def __init__(self, input_file: str, output_dir: str, version: str = 'v1_1'):
        """初始化文档处理器
        
        Args:
            input_file: 输入PDF文件路径
            output_dir: 输出目录
            version: 版本号
        """
        self.input_file = input_file
        self.version = version
        self.output_dir = os.path.join(output_dir, f"{version}_{self._get_timestamp()}")
        print(f"output_dir: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d%H%M%S")
    def _get_date(self) -> str:
        """获取日期字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d")
    
    def load_document(self) -> List[Document]:
        """加载PDF文档"""
        loader = PyPDFLoader(self.input_file)
        return loader.load()

#     def F(documents, filepath, chunk_size=400, chunk_overlap=40, seperators=['\n\n\n', '\n\n'], force_split=False):

    def split_doc(self, documents: List[Document], 
                       chunk_size: int, 
                       chunk_overlap: int,
                       cache_file: str,
                       force_split: bool = False) -> List[Document]:
        """分割文档
        
        Args:
            documents: 要分割的文档列表
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            cache_file: 缓存文件路径
            force_split: 是否强制重新分割
        """
        cache_path = os.path.join(self.output_dir, cache_file)
        
        if os.path.exists(cache_path) and not force_split:
            print('找到缓存，正在恢复...')
            return pickle.load(open(cache_path, 'rb'))
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=DOC_SPLITTER_SEPARATORS
        )
        
        splited_docs = splitter.split_documents(documents)
        
        # 为每个分块添加UUID
        for chunk in splited_docs:
            chunk.metadata['uuid'] = str(uuid4())
        
        # 保存缓存
        pickle.dump(splited_docs, open(cache_path, 'wb'))
        
        return splited_docs

    def clean_headers_and_footers(self, documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """自动检测并清理页眉页脚
        
        Args:
            documents: 要处理的文档列表
            
        Returns:
            Tuple[List[Document], List[Document]]: 清理后的文档列表和合并后的文档列表
        """
        # 提取可能的页眉页脚模式
        headers = []
        footers = []
        
        # 分析前10页和后10页来识别页眉页脚模式
        sample_pages = documents[:10] + documents[-10:] if len(documents) > 20 else documents
        
        for doc in sample_pages:
            lines = doc.page_content.split('\n')
            if len(lines) > 3:
                # 提取前3行作为可能的页眉
                headers.append('\n'.join(lines[:3]))
                # 提取后3行作为可能的页脚
                footers.append('\n'.join(lines[-3:]))
        
        # 找出频繁出现的页眉页脚模式
        from collections import Counter
        header_counter = Counter(headers)
        footer_counter = Counter(footers)
        
        # 获取出现频率最高的页眉页脚
        common_headers = [h for h, count in header_counter.items() if count > 1]
        common_footers = [f for f, count in footer_counter.items() if count > 1]
        
        print(f"检测到的页眉模式: {common_headers[:2]}")
        print(f"检测到的页脚模式: {common_footers[:2]}")
        
        # 构建页眉页脚清理的正则表达式
        header_patterns = [re.escape(h) for h in common_headers]
        footer_patterns = [re.escape(f) for f in common_footers]
        
        # 添加已知的页眉模式
        header_patterns.append(r"^\d{4}年年度报告")
        
        # 清理页眉页脚
        cleaned_docs = []
        for doc in documents:
            content = doc.page_content
            # 清理页眉
            for pattern in header_patterns:
                content = re.sub(pattern, '', content, flags=re.MULTILINE)
            # 清理页脚
            for pattern in footer_patterns:
                content = re.sub(pattern, '', content, flags=re.MULTILINE)
            # 清理多余的空行
            content = re.sub(r'\n{3,}', '\n\n', content)
            cleaned_docs.append(Document(page_content=content, metadata=doc.metadata))
        
        # 合并文档
        merged_docs = [Document(page_content='\n'.join(doc.page_content for doc in cleaned_docs))]
        
        return cleaned_docs, merged_docs

    def process(self) -> tuple:
        """处理文档的主流程"""
        
        # 1. langchain_community的提供的PyPDFLoader读取文档
        documents = self.load_document()
        print(f"文档页数: {len(documents)}")
        
        # 2. 对文档进行初步的清洗 自动检测并清理页眉页脚
        cleaned_docs, merged_docs = self.clean_headers_and_footers(documents)
        
        # 3. 分割文档（两种粒度）
        splitted_docs = self.split_doc(
            cleaned_docs,  # 使用清理后的文档
            chunk_size=500, 
            chunk_overlap=50,
            cache_file='split_docs.pkl',
            force_split=True
        )
        
        print(f"\n文档块数量: {len(splitted_docs)}")
        
        splitted_docs_large = self.split_doc(
            merged_docs, 
            chunk_size=1500, 
            chunk_overlap=100,
            cache_file='split_docs_large.pkl',
            force_split=True
        )
        
        return splitted_docs, splitted_docs_large

    @staticmethod
    def build_qa_prompt(prompt_tmpl: str, text: str) -> str:
        """构建QA提示
        
        Args:
            prompt_tmpl: 提示模板
            text: 要处理的文本
        """
        prompt = prompt_tmpl.replace('{{document}}', text).strip()
        return prompt

def main():
    """主函数"""
    # 修改为读取 600519_20240403_W0YD.pdf 文件
    input_file = os.path.join(DOC_INPUT_DIR, '600519_20240403_W0YD.pdf')

    # 0.初始化文档处理器
    processor = DocumentProcessor(
        input_file=input_file,
        output_dir=DOC_OUTPUT_DIR
    )
    
    # 1.处理文档
    print("\n=== 1.处理文档 ===")
    splitted_docs, splitted_docs_large = processor.process()
    
    # 2. 通过提示词生成QA对
    print("\n=== 2.生成QA对 ===")
    qa_gen_prompt_tmpl = QA_GEN_PROMPT_TMPL  # 使用配置文件中的模板
    qa_gen_prompt_tmpl_large_context = QA_GEN_PROMPT_TMPL_LARGE_CONTEXT  # 使用配置文件中的长上下文模板

    # 检查序列化数据是否存在
    detailed_qa_pkl = os.path.join(processor.output_dir, 'detailed_qa_dict.pkl')
    large_context_qa_pkl = os.path.join(processor.output_dir, 'large_context_qa_dict.pkl')
    uuid2doc_pkl = os.path.join(processor.output_dir, 'uuid2doc.pkl')
    large_context_qa_pkl = os.path.join(processor.output_dir, 'large_context_qa_dict.pkl')

    if os.path.exists(detailed_qa_pkl) and os.path.exists(large_context_qa_pkl) and os.path.exists(uuid2doc_pkl) and os.path.exists(large_context_qa_pkl):
        print("\n=== 从缓存加载QA数据 ===")
        with open(detailed_qa_pkl, 'rb') as f:
            detailed_qa_dict = pickle.load(f)
        with open(large_context_qa_pkl, 'rb') as f:
            large_context_qa_dict = pickle.load(f)
        with open(uuid2doc_pkl, 'rb') as f:
            uuid2doc = pickle.load(f)
        with open(large_context_qa_pkl, 'rb') as f:
            large_context_qa_dict = pickle.load(f)
    else:
        print("\n=== 生成新的QA数据 ===")
        dpchat = DeepSeekChat()
        # 2. 构建uuid2doc字典
        uuid2doc = {doc.metadata['uuid']: doc.page_content for doc in splitted_docs}
        uuid2large_doc = {doc.metadata['uuid']: doc.page_content for doc in splitted_docs_large}

        # 短上下文抽取结果 暂时取前十条数据
        detailed_qa_dict = dpchat.gen_qa(splitted_docs[:100], qa_gen_prompt_tmpl, 
                                       os.path.join(processor.output_dir, "qa_ckpt_detailed.jsonl"))
        # 长上下文抽取结果 暂时取前十条数据
        large_context_qa_dict = dpchat.gen_qa(splitted_docs_large[:100], qa_gen_prompt_tmpl_large_context, 
                                            os.path.join(processor.output_dir, "qa_ckpt_large_context.jsonl"))
        
        # 保存序列化数据
        with open(detailed_qa_pkl, 'wb') as f:
            pickle.dump(detailed_qa_dict, f)
        with open(large_context_qa_pkl, 'wb') as f:
            pickle.dump(large_context_qa_dict, f)
        with open(uuid2doc_pkl, 'wb') as f:
            pickle.dump(uuid2doc, f)
        with open(large_context_qa_pkl, 'wb') as f:
            pickle.dump(large_context_qa_dict, f)
    
    print(f"\n详细QA对数量: {len(detailed_qa_dict)}")
    print(f"\n长上下文QA对数量: {len(large_context_qa_dict)}")

    qa_df = build_qa_df(detailed_qa_dict, uuid2doc)
    large_context_qa_df = build_qa_df(large_context_qa_dict, uuid2large_doc)

if __name__ == "__main__":
    main()