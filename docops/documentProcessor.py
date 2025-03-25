from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
import os
import pickle
import re
from typing import List, Dict, Tuple
from deepseek_chat import DeepSeekChat
from config import DOC_INPUT_DIR, DOC_OUTPUT_DIR, DOC_SPLITTER_SEPARATORS
from jsonutil import convert2json, build_qa_df
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
        self.output_dir = os.path.join(output_dir, f"{version}_{self._get_date()}")
        print(f"output_dir: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        
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
        
        # 4. 创建UUID到文档内容的映射
        uuid2doc = {doc.metadata['uuid']: doc.page_content for doc in splitted_docs}
        uuid2large_doc = {doc.metadata['uuid']: doc.page_content for doc in splitted_docs_large}
        
        return splitted_docs, splitted_docs_large, uuid2doc, uuid2large_doc

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

    processor = DocumentProcessor(
        input_file=input_file,
        output_dir=DOC_OUTPUT_DIR
    )
    
    # 处理文档
    splitted_docs, splitted_docs_large, uuid2doc, uuid2large_doc = processor.process()
    
    # 5. 通过提示词生成QA对
    qa_gen_prompt_tmpl = """
    我会给你一段文本（<document></document>之间的部分），请仔细阅读并生成8个高质量的问答对。要求如下：

    1. 问题要求：
       - 问题必须与文本内容直接相关
       - 避免询问文档结构相关的问题（如"在哪一章"）
       - 问题应该有实质性的信息价值
       - 优先生成需要理解和分析的问题，而不是简单的事实查找

    2. 上下文要求：
       - 必须是原文的直接引用，不允许任何形式的改写
       - 应该包含完整的相关信息，确保上下文自包含
       - 如果信息分散在多处，可以用"..."连接相关段落

    3. 答案要求：
       - 基于上下文直接回答问题
       - 保持完整性和准确性
       - 简明扼要，避免冗余
       - 使用肯定的语气
       - 不要引用文档结构（如章节、页码）

    返回格式：
    [
        {
            "question": "问题描述",
            "context": "原文引用",
            "answer": "基于上下文的答案"
        },
        ...
    ]

    如果文本主要是目录、人名列表、联系方式等无实质内容的信息，请返回空数组 []。

    下方是待分析文本：
    <document>
    {{document}}
    </document>
    """

    qa_gen_prompt_tmpl_large_context = """
    我会给你一段文本（<document></document>之间的部分），请仔细阅读并生成2个深度问答对。要求如下：

    1. 问题要求：
       - 问题要求：
         - 问题必须需要综合理解大段文本才能回答
         - 关注公司战略、业务发展、财务表现等深层次问题
         - 避免过于具体的数据细节（如具体增长率）
         - 避免空泛的概述性问题（如"主要讲了什么"）
         - 优先考虑：
           * 业务分析（如"公司核心业务的竞争优势"）
           * 战略评估（如"公司未来发展战略及其可行性"）
           * 风险分析（如"公司面临的主要风险及应对措施"）
           * 财务表现（如"公司盈利能力的变化趋势及原因"）

    2. 答案要求：
       - 全面性：需要整合文本中的多个相关信息
       - 逻辑性：清晰展示因果关系或逻辑推理
       - 完整性：确保回答涵盖问题的各个方面
       - 准确性：严格基于文本内容，避免过度推测
       - 简洁性：去除冗余，突出关键信息

    返回格式：
    [
        {
            "question": "深度分析性问题",
            "answer": "综合性回答"
        },
        {
            "question": "深度分析性问题",
            "answer": "综合性回答"
        }
    ]

    如果文本主要是目录、人名列表、联系方式等无实质内容的信息，请返回空数组 []。

    下方是待分析文本：
    <document>
    {{document}}
    </document>
    """
    # 打印示例
    print("\n=== 生成QA对示例（500字符）===")
    dpchat = DeepSeekChat()

    # 短上下文抽取结果
    detailed_qa_dict = dpchat.gen_qa(splitted_docs, qa_gen_prompt_tmpl, 
                                   os.path.join(processor.output_dir, "qa_ckpt_detailed.jsonl"))
    # 长上下文抽取结果
    large_context_qa_dict = dpchat.gen_qa(splitted_docs_large, qa_gen_prompt_tmpl_large_context, 
                                        os.path.join(processor.output_dir, "qa_ckpt_large_context.jsonl"))

    print(f"\n详细QA对数量: {len(detailed_qa_dict)}")
    print(f"\n长上下文QA对数量: {len(large_context_qa_dict)}")
    # 打印全部结果
    # for qa in detailed_qa_dict:
    #     print(f"\n问题: {qa['question']}")
    #     print(f"答案: {qa['answer']}")
    #     print(f"上下文: {qa['context']}")
    #     print(f"UUID: {qa['uuid']}")   

    # for qa in large_context_qa_dict:
    #     print(f"\n问题: {qa['question']}")
    #     print(f"答案: {qa['answer']}")
    #     print(f"UUID: {qa['uuid']}") 

    # 使用正则表达式进行后置处理，提取JSON部分
    # qa_df = build_qa_df(detailed_qa_dict, uuid2doc)
    # qa_df.drop_duplicates('question', inplace=True)
    # qa_df['qa_type'] = 'detailed'
    # large_context_qa_df = build_qa_df(large_context_qa_dict, uuid2large_doc)
    # large_context_qa_df.drop_duplicates('question', inplace=True)
    # large_context_qa_df['qa_type'] = 'large_context'

    qa_df = pd.concat([qa_df, large_context_qa_df])


if __name__ == "__main__":
    main()