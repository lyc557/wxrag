# 企业级RAG系统

这是一个基于检索增强生成(Retrieval Augmented Generation, RAG)技术的企业级知识库问答系统。

## 项目概述

本项目旨在构建一个企业级的RAG系统，能够从企业内部文档、知识库中检索相关信息，并结合大语言模型生成准确、相关的回答，帮助企业员工快速获取所需信息。

## 系统架构

系统采用前后端分离架构：

- **前端**：基于React的Web应用
- **后端**：Python FastAPI服务
- **向量数据库**：存储文档的向量表示
- **文档处理**：文档解析、分块和向量化
- **LLM集成**：与大语言模型的集成接口

## 核心功能

1. **文档管理**：上传、更新和删除企业文档
2. **文档处理**：自动解析不同格式的文档，进行分块和向量化
3. **智能检索**：基于语义相似度的文档检索
4. **问答生成**：结合检索结果和大语言模型生成回答
5. **用户管理**：用户权限和访问控制
6. **使用分析**：系统使用情况统计和分析

## 技术栈

- **前端**：React, TypeScript, Ant Design
- **后端**：Python, FastAPI
- **向量数据库**：Milvus/Pinecone/Weaviate
- **嵌入模型**：各种开源或商业嵌入模型
- **LLM**：支持多种大语言模型接口

## 项目结构
### WXRAG项目 rag主项目
wxrag/
├── frontend/            # 前端React应用
├── backend/             # 后端FastAPI服务
│   ├── api/             # API路由
│   ├── core/            # 核心业务逻辑
│   ├── models/          # 数据模型
│   ├── services/        # 服务层
│   └── utils/           # 工具函数
│   └── document_processor/  # 文档处理模块
├── scripts/             # 部署和维护脚本
├── docs/                # 项目文档
└── tests/               # 测试代码

### mlx_openai_api_server open-ai接口项目
使用FASTAPI实现的open-ai接口

### 模型 open-ai鉴权项目

### 文档处理项目
使用LangChain实现的文档处理


这个实现完成了 RAG 服务的核心功能：

1. _retrieve_documents : 从向量数据库中检索相关文档
2. _build_context : 将检索到的文档构建成上下文
3. _generate_answer : 使用 LLM 基于上下文生成回答
4. _build_references : 构建引用信息，用于前端展示
5. index_document : 将文档索引到向量存储中
这个服务可以与文档处理服务配合使用，实现完整的 RAG 流程：文档处理 → 文档索引 → 查询检索 → 回答生成。