# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-a30b61682b6249f9842fa6f91b89139a"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_WORKERS = 2  # 并发工作线程数量


# 本地API配置
# DEEPSEEK_API_KEY = "test-api-key"
# DEEPSEEK_API_BASE = "http://127.0.0.1:8000/v1"
# # DEEPSEEK_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# DEEPSEEK_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# DEEPSEEK_MAX_WORKERS = 1  # 并发工作线程数量



# 文档处理配置
DOC_INPUT_DIR = "./docops/docs"
DOC_OUTPUT_DIR = "./docops/outputs"

# 文档分割配置
DOC_SPLITTER_SEPARATORS = ["\n\n", "。", "！", "？"]


# QA 生成提示词模板
QA_GEN_PROMPT_TMPL = """
我会给你一段文本（<document></document>之间的部分），请仔细阅读并生成5个高质量的问答对。要求如下
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
如果文本主要是目录、人名列表、联系方式等无实质内容的信息，请返回空数组 []。

返回格式：
[
    {
        "question": "问题描述",
        "context": "原文引用",
        "answer": "基于上下文的答案"
    } 
    ...
]

下方是待分析文本：
<document>
{{document}}
</document>
"""

# 长上下文 QA 生成提示词模板
QA_GEN_PROMPT_TMPL_LARGE_CONTEXT = """
我会给你一段文本（<document></document>之间的部分），请仔细阅读并生成2个深度问答对。要求如下：

1. 问题要求：
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
如果文本主要是目录、人名列表、联系方式等无实质内容的信息，请返回空数组 []。

返回格式：
[
    {
        "question": "深度分析性问题",
        "context": "原文引用",
        "answer": "综合性回答"
    }
   ...
]

下方是待分析文本：
<document>
{{document}}
</document>
"""


QA_CHECK_PROMPT_TMPL = """
你是一个优秀的NLP方面的助教，你的任务是帮助检查出题组所出的期末考试的题目。
你需要根据所出的问题（<question></question>之间的部分），以及参考答案（<answer></answer>）进行打分，并给出打分理由，分值是一个int类型的值，取值范围为1-5。
好的问题，应该是询问实时、观点等，而不是类似于“这一段描述了什么”，“文本描述了什么”；
好的答案，应该能够直接回答问题，而不是给出在原文中的引用，例如“第3章”等

结果请以JSON形式组织，格式为如下：
{"score": ..., "reason": ...}

问题：
<question>
{{question}}
</question>

答案：
<answer>
{{answer}}
</answer>

请打分：
"""