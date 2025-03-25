# DeepSeek API 配置
# DEEPSEEK_API_KEY = "sk-f6080528cdbb4bfe9e4abe8fefdc3aea"
# DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
# DEEPSEEK_MODEL = "deepseek-chat"
# DEEPSEEK_MAX_WORKERS = 1  # 并发工作线程数量


# 本地API配置
DEEPSEEK_API_KEY = "test-api-key"
DEEPSEEK_API_BASE = "http://127.0.0.1:8000/v1"
# DEEPSEEK_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEEPSEEK_MAX_WORKERS = 1  # 并发工作线程数量



# 文档处理配置
DOC_INPUT_DIR = "/Users/yangcailu/traeCode/wxrag/docops/docs/"
DOC_OUTPUT_DIR = "outputs"

# 文档分割配置
DOC_SPLITTER_SEPARATORS = ["\n\n", "。", "！", "？", "；", "，", " "]