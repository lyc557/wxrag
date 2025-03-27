import re
import pandas as pd
from tqdm import tqdm  # 添加tqdm导入
import json  # Add json import
from tqdm.auto import tqdm
import pandas as pd
import pickle
import re
from logger_config import get_logger

logger = get_logger(__name__)

def build_qa_df(qa_ckpt, uuid2doc_map):
    data = []
    for key, value in tqdm(qa_ckpt.items()):
        text = value['raw_resp']
        try:
            # 尝试直接解析
            qa_list = json.loads(text)
        except json.JSONDecodeError:
            try:
                # 处理不规范的 JSON 格式
                text = text.replace("'", '"')  # 替换单引号为双引号
                text = re.sub(r'(\w+):', r'"\1":', text)  # 为属性名添加双引号
                text = re.sub(r',\s*}', '}', text)  # 修复多余的逗号
                text = re.sub(r',\s*]', ']', text)  # 修复多余的逗号
                qa_list = json.loads(text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析错误: {str(e)}\n原始文本: {text[:2000]}...")
                break
        
        for item in qa_list:
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            context = item.get('context', '').strip()
            
            if question == '' or answer == '':
                logger.warning(f"无效的 QA 对: {item}")
                continue
            data.append({
                'uuid': key,
                'question': question,
                'answer': answer,
                'context': context,
                'doc': uuid2doc_map[key]
            })
    qa_df = pd.DataFrame(data)
    return qa_df