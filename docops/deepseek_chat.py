import os
import json
import threading
import concurrent.futures
from typing import List, Dict, Any
import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL, DEEPSEEK_MAX_WORKERS  # 导入配置

class DeepSeekChat:
    def __init__(self, api_key: str = None, api_base: str = None, model: str = None):
        """初始化 DeepSeek 聊天客户端
        
        Args:
            api_key: DeepSeek API密钥，如果为None则从配置或环境变量获取
            api_base: API基础URL，如果为None则从配置获取
            model: 使用的模型，如果为None则从配置获取
        """
        self.api_key = api_key or DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供 DEEPSEEK_API_KEY")
        
        self.api_base = api_base or DEEPSEEK_API_BASE
        self.model = model or DEEPSEEK_MODEL
        
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def chat(self, prompt: str, temperature: float = 0.7, debug: bool = False) -> str:
        """发送聊天请求到 DeepSeek API
        
        Args:
            prompt: 提示文本
            temperature: 温度参数，控制响应的随机性
            debug: 是否打印调试信息
            
        Returns:
            API 返回的响应文本
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        if debug:
            print(f"发送请求: {prompt[:100]}...")
            
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API 调用失败: {str(e)}")
            raise
    
    @staticmethod
    def build_qa_prompt(prompt_tmpl: str, text: str) -> str:
        """构建QA提示
        
        Args:
            prompt_tmpl: 提示模板
            text: 要处理的文本
        """
        prompt = prompt_tmpl.replace('{{document}}', text).strip()
        return prompt

    def gen_qa(self, splitted_docs, prompt_tmpl, qa_ckpt_filename):
        
        """生成问题和答案
        
       
        Args:
            splitted_docs: 文档列表
            prompt_tmpl: 提示模板
            qa_ckpt_filename: 问题和答案的检查点文件名

        Returns:
            {
                "uuid": "xxx",  # 文档块的唯一标识符
                "raw_resp": "[{
                    'question': '问题描述',
                    'context': '原文引用',
                    'answer': '基于上下文的答案'
                }, ...]"  # 字符串形式的JSON数组
            }
        """
         # 打印参数信息
        print(f"传入的文档数量: {len(splitted_docs)}")
        print(f"提示模板: {prompt_tmpl}")
        print(f"检查点文件名: {qa_ckpt_filename}")

        
        qa_ckpt = {}
        if os.path.exists(qa_ckpt_filename):
            qa_ckpt = open(qa_ckpt_filename).readlines()
            qa_ckpt = [json.loads(line.strip()) for line in qa_ckpt if line.strip() != '']
            qa_ckpt = {item['uuid']: item for item in qa_ckpt}
            print(f'found checkpoint, item count: {len(qa_ckpt)}')

        file_lock = threading.Lock()
        # 使用配置中的并发工作线程数量
        with concurrent.futures.ThreadPoolExecutor(max_workers=DEEPSEEK_MAX_WORKERS) as executor:
            futures = {doc.metadata['uuid']: executor.submit(
                self.chat, 
                self.build_qa_prompt(prompt_tmpl, doc.page_content), 
                0.7, False
            ) for doc in splitted_docs if len(doc.page_content.replace('\n', '')) >= 150 and doc.metadata['uuid'] not in qa_ckpt}
            
            for uuid in tqdm(futures):
                future = futures[uuid]
                result = future.result()
                if result is None:
                    continue

                item = {'uuid': uuid,'raw_resp': result}
                qa_ckpt[uuid] = item

                file_lock.acquire()
                try:
                    with open(qa_ckpt_filename, 'a') as f:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(e)
                finally:
                    file_lock.release()
        return qa_ckpt

def main():
    # 使用示例
    chat = DeepSeekChat()
    
    prompt = "请简要介绍一下你自己。"
    response = chat.chat(prompt)
    print(f"问题: {prompt}")
    print(f"回答: {response}")

if __name__ == "__main__":
    main()