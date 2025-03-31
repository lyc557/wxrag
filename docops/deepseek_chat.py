import os
import json
import threading
import concurrent.futures
from typing import List, Dict, Any
import requests
import re  # 添加 re 模块的导入
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

    @staticmethod
    def build_qa_scoring_prompt(prompt_teml: str,row):
        # QA质量检查prompt
        context = row['context']
        question = row['question']
        answer = row['answer']
        prompt = prompt_teml.replace('{{question}}', question).replace('{{answer}}', answer)
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

                item = {'uuid': uuid, 'raw_resp': result}
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

    def extract_json(self,text):
        """
        从raw_resp中提取出打分结果的JSON
        :param raw_resp: LLM的返回结果，数据样例：
        {\n"score": 5, \n"reason": "问题提出了一个具体且明确的观点询问，即针对美国房地产市场的风险评估，特别是对中小型银行的影响。参考答案直接回答了问题，并提供了详细的解释和原因，没有引用原始文本的位置，而是直接给出了分析性的内容。"\n}
        Based on the given criteria:\n\n- The question asks for a factual assessment of global trade performance in 2023, which is clear and direct.\n- The provided answer directly addresses the question without referring back to a source or chapter, offering an evaluation of global trade conditions in 2023.\n\nBoth the question and the answer meet the criteria for being high-quality and appropriately structured. Therefore, I would give them a high score.\n\n```json\n{"score": 5, "reason": "The question is clear and direct, seeking a factual evaluation. The answer provides a direct response without referring to specific sources or sections."}\n```
        """
        # pattern = r'\n```json\n(.*?)\n```'
        pattern = r'\{.*?\}'
        
        ret = {}
        try:
            ret = json.loads(text)
        except:
            match = re.search(pattern, text, re.DOTALL)
            try:
                matched = match.group(0)
                ret = json.loads(matched)
            except Exception as e:
                print(f"{match}, {str(e)}")
                
        return ret

    def score_qa_pairs(self,qa_df, prompt,qa_scoring_ckpt_filename):
        """对QA对进行评分
        
        Args:
            qa_df: 包含问答对的DataFrame
            output_dir: 输出目录
        Returns:
            tuple: (qa_df, hq_qa_df) - 完整的QA DataFrame和高质量QA DataFrame
        """
        qa_scoring_ckpt = {}
        if os.path.exists(qa_scoring_ckpt_filename):
            qa_scoring_ckpt = open(qa_scoring_ckpt_filename).readlines()
            qa_scoring_ckpt = [json.loads(line.strip()) for line in qa_scoring_ckpt if line.strip() != '']
            qa_scoring_ckpt = {item['question']: item for item in qa_scoring_ckpt}
            print(f'找到检查点，项目数量: {len(qa_scoring_ckpt)}')

        file_lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=DEEPSEEK_MAX_WORKERS) as executor:
            futures = {
                row['question']: executor.submit(
                    self.chat, 
                    self.build_qa_scoring_prompt(prompt, row),
                    0.7, False
                )for _, row in qa_df.iterrows() if row['question'] not in qa_scoring_ckpt
            }
            
            for question in tqdm(futures):
                future = futures[question]
                result = future.result()
                if result is None:
                    continue
                
                item = {'question': question, 'raw_resp': result}
                qa_scoring_ckpt[question] = item

                with file_lock:
                    try:
                        with open(qa_scoring_ckpt_filename, 'a') as f:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    except Exception as e:
                        print(f"写入检查点文件失败: {e}")

        qa_scoring_dict = {}
        for key, value in qa_scoring_ckpt.items():
            try:
                qa_scoring_dict[key] = self.extract_json(value['raw_resp'])
                if 'score' not in qa_scoring_dict[key]:
                    qa_scoring_dict[key] = {
                        'score': -1, 
                        'reason': f"解析失败，原始响应: {value['raw_resp']}"
                    }
                    raise ValueError(f'结果中没有分数，问题: {key}')
            except Exception as e:
                print(f"{key}, 错误: {e}")
        
        qa_df['score'] = qa_df['question'].apply(
            lambda q: qa_scoring_dict.get(q, {}).get('score', -1)
        )
        qa_df['score_reason'] = qa_df['question'].apply(
            lambda q: qa_scoring_dict.get(q, {}).get('reason', -1)
        )

        hq_qa_df = qa_df[qa_df['score'] >= 4]
        return qa_df, hq_qa_df

def main():
    # 使用示例
    chat = DeepSeekChat()
    
    document = "'，平台系\n统采用T+7模式进行货款\n结算'"
    prompt = QA_GEN_PROMPT_TMPL.replace('{{document}}', document)
    response = chat.chat(prompt)
    print(f"问题: {prompt}")
    print(f"回答: {response}")

if __name__ == "__main__":
    main()