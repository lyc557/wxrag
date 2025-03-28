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


def parse_json_text(data):
    result = []
    # 使用正则提取 JSON 部分
    match = re.search(r'\[\[(.*)\]\]', data, re.DOTALL)
    if match:
        json_str = '[' + match.group(1).strip() + ']'
        # 解析 JSON
        try:
            result = json.loads(json_str)
            print(json.dumps(result, indent=4, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print("JSON 解析错误:", e)
    else:
        print("未找到 JSON 数据")
    return result

def build_qa_df(qa_ckpt, uuid2doc_map):
    data = []
    for key, value in tqdm(qa_ckpt.items()):
        text = value['raw_resp']
        qa_list = parse_json_text(text)
        logger.debug(f"处理 {key}，提取到 {len(qa_list)} 个 QA 对")

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


if __name__ == "__main__":
    data = ''''[]\n\n[\n    {\n        "question": "公司董事会、监事会及董事、监事、高级管理人员对年度报告内容承担什么责任？",\n        "context": "本公司董事会、监事会及董事、监事、高级管理人员保证年度报告内容的真实 性、准确性、完整性，不存在虚假记载、误导性陈述或重大遗漏，并承担个别和连带的法律责任。",\n        "answer": "他们保证年度报告内容的真实性、准确性、完整性，不存在虚假记载、误导性陈述或重大遗漏，并承担个别和连带的法律责任。"\n    },\n    {\n        "question": "公司全体董事出席了哪次会议？",\n        "context": "公司全体董事出席董事会会议。",\n        "answer": "公司全体董事出席了董事会会议。"\n    },\n    {\n        "question": "谁为公司出具了审计报告？",\n        "context": "天职国际会计师事务所（特殊普通合伙）为本公司出具了标准无保留意见的审计报告。",\n        "answer": "天职国际会计师事务所（特殊普通合伙）为公司出具了标准无保留意见的审计报告。"\n    },\n    {\n        "question": "公司负责人、主管会计工作负责人及会计机构负责人需要保证年度报告中财务报告的什么？",\n        "context": "公司负责人丁雄军、主管会计工作负责人蒋焰及会计机构负责人（会计主管人员）蔡聪应声明：保证年度报告中财务报告的真实、准确、完整。",\n        "answer": "他们需要保证年度报告中财务报告的真实、准确、完整。"\n    },\n    {\n        "question": "公司计划如何进行2023年度的利润分配？",\n        "context": "公司以实施权益分派股权登记日公司总股本为基数实施2023年度利润分配，向全体股东每10股派发现金红利308.76元（含税）。截至2023年12月31日，公司总股本为125,619.78万股，以此计算合计拟派发现金红利38,786,363,272.80元（含税）。在实施权益分派的股权登记日前公司总股本如发生变动的，将维持分红总额不变，相应调整每股分红比例。以上利润分配预案需提交公司股东大会审议通过后实施。",\n        "answer": "公司计划以实施权益分派股权登记日公司总股本为基数，向全体股东每10股派发现金红利308.76元（含税）。截至2023年12月31日，公司总股本为125,619.78万股，以此计算合计拟派发现金红利38,786,363,272.80元（含税）。在实施权益分派的股权登记日前公司总股本如发生变动的，将维持分红总额不变，相应调整每股分红比例。以上利润分配预案需提交公司股东大会审议通过后实施。"\n    }\n]'''
    qa_list = parse_json_text(data)
    print(qa_list)