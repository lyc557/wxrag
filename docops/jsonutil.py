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


def parse_json_text(text):
    pattern = r'\[.*\]'

    text = text.replace('>>>', '')
    try:
        return json.loads(text)
    except:
        match = re.search(pattern, text, re.DOTALL)
        try:
            matched = match.group(0)
            return json.loads(matched)
        except Exception as e:
            print(f"{match}, {str(e)}")
            return []

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
    data = '[\n    {\n        "question": "2023年公司采用了哪些公允价值计量的项目？这些项目的期末余额和当期变动分别是多少？",\n        "context": "十一、 采用公允价值计量的项目 \\n√适用 □不适用  \\n单位：元 币种：人民币 \\n项目名称 期初\\n余额 期末余额 当期变动 对当期利润的\\n影响金额 \\n交易性金融资产  400,712,059.93 400,712,059.93 24,072,241.71 \\n其他非流动金融资产  4,002,439,902.57 4,002,439,902.57 2,439,902.57 \\n合计  4,403,151,962.50 4,403,151,962.50 26,512,144.28",\n        "answer": "公司采用了交易性金融资产和其他非流动金融资产两个公允价值计量项目。交易性金融资产的期末余额为400,712,059.93元，当期变动为24,072,241.71元；其他非流动金融资产的期末余额为4,002,439,902.57元，当期变动为2,439,902.57元。"\n    },\n    {\n        "question": "2023年公司公允价值计量项目对当期利润的总影响金额是多少？",\n        "context": "十一、 采用公允价值计量的项目 \\n√适用 □不适用  \\n单位：元 币种：人民币 \\n项目名称 期初\\n余额 期末余额 当期变动 对当期利润的\\n影响金额 \\n交易性金融资产  400,712,059.93 400,712,059.93 24,072,241.71 \\n其他非流动金融资产  4,002,439,902.57 4,002,439,902.57 2,439,902.57 \\n合计  4,403,151,962.50 4,403,151,962.50 26,512,144.28",\n        "answer": "2023年公司公允价值计量项目对当期利润的总影响金额是26,512,144.28元。"\n    },\n    {\n        "question": "2023年公司是如何推动高质量发展的？",\n        "context": "2023年，公司坚持以习近平新时代中国特色社会主义思想为指导，深入学习贯彻党的二十大\\n精神和习近平总书记视察贵州重要讲话精神，主动抢抓国发〔2022〕2 号文件机遇，全面落实省\\n委、省政府决策部署，聚焦集团公司\'双一流、三突破、五跨越\'战略目标，持续走好以茅台美\\n学为价值内涵的\'五线\'高质量发展道路，\'棋心\'拼搏，团结奋斗，圆满完成各项目标任务，\\n推动公司高质量发展取得了重要进展，现代化建设迈出了坚实步伐",\n        "answer": "公司通过坚持以习近平新时代中国特色社会主义思想为指导，深入学习贯彻党的二十大精神和习近平总书记视察贵州重要讲话精神，抢抓国发〔2022〕2号文件机遇，落实省委、省政府决策部署，聚焦\'双一流、三突破、五跨越\'战略目标，走以茅台美学为价值内涵的\'五线\'高质量发展道路，团结奋斗，圆满完成了各项目标任务。"\n    },\n    {\n        "question": "2023年公司发展的指导思想是什么？",\n        "context": "2023年，公司坚持以习近平新时代中国特色社会主义思想为指导，深入学习贯彻党的二十大\\n精神和习近平总书记视察贵州重要讲话精神...",\n        "answer": "2023年公司发展的指导思想是坚持以习近平新时代中国特色社会主义思想为指导，深入学习贯彻党的二十大精神和习近平总书记视察贵州重要讲话精神。"\n    },\n    {\n        "question": "2023年公司取得了哪些主要成就？",\n        "context": "2023年...圆满完成各项目标任务，\\n推动公司高质量发展取得了重要进展，现代化建设迈出了坚实步伐",\n        "answer": "2023年公司圆满完成了各项目标任务，推动高质量发展取得了重要进展，现代化建设迈出了坚实步伐。"\n    }\n]'
    qa_list = parse_json_text(data)
    print(qa_list)