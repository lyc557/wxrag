import re
import pandas as pd

def convert2json(text):
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
        qa_list = convert2json(text)

        for item in qa_list:
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            context = item.get('context', '').strip()

            if question == '' or answer == '':
                print(qa_list)
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