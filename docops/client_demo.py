import requests
import json

def chat_with_model(prompt, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", temperature=None):
    """
    与本地运行的大模型API服务进行对话
    """
    url = "http://localhost:8888/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "test-api-key"  # 替换为你的API密钥
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    # 只有当temperature不为None时才添加到请求中
    if temperature is not None:
        data["temperature"] = temperature
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"错误: {response.status_code}, {response.text}"

def main():
    print("欢迎使用本地大模型聊天程序！输入'退出'结束对话。")
    
    while True:
        user_input = input("\n你: ")
        
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("再见！")
            break
        
        print("\n正在思考...")
        response = chat_with_model(user_input)
        print(f"\n大模型: {response}")

if __name__ == "__main__":
    main()