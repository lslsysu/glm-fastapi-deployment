import requests
import json

url = "http://localhost:8000/v1/chat/completions"

def get_completion(prompt, temperature=0, top_p=1.0, max_length=8192):
    request = {
                'model': 'chatglm2-6b',
                'messages': [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                'max_length': max_length,
                'stream': False
            }
    response = requests.post(url, json=request).json()
    return response["choices"][0]["message"]['content']

def get_completion_stream(prompt, temperature=0, top_p=1.0, max_length=8192):
    request = {
                'model': 'chatglm2-6b',
                'messages': [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                'max_length': max_length,
                'stream': True
            }
    response = requests.post(url, json=request, stream=True)
    result = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line.strip():
            continue
        # 去掉冗余前缀 "data: data: 或data: "
        if line.startswith("data: data: "):
            line = line[len("data: data: "):]
        elif line.startswith("data:"):
            line = line[len("data:"):].strip()

        if line == "[DONE]":
            break

        try:
            data = json.loads(line)
            delta = data["choices"][0]["delta"]
            content = delta.get("content")
            if content:
                yield content
        except json.JSONDecodeError:
            continue

if __name__ == "__main__":
    # get completion
    text = get_completion("你好")
    print(text)

    # stream get completion
    for chunk in get_completion_stream("你好"):
        print(chunk, end="", flush=True)
