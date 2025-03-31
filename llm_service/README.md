### 本地部署一个大模型
`llm_deploy.py` 是一个用于本地部署大语言模型的脚本。主要功能如下：

- 使用vLLM框架本地部署一个LLM大模型
- 提供HTTP服务接口，方便其他应用程序调用
- 内存管理优化，高效利用GPU资源

使用方法：
运行脚本 ```python llm_deploy.py```, 本地开启一个由FastAPI部署的服务
服务IP: 为local(本地IP)
端口: 8777

**注意**
使用这个方案部署的大模型有一个问题: 如果大模型正在做上一个请求的inference时候，无法处理新来的请求。（需要更换vllm的使用方案，使用OpenAI-Compatible server的方式。该方式的部署方案可以参考```llm_open_ai.sh```）

### URL请求
在运行 ```python llm_deploy.py``` 之后, 模型服务已经部署好, 可以通过下面的方法得到大模型的输出。

```
prompts = [
    {"role": "system", "content": systemp_prompt},
    {"role": "user", "content": user_prompt},
]
response = requests.post(
    url="http://{local}:8777/generate",
    data=json.dumps(
        {
            "prompts": prompts,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 512,
        }
    ),
    timeout=60,
).json()
```

