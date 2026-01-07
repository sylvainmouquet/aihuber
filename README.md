# AIHuber

__aihuber__ is the easiest and quickest way to communicate with AI providers. 
Free and open-source.


<!-- Project Status Badges -->
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Version](https://img.shields.io/badge/version-1.0.0-green.svg)

<!-- Technology Badges 
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Asyncio](https://img.shields.io/badge/Asyncio-FFD43B?style=flat&logo=python&logoColor=blue)
-->



## ✨ Features

__aihuber__ is compatible with main AI providers :

### Chat completion (Text generation)

##### AI Providers Comparison

| AI Provider | Buffering sync & async | Streaming (sync & async) | Endpoint                                                                                                                     |
|-------------|------------------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------|
| OpenAI | ✅ | ✅ | `https://api.openai.com/v1/chat/completions`                                                                                 |
| Mistral AI | ✅ | ✅ | `https://api.mistral.ai/v1/chat/completions`                                                                                 |
| Anthropic | ✅ | ✅ | `https://api.anthropic.com/v1/messages`                                                                                      |
| Google (Gemini) | ❌ | ❌ | `https://generativelanguage.googleapis.com/v1/models/{model}:generateContent`                                                |
| Cohere | ✅ | ✅ | `https://api.cohere.com/v2/chat`                                                                                             |
| Together AI | ✅ | ✅ | `https://api.together.xyz/v1/chat/completions`                                                                               |
| Replicate | ❌ | ❌ | `https://api.replicate.com/v1/predictions`                                                                                   |
| Hugging Face | ❌ | ❌ | `https://api-inference.huggingface.co/models/{model_id}`                                                                     |
| Azure OpenAI | ❌ | ❌ | `https://{your-resource-name}.openai.azure.com/openai/deployments/{deployment-name}/chat/completions?api-version=2024-02-01` |
| AWS Bedrock | ❌ | ❌ | `https://bedrock-runtime.{region}.amazonaws.com/model/{model-id}/invoke`                                                     |
| Perplexity | ✅ | ✅ | `https://api.perplexity.ai/chat/completions`                                                                                 |
| xAI (Grok) | ❌ | ❌ | `https://api.x.ai/v1/chat/completions`                                                                                       |

## Legend

- ✅ = Available
- ❌ = Not Available
- Text in parentheses = Specific service/model name

## 🚀 Quick Start

## Installation

```bash
pip install aihuber
```

Or with uv:

```bash
uv add aihuber
```

### Basic Usage

```python
import asyncio
import os

from dotenv import load_dotenv

from aihuber import LLM, Message
from pydantic import SecretStr

load_dotenv()

MISTRAL_AI_TOKEN = SecretStr(os.getenv("MISTRAL_AI_TOKEN"))  # type: ignore


async def main():
    """Chat completion example with asynchronous"""
    llm = LLM(model="mistral:mistral-small-latest", api_key=MISTRAL_AI_TOKEN)

    messages = [
        Message(role="system", content="You are a helpful Python developer assistant."),
        Message(role="user", content="Give me 10 numbers from 0"),
    ]

    async_chat_completion = await llm.chat_completion_async(messages, stream=True)
    async for chunk in async_chat_completion:
        print(chunk)

if __name__ == "__main__":
    asyncio.run(main())
```

## 🐳 Docker Usage


```bash
docker build -t aihuber -f dockerfiles/aihuber.Dockerfile .
docker run  -p 8080:8080 aihuber
```

## 📄 License

[MIT](LICENSE)