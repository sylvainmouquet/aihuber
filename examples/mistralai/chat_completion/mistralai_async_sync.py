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

    response = await llm.chat_completion_async(messages)
    print(f"Response: {response}")

    response = llm.chat_completion(messages)
    print(f"Response: {response}")

    response = await llm.chat_completion_async(messages)
    print(f"Response: {response}")

    response = llm.chat_completion(messages)
    print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
