import asyncio
import os
from dotenv import load_dotenv
from aihuber import LLM, Message
from pydantic import SecretStr

load_dotenv()

MISTRAL_AI_TOKEN = SecretStr(os.getenv("MISTRAL_AI_TOKEN"))  # type: ignore


async def main():
    llm = LLM(
        model="mistral:mistral-small-latest",
        api_key=MISTRAL_AI_TOKEN.get_secret_value(),
    )

    messages = [
        Message(role="system", content="You are a helpful Python developer assistant."),
        Message(
            role="user",
            content="Give me 10 numbers from 0 to 100 as a comma-separated list in ascending order.",
        ),
    ]

    response = await llm.chat_completion_async(messages)
    print(f"Response 1: {response}")

    response = await llm.chat_completion_async(messages)
    print(f"Response 2: {response}")

    await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
