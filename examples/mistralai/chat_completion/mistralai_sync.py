import os

from dotenv import load_dotenv

from aihuber import LLM, Message

load_dotenv()

if __name__ == "__main__":
    """Chat completion example"""
    llm = LLM(
        model="mistral:mistral-small-latest", api_key=os.getenv("MISTRAL_AI_TOKEN")
    )

    messages = [
        Message(role="system", content="You are a helpful Python developer assistant."),
        Message(role="user", content="Give me 10 numbers from 0"),
    ]

    response = llm.chat_completion(messages)
    print(f"Response: {response}")

    response = llm.chat_completion(messages)
    print(f"Response: {response}")

    response = llm.chat_completion(messages)
    print(f"Response: {response}")
