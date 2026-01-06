import pytest
from dotenv import load_dotenv
import os
from aihuber import LLM
from aihuber.aihuber import Message
from pydantic import SecretStr


load_dotenv()

COHERE_TOKEN = SecretStr(os.getenv("COHERE_TOKEN"))  # type: ignore


@pytest.fixture
def llm_instance():
    return LLM(
        model="cohere:command-a-03-2025",
        api_key=COHERE_TOKEN.get_secret_value(),
    )


@pytest.fixture
def test_messages(stream: bool = False):
    return [
        Message(role="system", content="You are a game player"),
        Message(
            role="user",
            content="Ping. Return Pong in JSON format with the json key message",
        ),
    ]


def test_cohere_chat_completion_sync(llm_instance, test_messages):
    """Test streaming chat completion with Cohere."""
    response = llm_instance.chat_completion(test_messages)
    assert response == {"message": "Pong"}, (
        f"Expected {{'message': 'Pong'}}, got {response}"
    )


def test_cohere_chat_completion_sync_streaming(llm_instance, test_messages):
    """Test streaming chat completion with Cohere."""
    response = llm_instance.chat_completion(test_messages, stream=True)

    chunks = []
    for chunk in response:
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    assert len(chunks) == 6
    assert chunks == ['{"', "message", '":', ' "', "Pong", '"}']


@pytest.mark.asyncio
async def test_cohere_chat_completion_async(llm_instance, test_messages):
    """Test asynchronous chat completion with Cohere."""
    response = await llm_instance.chat_completion_async(test_messages)
    assert response == {"message": "Pong"}


@pytest.mark.asyncio
async def test_cohere_chat_completion_async_streaming(llm_instance, test_messages):
    """Test asynchronous chat completion with Cohere."""
    response = await llm_instance.chat_completion_async(test_messages, stream=True)

    chunks = []
    async for chunk in response:
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    assert len(chunks) == 6
    assert chunks == ['{"', "message", '":', ' "', "Pong", '"}']
