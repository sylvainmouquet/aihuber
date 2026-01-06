import pytest
from dotenv import load_dotenv
import os
from aihuber import LLM
from aihuber.aihuber import Message
from pydantic import SecretStr

load_dotenv()

ANTHROPIC_TOKEN = SecretStr(os.getenv("ANTHROPIC_TOKEN"))  # type: ignore


@pytest.fixture
def llm_instance():
    return LLM(
        model="anthropic:claude-sonnet-4-20250514",
        api_key=ANTHROPIC_TOKEN.get_secret_value(),
    )


@pytest.fixture
def test_messages(stream: bool = False):
    return [
        Message(
            role="user",
            content="Ping. Return Pong in JSON object with the json key message. Not markdown",
        ),
    ]


def test_anthropic_chat_completion_sync(llm_instance, test_messages):
    """Test streaming chat completion with Anthropic."""
    response = llm_instance.chat_completion(test_messages)
    assert response == {"message": "Pong"}, (
        f"Expected {{'message': 'Pong'}}, got {response}"
    )


def test_anthropic_chat_completion_sync_streaming(llm_instance, test_messages):
    """Test streaming chat completion with Anthropic."""
    response = llm_instance.chat_completion(test_messages, stream=True)

    chunks = []
    for chunk in response:
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    assert len(chunks) >= 2


@pytest.mark.asyncio
async def test_anthropic_chat_completion_async(llm_instance, test_messages):
    """Test asynchronous chat completion with Anthropic."""
    response = await llm_instance.chat_completion_async(test_messages)
    assert response == {"message": "Pong"}


@pytest.mark.asyncio
async def test_anthropic_chat_completion_async_streaming(llm_instance, test_messages):
    """Test asynchronous chat completion with Anthropic."""
    response = await llm_instance.chat_completion_async(test_messages, stream=True)

    chunks = []
    async for chunk in response:
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    assert len(chunks) >= 2
