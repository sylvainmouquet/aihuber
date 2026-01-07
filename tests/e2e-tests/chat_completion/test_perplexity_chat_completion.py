import pytest
from dotenv import load_dotenv
import os
from aihuber import LLM
from aihuber.aihuber import Message
from pydantic import SecretStr


load_dotenv()

PERPLEXITY_TOKEN = SecretStr(os.getenv("PERPLEXITY_TOKEN"))  # type: ignore


@pytest.fixture
def llm_instance():
    return LLM(
        model="perplexity:sonar",
        api_key=PERPLEXITY_TOKEN.get_secret_value(),
    )


@pytest.fixture
def test_messages(stream: bool = False):
    return [
        Message(
            role="system",
            content=(
                "You must respond with raw JSON only. "
                "Do not use Markdown. "
                "Do not wrap the response in ``` fences. "
                "Do not include explanations or comments. "
                "Return a single valid JSON object."
            ),
        ),
        Message(
            role="user",
            content="Ping. Return Pong in JSON format with the json key message. Only 11 characters. Dont shrink pong",
        ),
    ]


def test_perplexity_chat_completion_sync(llm_instance, test_messages):
    """Test streaming chat completion with Perplexity."""
    response = llm_instance.chat_completion(test_messages)
    assert {"message": "Pong"} == response


def test_perplexity_chat_completion_sync_streaming(llm_instance, test_messages):
    """Test streaming chat completion with Perplexity."""
    response = llm_instance.chat_completion(test_messages, stream=True)

    chunks = []
    for chunk in response:
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    assert len(chunks) > 4


@pytest.mark.asyncio
async def test_perplexity_chat_completion_async(llm_instance, test_messages):
    """Test asynchronous chat completion with Perplexity."""
    response = await llm_instance.chat_completion_async(test_messages)
    assert {"message": "Pong"} == response


@pytest.mark.asyncio
async def test_perplexity_chat_completion_async_streaming(llm_instance, test_messages):
    """Test asynchronous chat completion with Perplexity."""
    response = await llm_instance.chat_completion_async(test_messages, stream=True)

    chunks = []
    async for chunk in response:
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    assert len(chunks) > 4
