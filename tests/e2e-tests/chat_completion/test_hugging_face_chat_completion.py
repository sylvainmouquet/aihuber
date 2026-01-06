"""
import pytest
from dotenv import load_dotenv
import os
from aihuber import LLM
from aihuber.providers.hugging_face.hugging_face_models import HuggingFaceModels
from aihuber.aihuber import Message
from pydantic import SecretStr

load_dotenv()

HF_TOKEN = SecretStr(os.getenv("HF_TOKEN"))  # type: ignore


@pytest.fixture
def llm_instance():
    return LLM(
        model=HuggingFaceModels.LLAMA_2_7B_CHAT,
        api_key=HF_TOKEN.get_secret_value(),
    )


@pytest.fixture
def test_messages():
    return [
        Message(role="system", content="You are a game player"),
        Message(
            role="user",
            content="Ping. Return Pong in JSON format with the json key message",
        ),
    ]


def test_hugging_face_chat_completion_sync(llm_instance, test_messages):
    from transformers import pipeline

    pipe = pipeline(
        "text-generation", model="deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True
    )
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    pipe(messages)
"""
