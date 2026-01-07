import asyncio
import contextlib
import weakref
from typing import Iterator, AsyncIterator
from typing import Protocol
from typing import Any

from aihuber.providers.anthropic.anthropic_api import AnthropicApi
from aihuber.providers.cohere.cohere_api import CohereAIApi
from aihuber.providers.hugging_face.hugging_face_api import HuggingFaceApi
from aihuber.providers.mistralai.mistralai_api import MistralAIApi
from aihuber.providers.openai.openai_api import OpenAIApi
from aihuber.providers.perplexity.perplexity_api import PerplexityApi
from aihuber.providers.togetherai.togetherai_api import TogetherAIApi
from aihuber.schema import Message, Response
from proxycraft import ProxyCraft
from proxycraft.config.models import Config
from pydantic import SecretStr, BaseModel
from pathlib import Path

from aihuber.logger import get_logger
import nest_asyncio
import json

nest_asyncio.apply()

logger = get_logger(__name__)

current_dir = Path(__file__).parent
config_path = current_dir / "proxy.json"
with open(config_path, "r") as f:
    CONFIGURATION = json.load(f)


PROVIDER_MAP = {
    "mistral:": MistralAIApi,
    "cohere:": CohereAIApi,
    "togetherai:": TogetherAIApi,
    "perplexity:": PerplexityApi,
    "huggingface:": HuggingFaceApi,
    "anthropic:": AnthropicApi,
    "openai": OpenAIApi,
}


class AudioResponse:
    def __init__(self): ...

    def stream_to_file(self, data):
        print("stream_to_file")


class SpeechStreamingResponse:
    def __init__(self, parent):
        self.parent = parent

    @contextlib.contextmanager
    def create(self, model, voice, input):
        response = AudioResponse()
        yield response


class SpeechAPI:
    def __init__(self, parent):
        self.parent = parent
        self.with_streaming_response = SpeechStreamingResponse(self)


class LLM:
    class AudioAPI:
        def __init__(self, parent):
            self.speech = SpeechAPI(self)

    class EmbeddingsAPI:
        def __init__(self, parent, loop):
            self.parent = parent
            self.loop = loop

        def create(self, model: str, inputs: list[str]) -> dict:
            """Handle synchronous (buffered and streaming) chat completion requests."""
            api = self.parent._get_api()
            if api is None:
                raise ValueError("Model not supported")

            asyncio.set_event_loop(self.loop)
            func = api.embeddings
            return self.loop.run_until_complete(func(model=model, inputs=inputs))

    def __init__(
        self,
        model: str,
        api_key: str | SecretStr | None = None,
        debug: bool = False,
        response_format: type[BaseModel] | None = None,
    ):
        if not any(model.startswith(prefix) for prefix in PROVIDER_MAP):
            allowed_prefixes = ", ".join(f"'{p}'" for p in PROVIDER_MAP.keys())
            raise ValueError(
                f"Invalid model name '{model}'. "
                f"The model must start with one of the following prefixes: {allowed_prefixes}"
            )

        self.model = model
        self.api_key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        self.response_format = response_format

        # Initialize the proxy
        self.proxycraft: ProxyCraft = ProxyCraft(config=Config(**CONFIGURATION))

        self.audio = self.AudioAPI(self)
        self._initialized = False
        try:
            self.loop = asyncio.get_running_loop()
            self.startup_task = asyncio.create_task(self.proxycraft.startup_event())

            logger.debug("We're in an asyncio context")
        except RuntimeError:
            logger.debug("We're NOT in an asyncio context")
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.proxycraft.startup_event())

        self.embeddings = self.EmbeddingsAPI(self, loop=self.loop)
        weakref.finalize(self, self._cleanup_sync, self.proxycraft, self.loop)

    async def close(self):
        """Async cleanup method"""
        await self.proxycraft.shutdown_event()

    def __del__(self):
        """Synchronous cleanup - calls async close() properly"""
        try:
            # Try to get the current running loop
            current_loop = asyncio.get_running_loop()
            # If we're in an async context, schedule the cleanup
            current_loop.create_task(self.close())
        except RuntimeError:
            # No running loop, check if we have our own loop
            if hasattr(self, "loop") and self.loop and not self.loop.is_closed():
                # Use our own loop
                asyncio.set_event_loop(self.loop)
                try:
                    self.loop.run_until_complete(self.close())
                finally:
                    self.loop.close()
            else:
                # Last resort: create new event loop
                try:
                    asyncio.run(self.close())
                except Exception:
                    ...
                    # If all else fails, at least log the issue
                    # print("Warning: Could not properly close async resources")

    @staticmethod
    def _cleanup_sync(proxy: Any, loop: asyncio.AbstractEventLoop):
        """Static cleanup method for weakref.finalize"""
        if loop and not loop.is_closed():
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(proxy.shutdown_event())
                loop.close()
            except Exception as e:
                print(f"Warning: Cleanup failed: {e}")

    class ChatAPI(Protocol):
        """Protocol for chat completion APIs."""

        async def async_chat_completion_buffering(
            self, messages: list[Message]
        ) -> Response: ...

        def sync_chat_completion_streaming(
            self, messages: list[Message]
        ) -> Iterator[str]: ...
        async def async_chat_completion_streaming(
            self, messages: list[Message]
        ) -> Response: ...

    def _get_api(self):
        """Factory to get provider-specific API."""
        for prefix, api_class in PROVIDER_MAP.items():
            if self.model.startswith(prefix):
                return api_class(
                    model=self.model, token=self.api_key, app=self.proxycraft.app
                )
        return OpenAIApi(model=self.model, token=self.api_key, app=self.proxycraft.app)

    def chat_completion(
        self, messages: list[Message], stream: bool = False
    ) -> Response | Iterator[str]:
        """Handle synchronous (buffered and streaming) chat completion requests."""
        api = self._get_api()
        if api is None:
            raise ValueError("Model not supported")

        if stream:
            return api.sync_chat_completion_streaming(messages)
        func = api.async_chat_completion_buffering

        asyncio.set_event_loop(self.loop)
        return self.loop.run_until_complete(func(messages=messages))

    async def ensure_initialized(self):
        if not self._initialized:
            await self.proxycraft.startup_event()
            self._initialized = True

    async def chat_completion_async(
        self, messages: list[Message], stream: bool = False
    ) -> Response | AsyncIterator[str]:
        """Handle asynchronous  (buffered and streaming) chat completion requests."""
        await self.ensure_initialized()

        api = self._get_api()
        if api is None:
            raise ValueError("Model not supported")

        func = (
            api.async_chat_completion_streaming
            if stream
            else api.async_chat_completion_buffering
        )
        return await func(messages=messages)
