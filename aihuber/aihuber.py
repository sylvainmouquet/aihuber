import asyncio
import contextlib
import socket
import threading
import time
import weakref
from typing import Iterator, AsyncIterator
from typing import Protocol
from typing import Any

import httpx

from aihuber.providers.abstract_api import AbstractAPI
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

from aihuber.logger import get_logger
import nest_asyncio

nest_asyncio.apply()

logger = get_logger(__name__)

CONFIGURATION = {
    "version": "1.0",
    "name": "AIHuber",
    "server": {"type": "local"},
    "endpoints": [
        {
            "identifier": "/proxy/openai",
            "prefix": "/proxy/openai",
            "match": "/proxy/openai/**",
            "backends": [
                {
                    "https": [
                        {
                            "id": "openai-api",
                            "url": "https://api.openai.com",
                            "ssl": True,
                            "methods": ["POST"],
                        }
                    ]
                }
            ],
            "upstream": {"proxy": {"enabled": True}},
        },
        {
            "identifier": "/proxy/mistral",
            "prefix": "/proxy/mistral",
            "match": "/proxy/mistral/**",
            "backends": [
                {
                    "https": [
                        {
                            "id": "mistral-api",
                            "url": "https://api.mistral.ai",
                            "ssl": True,
                            "methods": ["POST"],
                        }
                    ]
                }
            ],
            "upstream": {"proxy": {"enabled": True}},
        },
        {
            "identifier": "/proxy/anthropic",
            "prefix": "/proxy/anthropic",
            "match": "/proxy/anthropic/**",
            "backends": [
                {
                    "https": [
                        {
                            "id": "anthropic-api",
                            "url": "https://api.anthropic.com",
                            "ssl": True,
                            "methods": ["POST"],
                        }
                    ]
                }
            ],
            "upstream": {"proxy": {"enabled": True}},
        },
        {
            "identifier": "/proxy/cohere",
            "prefix": "/proxy/cohere",
            "match": "/proxy/cohere/**",
            "backends": [
                {
                    "https": [
                        {
                            "id": "cohere-api",
                            "url": "https://api.cohere.com",
                            "ssl": True,
                            "methods": ["POST"],
                        }
                    ]
                }
            ],
            "upstream": {"proxy": {"enabled": True}},
        },
        {
            "identifier": "/proxy/huggingface",
            "prefix": "/proxy/huggingface",
            "match": "/proxy/huggingface/**",
            "backends": [
                {
                    "https": [
                        {
                            "id": "huggingface-api",
                            "url": "https://router.huggingface.co/novita/v3",
                            "ssl": True,
                            "methods": ["POST"],
                        }
                    ]
                }
            ],
            "upstream": {"proxy": {"enabled": True}},
        },
        {
            "identifier": "/proxy/anthropic",
            "prefix": "/proxy/anthropic",
            "match": "/proxy/anthropic/**",
            "backends": [
                {
                    "https": [
                        {
                            "id": "anthropic-api",
                            "url": "https://api.anthropic.com",
                            "ssl": True,
                            "methods": ["POST"],
                        }
                    ]
                }
            ],
            "upstream": {"proxy": {"enabled": True}},
        },
        {
            "identifier": "/proxy/perplexity",
            "prefix": "/proxy/perplexity",
            "match": "/proxy/perplexity/**",
            "backends": [
                {
                    "https": [
                        {
                            "id": "perplexity-api",
                            "url": "https://api.perplexity.ai",
                            "ssl": True,
                            "methods": ["POST"],
                        }
                    ]
                }
            ],
            "upstream": {"proxy": {"enabled": True}},
        },
        {
            "identifier": "/proxy/together",
            "prefix": "/proxy/together",
            "match": "/proxy/together/**",
            "backends": [
                {
                    "https": [
                        {
                            "id": "together-api",
                            "url": "https://api.together.xyz",
                            "ssl": True,
                            "methods": ["POST"],
                        }
                    ]
                }
            ],
            "upstream": {"proxy": {"enabled": True}},
        },
    ],
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
        def __init__(self, parent):
            self.parent = parent

        def create(self, model: str, inputs: list[str]) -> dict:
            """Handle synchronous (buffered and streaming) chat completion requests."""
            api = self.parent._get_api()
            if api is None:
                raise ValueError("Model not supported")

            func = api.embeddings
            return func(model=model, inputs=inputs)

    def __init__(
        self,
        model: str,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        debug: bool = False,
        response_format: type[BaseModel] | None = None,
    ):
        self.model = model
        self.api_key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        self.base_url = base_url
        self.response_format = response_format

        # Initialize the proxy
        self.proxycraft: ProxyCraft = ProxyCraft(config=Config(**CONFIGURATION))

        # Start proxy in a separate daemon thread
        self.proxy_thread = threading.Thread(
            target=self.proxycraft.serve, daemon=True, name="ProxyCraftThread"
        )

        self.embeddings = self.EmbeddingsAPI(self)
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

        self.transport = httpx.ASGITransport(app=self.proxycraft.app)
        weakref.finalize(self, self._cleanup_sync, self.proxycraft, self.loop)

        def wait_port_available(host: str, port: int):
            def _socket_test_connection():
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)  # Add timeout to avoid blocking indefinitely
                    result = s.connect_ex((host, port))
                    s.close()
                    return result == 0  # 0 means connection successful
                except Exception:
                    return False

            while _socket_test_connection():
                logger.info(f"waiting for port {port}")
                time.sleep(1)

        def check_port_available(host: str, port: int):
            def _socket_test_connection():
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)  # Add timeout to avoid blocking indefinitely
                    result = s.connect_ex((host, port))
                    s.close()
                    return result == 1
                except Exception:
                    return True

            while _socket_test_connection():
                logger.info(f"port {port} is not available")
                time.sleep(1)

        check_port_available(host="0.0.0.0", port=8080)
        self.proxy_thread.start()
        wait_port_available(host="0.0.0.0", port=8080)

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
        if hasattr(self, "proxy_thread"):
            self.proxy_thread.join(timeout=1)
            # del self.proxycraft

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

    def _get_api(self) -> AbstractAPI | None:
        """Get the appropriate API instance for the current model."""

        if self.model.startswith("mistral:"):
            return MistralAIApi(
                model=self.model, token=self.api_key, app=self.proxycraft.app
            )
        elif self.model.startswith("cohere:"):
            return CohereAIApi(
                model=self.model, token=self.api_key, app=self.proxycraft.app
            )
        elif self.model.startswith("togetherai:"):
            return TogetherAIApi(
                model=self.model, token=self.api_key, app=self.proxycraft.app
            )
        elif self.model.startswith("perplexity:"):
            return PerplexityApi(
                model=self.model, token=self.api_key, app=self.proxycraft.app
            )
        elif self.model.startswith("huggingface:"):
            return HuggingFaceApi(
                model=self.model, token=self.api_key, app=self.proxycraft.app
            )
        elif self.model.startswith("anthropic:"):
            return AnthropicApi(
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

        # if self._initialized:
        #    asyncio.set_event_loop(self.loop)
        #    return self.loop.run_until_complete(func(messages=messages))
        # else:
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
