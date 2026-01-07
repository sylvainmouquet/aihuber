import asyncio
import json
from abc import ABC, abstractmethod
from typing import Iterator, AsyncIterator

import httpx
from pydantic import BaseModel

from aihuber.logger import get_logger
from aihuber.schema import Message

logger = get_logger(__name__)


class AbstractAPI(ABC):
    def __init__(
        self,
        app,
        completion_url: str,
        embeddings_url: str | None = None,
        completion_method: str = "POST",
        embeddings_method: str = "POST",
        response_format: type[BaseModel] | None = None,
    ):
        self.app = app
        self.completion_url = completion_url
        self.completion_method = completion_method
        self.embeddings_url = embeddings_url
        self.embeddings_method = embeddings_method
        self.response_format = response_format

    @abstractmethod
    def _forge_headers(self, stream: bool): ...

    @abstractmethod
    def _forge_payload(self, messages, stream: bool): ...

    @abstractmethod
    async def _buffered_request(self, messages) -> str | None: ...

    """
    @abstractmethod
    async def _buffered_embeddings(self, model, inputs) -> str | None: ...
    """

    async def _session_client(self, app, method, url, headers, payload, stream):
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aihuber", timeout=60
        ) as client:
            if stream:
                async with client.stream(
                    method=method,
                    url=url,
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    yield response
            else:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=payload,
                )
                try:
                    response.raise_for_status()
                except Exception as e:
                    logger.exception(e)

                    headers = response.headers if hasattr(response, "headers") else {}

                    logger.error(
                        f"Requests remaining: {headers.get('X-RateLimit-Remaining-Requests', 'N/A')}"
                    )
                    logger.error(
                        f"Tokens remaining: {headers.get('X-RateLimit-Remaining-Tokens', 'N/A')}"
                    )
                    logger.error(
                        f"Reset time: {headers.get('X-RateLimit-Reset-Requests', 'N/A')}"
                    )
                    logger.error(
                        f"Reset tokens: {headers.get('X-RateLimit-Reset-Tokens', 'N/A')}"
                    )

                    raise
                yield response

    async def _streaming_request(self, app, method, url, headers, payload, stream):
        async for response in self._session_client(
            app=app,
            method=method,
            url=url,
            headers=headers,
            payload=payload,
            stream=stream,
        ):
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix

                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]

                        # data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ello frien"}}
                        if "type" in chunk and chunk["type"] == "content_block_delta":
                            delta = chunk.get("delta", {})
                            if "text" in delta:
                                yield delta["text"]

                        # COHERE : {'type': 'content-start', 'index': 0, 'delta': {'message': {'content': {'type': 'text', 'text': ''}}}}
                        if "type" in chunk and chunk["type"] == "content-delta":
                            delta = chunk.get("delta", {})
                            if (
                                "message" in delta
                                and "content" in delta["message"]
                                and "text" in delta["message"]["content"]
                            ):
                                yield delta["message"]["content"]["text"]
                    except json.JSONDecodeError:
                        continue

    """
    def sync_chat_completion_buffering(self, messages: list[Message]) -> dict | str:
        try:
            # Try to get current loop
            _loop = asyncio.get_running_loop()

            # If we're here, a loop is running
            # Check if we're in the main thread
            import threading

            if threading.current_thread() is threading.main_thread():
                # Run in thread pool to avoid blocking
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._buffered_request(messages=messages)
                    )
                    return future.result()
            else:
                # We're in a worker thread, safe to create new loop
                return asyncio.run(self._buffered_request(messages=messages))

        except RuntimeError:
            # No loop running, safe to use asyncio.run()
            return asyncio.run(self._buffered_request(messages=messages))
    """

    async def async_chat_completion_buffering(self, messages: list[Message]):
        return await self._buffered_request(messages=messages)

    def sync_chat_completion_streaming(self, messages: list[Message]) -> Iterator:
        def _():
            stream = True
            headers = self._forge_headers(stream)
            payload = self._forge_payload(messages=messages, stream=stream)

            # Run the async generator in a sync context
            async def run_collection():
                async for chunk in self._streaming_request(
                    app=self.app,
                    method=self.completion_method,
                    url=self.completion_url,
                    headers=headers,
                    payload=payload,
                    stream=stream,
                ):
                    yield chunk

            with asyncio.Runner() as runner:
                async_gen = run_collection()
                try:
                    while True:
                        try:
                            yield runner.run(async_gen.__anext__())
                        except StopAsyncIteration:
                            break
                finally:
                    runner.run(async_gen.aclose())

        return _()

    async def async_chat_completion_streaming(
        self, messages: list[Message]
    ) -> AsyncIterator[str]:
        async def _():
            stream = True
            headers = self._forge_headers(stream)
            payload = self._forge_payload(messages=messages, stream=stream)

            # Run the async generator in a sync context
            async for chunk in self._streaming_request(
                app=self.app,
                method=self.completion_method,
                url=self.completion_url,
                headers=headers,
                payload=payload,
                stream=stream,
            ):
                yield chunk

        return _()

    def embeddings(self, model, inputs):
        """Sync wrapper that works in any context."""
        try:
            # Try to get current loop
            _loop = asyncio.get_running_loop()

            # If we're here, a loop is running
            # Check if we're in the main thread
            import threading

            if threading.current_thread() is threading.main_thread():
                # Run in thread pool to avoid blocking
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._buffered_embeddings(model=model, inputs=inputs),
                    )
                    return future.result()
            else:
                # We're in a worker thread, safe to create new loop
                return asyncio.run(
                    self._buffered_embeddings(model=model, inputs=inputs)
                )

        except RuntimeError:
            # No loop running, safe to use asyncio.run()
            return asyncio.run(self._buffered_embeddings(model=model, inputs=inputs))
