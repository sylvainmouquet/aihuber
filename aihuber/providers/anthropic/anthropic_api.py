from pydantic import SecretStr
import json

from aihuber.providers.abstract_api import AbstractAPI
from aihuber.logger import get_logger

logger = get_logger(__name__)


class AnthropicApi(AbstractAPI):
    def __init__(self, model, token: SecretStr, app):
        super().__init__(app=app, completion_url="/proxy/anthropic/v1/messages")
        self.token = token
        self.model = model.replace("anthropic:", "")

    def _forge_headers(self, stream: bool):
        # docs: https://docs.anthropic.com/en/api/overview
        return {
            "x-api-key": f"{self.token.get_secret_value()}",
            "Anthropic-Version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "application/json" if not stream else "text/event-stream",
        }

    def _forge_payload(self, messages, stream: bool):
        formatted_messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]

        return {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": 1024,
            "stream": stream,
        }

    async def _buffered_request(self, messages) -> str | None:
        stream = False
        headers = self._forge_headers(stream=stream)
        payload = self._forge_payload(messages=messages, stream=stream)

        async for response in self._session_client(
            app=self.app,
            method=self.completion_method,
            url=self.completion_url,
            headers=headers,
            payload=payload,
            stream=stream,
        ):
            resp_json = response.json()

            try:
                raw_text = resp_json["content"][0]["text"]
                try:
                    return json.loads(raw_text)
                except json.JSONDecodeError:
                    return raw_text
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected Anthropic response structure: {resp_json}")
                raise ValueError(f"Failed to parse Anthropic response: {e}")

        raise ValueError("No response received from session client")
