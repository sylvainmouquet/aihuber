import json

from pydantic import SecretStr

from aihuber.providers.abstract_api import AbstractAPI


class GrokAIApi(AbstractAPI):
    def __init__(self, model, token: SecretStr, app):
        super().__init__(
            app=app,
            completion_url="/proxy/grok/v1/chat/completions"
        )
        self.token = token
        self.model = model.replace("grok:", "")

    def _forge_headers(self, stream: bool):
        return {
            "Authorization": f"Bearer {self.token.get_secret_value()}",
            "Accept": "application/json" if not stream else "text/event-stream",
            "Content-Type": "application/json",
        }

    def _forge_payload(self, messages, stream: bool):
        return {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "response_format": {"type": "json_object"},
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
            content = json.loads(resp_json["choices"][0]["message"]["content"])
            return content

        raise ValueError("No response received from session client")
