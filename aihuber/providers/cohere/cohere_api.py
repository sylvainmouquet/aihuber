import json

from pydantic import SecretStr

from aihuber.providers.abstract_api import AbstractAPI


class CohereAIApi(AbstractAPI):
    def __init__(self, model, token: SecretStr, app):
        super().__init__(app=app, completion_url="/proxy/cohere/v2/chat")
        self.token = token
        self.model = model.replace("cohere:", "")

    def _forge_headers(self, stream: bool):
        return {
            "Authorization": f"Bearer {self.token.get_secret_value()}",
            "Accept": "application/json" if not stream else "text/event-stream",
            "Content-Type": "application/json",
        }

    def _forge_payload(self, messages, stream: bool):
        formatted_messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "response_format": {"type": "json_object"},
            "stream": stream,
        }

        return payload

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
            content = resp_json["message"]["content"][0]["text"]

            try:
                return json.loads(content)
            except json.decoder.JSONDecodeError:
                return content

        raise ValueError("No response received from session client")
