import json
from typing import Any

from pydantic import SecretStr
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import core_schema

from aihuber.providers.abstract_api import AbstractAPI

import re


class OpenAISchemaGenerator(GenerateJsonSchema):
    """
    Custom schema generator that inherits from GenerateJsonSchema
    and formats schemas for OpenAI API structured outputs.

    Usage:
        schema = MyModel.model_json_schema(schema_generator=OpenAISchemaGenerator)
    """

    def generate(
        self, schema: core_schema.CoreSchema, mode: str = "validation"
    ) -> dict[str, Any]:
        """
        Override the main generate method to wrap the final schema in OpenAI format.

        Args:
            schema: The core schema from Pydantic
            mode: The generation mode ('validation' or 'serialization')

        Returns:
            OpenAI-formatted schema dict with required name and schema wrapper
        """
        # Call parent to generate the standard JSON schema
        json_schema = super().generate(schema, mode)

        # Extract model name from title or generate one
        schema_name = self._generate_name_from_schema(json_schema)

        # Wrap in OpenAI format with required structure
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": json_schema},
        }

    def _generate_name_from_schema(self, schema: dict[str, Any]) -> str:
        """Generate a schema name from the schema title."""
        title = schema.get("title", "GeneratedSchema")
        # Convert CamelCase to snake_case
        snake_case = re.sub("([a-z0-9])([A-Z])", r"\1_\2", title).lower()
        return snake_case


class OpenAIApi(AbstractAPI):
    def __init__(self, model, token: SecretStr, app, response_format=None):
        super().__init__(
            app=app,
            completion_url="/proxy/openai/v1/chat/completions",
            embeddings_url="/proxy/openai/v1/embeddings",
            response_format=response_format,
        )
        self.token = token
        self.model = model.replace("openai:", "")

    def _forge_headers(self, stream: bool):
        return {
            "Authorization": f"Bearer {self.token.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json" if not stream else "text/event-stream",
        }

    def _forge_payload(self, messages, stream: bool):
        # Add a system message to ensure JSON response
        messages_with_json_instruction = [
            {
                "role": "developer",
                "content": "Return a JSON object. Don't use simple quote '",
            }
        ] + [{"role": message.role, "content": message.content} for message in messages]

        payload = {
            "model": self.model,
            "messages": messages_with_json_instruction,
            "stream": stream,
        }

        if self.response_format is not None:
            json_schema = self.response_format.model_json_schema(
                schema_generator=OpenAISchemaGenerator
            )

            payload["response_format"] = json_schema
        elif self.model in [
            "gpt-4-1106-preview",
            "gpt-3.5-turbo-1106",
            "gpt-4o",
            "gpt-4.1-nano",
        ]:
            # Add response_format only if the model supports it
            payload["response_format"] = {"type": "json_object"}

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
            content = resp_json["choices"][0]["message"]["content"]

            try:
                return json.loads(content)
            except json.decoder.JSONDecodeError:
                return content

        raise ValueError("No response received from session client")

    async def _buffered_embeddings(self, model, inputs) -> str | None:
        """docs: https://platform.openai.com/docs/api-reference/embeddings"""
        stream = False

        headers = self._forge_headers(stream=stream)

        async for response in self._session_client(
            app=self.app,
            method=self.embeddings_method,
            url=self.embeddings_url,
            headers=headers,
            payload={"model": model, "input": inputs, "encoding_format": "float"},
            stream=stream,
        ):
            resp_json = response.json()
            return resp_json["data"][0]["embedding"]

        raise ValueError("No response received from session client")
