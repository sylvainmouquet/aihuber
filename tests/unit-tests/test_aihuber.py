from aihuber import LLM
from aihuber.aihuber import PROVIDER_MAP
import pytest


def test_llm_init_with_valid_prefix():
    valid_prefix = list(PROVIDER_MAP.keys())[0]
    model_name = f"{valid_prefix}large"

    llm = LLM(model=model_name)

    assert llm.model == model_name


def test_llm_init_with_invalid_prefix():
    invalid_model = "fake:model-name"

    with pytest.raises(ValueError) as excinfo:
        LLM(model=invalid_model)

    assert "Invalid model name" in str(excinfo.value)
    assert list(PROVIDER_MAP.keys())[0] in str(excinfo.value)


def test_llm_chat_completion_logic(mocker):
    mocker.patch(
        "aihuber.providers.mistralai.mistralai_api.MistralAIApi._buffered_request",
        return_value="Success",
    )

    llm = LLM(model="mistral:large")
    messages = [{"role": "user", "content": "Hello"}]

    response = llm.chat_completion(messages=messages)
    assert response == "Success"
