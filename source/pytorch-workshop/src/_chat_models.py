from functools import cache
from typing import TypeAlias

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.schema import (
    AllChatModelEnum,
    AnthropicChatModelName,
    FakeChatModelName,
    GoogleChatModelName,
    OpenAIChatModelName,
)

__all__ = ["get_chat_model"]

_MODEL_TABLE = {
    OpenAIChatModelName.GPT_4O_MINI: "gpt-4o-mini",
    OpenAIChatModelName.GPT_4O: "gpt-4o",
    AnthropicChatModelName.HAIKU_35: "claude-3-5-haiku-latest",
    AnthropicChatModelName.SONNET_35: "claude-3-5-sonnet-latest",
    GoogleChatModelName.GEMINI_15_PRO: "gemini-1.5-pro",
    GoogleChatModelName.GEMINI_20_FLASH: "gemini-2.0-flash",
    FakeChatModelName.FAKE: "fake",
}

ModelT: TypeAlias = ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI


@cache
def get_chat_model(model_name: AllChatModelEnum, /) -> ModelT:
    """Get the chat model based on the model name.

    Args:
    ----
        model_name (AllChatModelEnum): The chat model name to get the chat model for.

    """
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIChatModelName:
        return ChatOpenAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in AnthropicChatModelName:
        return ChatAnthropic(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GoogleChatModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in FakeChatModelName:
        return FakeListChatModel(responses=["This is a test response from the fake model."])
