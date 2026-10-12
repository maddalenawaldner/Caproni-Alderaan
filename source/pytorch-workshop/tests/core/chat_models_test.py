import os
from unittest.mock import patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import FakeListChatModel
from langchain_openai import ChatOpenAI

from src import get_chat_model
from src.schema import (
    AnthropicChatModelName,
    FakeChatModelName,
    OpenAIChatModelName,
)


def test_get_chat_model_openai() -> None:
    """Test getting an OpenAI model."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        model = get_chat_model(OpenAIChatModelName.GPT_4O_MINI)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-4o-mini"
        assert model.temperature == 0.5
        assert model.streaming is True


def test_get_chat_model_anthropic() -> None:
    """Test getting an Anthropic model."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
        model = get_chat_model(AnthropicChatModelName.HAIKU_35)
        assert isinstance(model, ChatAnthropic)
        assert model.model == "claude-3-5-haiku-latest"
        assert model.temperature == 0.5
        assert model.streaming is True


def test_get_chat_model_fake() -> None:
    """Test getting a fake model."""
    model = get_chat_model(FakeChatModelName.FAKE)
    assert isinstance(model, FakeListChatModel)
    assert model.responses == ["This is a test response from the fake model."]


def test_get_chat_model_invalid() -> None:
    """Test getting an invalid model."""
    with pytest.raises(ValueError, match="Unsupported model:"):
        # Using type: ignore since we're intentionally testing invalid input
        get_chat_model("invalid_model")  # type: ignore
