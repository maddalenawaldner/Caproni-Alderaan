import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from src import ChatSettings
from src.schema import AnthropicChatModelName, OpenAIChatModelName
from src.utils import check_str_is_http


def testcheck_str_is_http() -> None:
    """Test the check_str_is_http function."""
    # Test valid HTTP URLs
    assert check_str_is_http("http://example.com/") == "http://example.com/"
    assert check_str_is_http("https://api.test.com/") == "https://api.test.com/"

    # Test invalid URLs
    with pytest.raises(ValidationError):
        check_str_is_http("not_a_url")
    with pytest.raises(ValidationError):
        check_str_is_http("ftp://invalid.com")


def test_chat_settings_default_values() -> None:
    """Test the default values of the chat settings."""
    chat_settings = ChatSettings(_env_file=None)
    assert chat_settings.HOST == "127.0.0.1"
    assert chat_settings.PORT == 8000
    assert chat_settings.USE_FAKE_CHAT_MODEL is False


def test_chat_settings_no_api_keys() -> None:
    """Test that chat settings raises error when no API keys are provided."""
    with patch.dict(os.environ, {}, clear=True):  # noqa: SIM117
        with pytest.raises(ValueError, match="At least one LLM API key must be provided"):
            _ = ChatSettings(_env_file=None)


def test_chat_settings_with_openai_key() -> None:
    """Test chat settings with an OpenAI key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        chat_settings = ChatSettings(_env_file=None)
        assert SecretStr("test_key") == chat_settings.OPENAI_API_KEY
        assert chat_settings.DEFAULT_CHAT_MODEL == OpenAIChatModelName.GPT_4O_MINI
        assert set(OpenAIChatModelName) == chat_settings.AVAILABLE_CHAT_MODELS


def test_chat_settings_with_anthropic_key() -> None:
    """Test chat settings with an Anthropic key."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True):
        chat_settings = ChatSettings(_env_file=None)
        assert SecretStr("test_key") == chat_settings.ANTHROPIC_API_KEY
        assert chat_settings.DEFAULT_CHAT_MODEL == AnthropicChatModelName.HAIKU_3
        assert set(AnthropicChatModelName) == chat_settings.AVAILABLE_CHAT_MODELS


def test_chat_settings_with_multiple_api_keys() -> None:
    """Test chat settings with multiple API keys."""
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_openai_key",
            "ANTHROPIC_API_KEY": "test_anthropic_key",
        },
        clear=True,
    ):
        chat_settings = ChatSettings(_env_file=None)
        assert SecretStr("test_openai_key") == chat_settings.OPENAI_API_KEY
        assert SecretStr("test_anthropic_key") == chat_settings.ANTHROPIC_API_KEY
        # When multiple providers are available, OpenAI should be the default
        assert chat_settings.DEFAULT_CHAT_MODEL == OpenAIChatModelName.GPT_4O_MINI
        # Available models should include exactly all OpenAI and Anthropic models
        expected_models = set(OpenAIChatModelName)
        expected_models.update(set(AnthropicChatModelName))
        assert expected_models == chat_settings.AVAILABLE_CHAT_MODELS


def test_chat_settings_base_url() -> None:
    """Test the base URL of the chat settings."""
    chat_settings = ChatSettings(HOST="localhost", PORT=8000, _env_file=None)
    assert chat_settings.BASE_URL == "http://localhost:8000"


def test_settings_is_dev() -> None:
    """Test the is_dev method of the chat settings."""
    chat_settings = ChatSettings(MODE="dev", _env_file=None)
    assert chat_settings.is_dev() is True

    chat_settings = ChatSettings(MODE="prod", _env_file=None)
    assert chat_settings.is_dev() is False
