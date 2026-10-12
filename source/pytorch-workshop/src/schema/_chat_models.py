from enum import StrEnum, auto
from typing import TypeAlias

__all__ = [
    "Provider",
    "AnthropicChatModelName",
    "GoogleChatModelName",
    "OpenAIChatModelName",
    "FakeChatModelName",
    "AllChatModelEnum",
]


class Provider(StrEnum):
    """Enum of model providers."""

    OPENAI = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    FAKE = auto()


class AnthropicChatModelName(StrEnum):
    """Enum of Anthropic chat models.

    References
    ----------
        https://docs.anthropic.com/en/docs/about-claude/models#model-names

    """

    HAIKU_3 = "claude-3-haiku"
    HAIKU_35 = "claude-3.5-haiku"
    SONNET_35 = "claude-3.5-sonnet"


class GoogleChatModelName(StrEnum):
    """Enum of Google chat models.

    References
    ----------
        https://ai.google.dev/gemini-api/docs/models/gemini

    """

    GEMINI_15_PRO = "gemini-1.5-pro"
    GEMINI_20_FLASH = "gemini-2.0-flash"


class OpenAIChatModelName(StrEnum):
    """Enum of OpenAI chat models.

    References
    ----------
        https://platform.openai.com/docs/models/gpt-4o

    """

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"


class FakeChatModelName(StrEnum):
    """Fake chat model for testing."""

    FAKE = "fake"


AllChatModelEnum: TypeAlias = (
    AnthropicChatModelName | GoogleChatModelName | OpenAIChatModelName | FakeChatModelName
)
