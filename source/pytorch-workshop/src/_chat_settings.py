from typing import Annotated, Any

from dotenv import find_dotenv
from pydantic import BeforeValidator, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src import utils
from src.schema import (
    AllChatModelEnum,
    AnthropicChatModelName,
    FakeChatModelName,
    GoogleChatModelName,
    OpenAIChatModelName,
    Provider,
)

__all__ = ["ChatSettings", "chat_settings"]


class ChatSettings(BaseSettings):
    """ChatSettings for the application."""

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    MODE: str | None = None

    HOST: str = "127.0.0.1"
    PORT: int = 8000

    AUTH_SECRET: SecretStr | None = None

    OPENAI_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    USE_FAKE_CHAT_MODEL: bool = False

    # If DEFAULT_CHAT_MODEL is None, it will be set in chat_model_post_init
    DEFAULT_CHAT_MODEL: AllChatModelEnum | None = None
    AVAILABLE_CHAT_MODELS: set[AllChatModelEnum] = set()

    OPENWEATHERMAP_API_KEY: SecretStr | None = None
    TAVILY_API_KEY: SecretStr | None = None

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[
        str, BeforeValidator(utils.check_str_is_http)
    ] = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: SecretStr | None = None

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Set the default chat model based on the provided API keys.

        Args:
        ----
            __context (Any): The context of the chat_model_post_init method.

        """
        api_keys = {
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            Provider.GOOGLE: self.GOOGLE_API_KEY,
            Provider.FAKE: self.USE_FAKE_CHAT_MODEL,
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        for provider in active_keys:
            match provider:
                case Provider.OPENAI:
                    if self.DEFAULT_CHAT_MODEL is None:
                        self.DEFAULT_CHAT_MODEL = OpenAIChatModelName.GPT_4O_MINI
                    self.AVAILABLE_CHAT_MODELS.update(set(OpenAIChatModelName))
                case Provider.ANTHROPIC:
                    if self.DEFAULT_CHAT_MODEL is None:
                        self.DEFAULT_CHAT_MODEL = AnthropicChatModelName.HAIKU_3
                    self.AVAILABLE_CHAT_MODELS.update(set(AnthropicChatModelName))
                case Provider.GOOGLE:
                    if self.DEFAULT_CHAT_MODEL is None:
                        self.DEFAULT_CHAT_MODEL = GoogleChatModelName.GEMINI_20_FLASH
                    self.AVAILABLE_CHAT_MODELS.update(set(GoogleChatModelName))
                case Provider.FAKE:
                    if self.DEFAULT_CHAT_MODEL is None:
                        self.DEFAULT_CHAT_MODEL = FakeChatModelName.FAKE
                    self.AVAILABLE_CHAT_MODELS.update(set(FakeChatModelName))
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

    @computed_field
    @property
    def BASE_URL(self) -> str:
        """Get the base URL of the application."""
        return f"http://{self.HOST}:{self.PORT}"

    def is_dev(self) -> bool:
        """Check if the application is in development mode."""
        return self.MODE == "dev"


chat_settings = ChatSettings()
