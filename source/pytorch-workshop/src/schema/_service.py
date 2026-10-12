from pydantic import BaseModel, Field

from src.schema._base import AgentInfo, ModelInfo
from src.schema._chat_models import AllChatModelEnum

__all__ = ["ServiceMetadata"]


class ServiceMetadata(BaseModel):
    """Metadata about the service including available agents and models."""

    agents: list[AgentInfo] = Field(
        description="List of available agents.",
    )
    chat_models: list[AllChatModelEnum] = Field(
        description="List of available chat models.",
    )
    models: list[ModelInfo] = Field(
        description="List of available models.",
    )

    default_agent: str = Field(
        description="Default agent used when none is specified.",
        examples=["research-assistant"],
    )
    default_chat_model: AllChatModelEnum = Field(
        description="Default model used when none is specified.",
    )
