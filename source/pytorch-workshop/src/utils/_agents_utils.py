from typing import Any

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import ChatMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from pydantic import BaseModel, Field

__all__ = ["CustomData"]


class CustomData(BaseModel):
    """A custom data class for dispatching custom events."""

    type: str = Field(
        description="The type of custom data, used in dispatch events",
        default="custom_data",
    )
    data: dict[str, Any] = Field(description="The custom data")

    def to_langchain(self) -> ChatMessage:
        """Convert the custom data to a LangChain message."""
        return ChatMessage(content=[self.data], role="custom")

    async def adispatch(self, config: RunnableConfig | None = None) -> None:
        """Dispatch the custom data as an event.

        Args:
        ----
            config (RunnableConfig, optional): The configuration for the dispatch event.

        """
        dispatch_config = RunnableConfig(
            tags=["custom_data_dispatch"],
        )
        await adispatch_custom_event(
            name=self.type,
            data=self.to_langchain(),
            config=merge_configs(config, dispatch_config),
        )
