from collections.abc import Callable

from pydantic import BaseModel, Field

__all__ = [
    "AgentInfo",
    "ModelInfo",
]


class AgentInfo(BaseModel):
    """Info about an available agent."""

    name: str = Field(
        description="Agent key.",
        examples=["research-assistant"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["A research assistant for generating research papers."],
    )
    builder_fn: Callable | None = Field(
        description="Function to build the agent.",
        examples=["src.agents._research_assistant.research_assistant"],
        exclude=True,
        default=None,
    )


class ModelInfo(BaseModel):
    """Info about an available model."""

    name: str = Field(
        description="Model key.",
        examples=["image-captioning", "image-tagging"],
    )
    builder_fn: Callable | None = Field(
        description="Function to build a model.",
        examples=["src.models._clip.clip"],
        exclude=True,
        default=None,
    )
