from typing import Callable

from langgraph.graph.state import CompiledStateGraph

from src.schema import AgentInfo

__all__ = [
    "AGENTS",
    "DEFAULT_AGENT",
    "get_agent",
    "get_agent_builder",
    "get_agent_info",
    "get_agents_info",
    "register_agent",
]

AGENTS: dict[str, AgentInfo] = {}
DEFAULT_AGENT = "chatbot"


def get_agent(agent_id: str, **agent_kwargs) -> CompiledStateGraph:
    """Get an agent.

    Args:
    ----
        agent_id (str): The name of the agent.
        **agent_kwargs (dict): Keyword arguments to pass to the agent builder.

    """
    return AGENTS[agent_id].builder_fn(**agent_kwargs)


def get_agent_builder(agent_id: str) -> Callable:
    """Get an agent builder.

    Args:
    ----
        agent_id (str): The name of the agent.

    """
    return AGENTS[agent_id].builder_fn


def get_agent_info(agent_id: str) -> AgentInfo:
    """Get the agent info.

    Args:
    ----
        agent_id (str): The name of the agent.

    """
    return AGENTS[agent_id]


def get_agents_info() -> list[AgentInfo]:
    """Get all agents info."""
    return list(AGENTS.values())


def register_agent(name: str | None = None, description: str | None = None) -> CompiledStateGraph:
    """Register an agent.

    Args:
    ----
        name (str, optional): The name of the agent.
        description (str): The description of the agent.

    """

    def decorator(agent: Callable) -> Callable:
        AGENTS[name or agent.__name__.lower()] = AgentInfo(
            name=name or agent.__name__.lower(),
            description=description or agent.__doc__,
            builder_fn=agent,
        )
        return agent

    return decorator
