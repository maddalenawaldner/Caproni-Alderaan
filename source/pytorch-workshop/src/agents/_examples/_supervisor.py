from contextlib import suppress
from typing import Literal, TypedDict

from langchain_community.tools import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from src import chat_settings, get_chat_model
from src.agents._api import register_agent
from src.agents.tools import python_repl

__all__ = ["AgentState", "supervisor"]


class AgentState(MessagesState, total=False):
    """The state of the agent."""

    next: str


web_search = None
with suppress(ValueError):
    web_search = TavilySearchResults(name="WebSearch", max_results=5)

members = ["web_searcher", "coder"]
options = members + ["FINISH"]

instructions = f"""You are a supervisor tasked with managing a conversation between the following
workers: {members}. Given the following user request, respond with the worker to act next. Each
worker will perform a task and respond with their results and status. When finished,respond with
FINISH.
"""


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal["web_searcher", "coder", "FINISH"]


def wrap_model(
    model: BaseChatModel, scope: Literal["supervise", "web_search", "code"]
) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model in a RunnableLambda.

    Args:
    ----
        model (BaseChatModel): The chat model to wrap.
        scope (Literal["supervise", "web_searcher", "code"]): The scope of the agent.

    """
    if web_search is None:
        raise ValueError("To use the `web_search` tool, you must set the TAVILY_API_KEY!")

    if scope == "supervise":
        model = model.with_structured_output(Router)
        preprocessor = RunnableLambda(
            lambda state: [SystemMessage(content=instructions)] + state["messages"],
            name="StateModifier",
        )

        return preprocessor | model
    elif scope == "web_search":
        web_search_instructions = "You are a web browser. DO NOT do any math."
        model = create_react_agent(
            model, tools=[web_search], state_modifier=web_search_instructions
        )
        return model
    elif scope == "code":
        model = create_react_agent(model, tools=[python_repl])
        return model


async def acall_supervise_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the supervise model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="supervise")
    response = await model_runnable.ainvoke(state, config)

    next_ = response["next"]
    if next_ == "FINISH":
        next_ = END

    return {"next": next_}


async def acall_web_search_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the web search model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="web_search")

    response = await model_runnable.ainvoke(state, config)
    message = AIMessage(content=response["messages"][-1].content, name="web_searcher")

    return {"messages": [message]}


async def acall_code_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the code model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="code")

    response = await model_runnable.ainvoke(state, config)
    message = AIMessage(content=response["messages"][-1].content, name="coder")

    return {"messages": [message]}


@register_agent(description="A supervisor agent supervising others.")
def supervisor() -> CompiledStateGraph:
    """Get the supervisor agent."""
    agent = StateGraph(AgentState)
    agent.add_node("supervisor", acall_supervise_model)
    agent.add_node("web_searcher", acall_web_search_model)
    agent.add_node("coder", acall_code_model)
    agent.set_entry_point("supervisor")

    # Connect all the members to the supervisor
    for member in members:
        agent.add_edge(member, "supervisor")

    agent.add_conditional_edges("supervisor", lambda state: state["next"])

    return agent.compile()
