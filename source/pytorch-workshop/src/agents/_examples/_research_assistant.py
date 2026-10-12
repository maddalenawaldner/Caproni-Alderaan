from contextlib import suppress
from datetime import datetime
from typing import Literal

from langchain_community.tools import OpenWeatherMapQueryRun, TavilySearchResults
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from src import chat_settings, get_chat_model
from src.agents._api import register_agent
from src.agents.tools import calculator

__all__ = ["AgentState", "research_assistant"]


class AgentState(MessagesState, total=False):
    """The state of the agent."""

    remaining_steps: RemainingSteps


web_search = None
with suppress(ValueError):
    web_search = TavilySearchResults(name="WebSearch", max_results=5)
tools = [web_search, calculator]

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if chat_settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=chat_settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""You are a helpful research assistant with the ability to search the web and use
other tools.
Today's date is {current_date}.

NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

A few things to remember:
- Please include markdown-formatted links to any citations used in your response. Only include
one or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
- Use calculator tool with numexpr to answer math questions. The user does not understand
numexpr, so for the final response, use human readable format - e.g. "300 * 200", not
"(300 \\times 200)".
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model in a RunnableLambda.

    Args:
    ----
        model (BaseChatModel): The chat model to wrap.

    """
    if web_search is None:
        raise ValueError("To use the `web_search` tool, you must set the TAVILY_API_KEY!")

    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )

    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }

    return {"messages": [response]}


def router(state: AgentState) -> Literal["tools", "done"]:
    """Route the agent based on the state.

    Args:
    ----
        state (AgentState): The state of the agent.

    """
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


@register_agent(description="A research assistant with web search and calculator.")
def research_assistant() -> CompiledStateGraph:
    """Get the research assistant agent."""
    if web_search is None:
        raise ValueError("To use the `web_search` tool, you must set the TAVILY_API_KEY!")

    agent = StateGraph(AgentState)
    agent.add_node("model", acall_model)
    agent.add_node("tools", ToolNode(tools))
    agent.set_entry_point("model")

    # Always run "model" after "tools"
    agent.add_edge("tools", "model")

    # After "model", if there are tool calls, run "tools". Otherwise END.
    agent.add_conditional_edges("model", router, {"tools": "tools", "done": END})

    return agent.compile(checkpointer=MemorySaver())
