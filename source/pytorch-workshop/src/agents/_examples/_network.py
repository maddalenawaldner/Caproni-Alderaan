from contextlib import suppress
from typing import Literal

from langchain_community.tools import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from src import chat_settings, get_chat_model
from src.agents._api import register_agent
from src.agents.tools import python_repl

__all__ = ["AgentState", "network"]


class AgentState(MessagesState, total=False):
    """The state of the agent."""


web_search = None
with suppress(ValueError):
    web_search = TavilySearchResults(name="WebSearch", max_results=5)


model_template = """You are a helpful AI assistant, collaborating with other assistants.
Use the provided tools to progress towards answering the question. If you are unable to fully
answer, that's OK, another assistant with different tools will help where you left off. Execute
what you can to make progress. If you or any of the other assistants have the final answer or
deliverable, prefix your response with FINAL ANSWER so the team knows to stop.

{suffix}
"""


def wrap_model(
    model: BaseChatModel, scope: Literal["research", "chart"]
) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model in a RunnableLambda.

    Args:
    ----
        model (BaseChatModel): The chat model to wrap.
        scope (Literal["research", "chart"]): The scope of the agent.

    """
    if web_search is None:
        raise ValueError("To use the `web_search` tool, you must set the TAVILY_API_KEY!")

    if scope == "research":
        suffix = "You can only do research. You are working with a chart generator colleague."
        research_instructions = model_template.format(suffix=suffix)
        model = create_react_agent(model, tools=[web_search], state_modifier=research_instructions)
        return model
    elif scope == "chart":
        suffix = "You can only generate charts. You are working with a researcher colleague."
        chart_instructions = model_template.format(suffix=suffix)
        model = create_react_agent(model, tools=[python_repl], state_modifier=chart_instructions)
        return model
    else:
        raise ValueError("Invalid type for the agent.")


async def acall_research_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the research model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="research")

    response = await model_runnable.ainvoke(state, config)
    message = AIMessage(content=response["messages"][-1].content, name="researcher")

    return {"messages": [message]}


async def acall_chart_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the chart model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="chart")

    response = await model_runnable.ainvoke(state, config)
    message = AIMessage(content=response["messages"][-1].content, name="chart_generator")

    return {"messages": [message]}


def router(state: AgentState) -> Literal["continue", "done"]:
    """Route the agent based on the state.

    Args:
    ----
        state (AgentState): The state of the agent.

    """
    last_message = state["messages"][-1]
    if "FINAL ANSWER" in last_message.content:
        return "done"
    return "continue"


@register_agent(description="A network of agents.")
def network() -> CompiledStateGraph:
    """Get the network agent."""
    agent = StateGraph(AgentState)
    agent.add_node("researcher", acall_research_model)
    agent.add_node("chart_generator", acall_chart_model)
    agent.set_entry_point("researcher")

    # After "researcher"/"chart_generator", route to the other node if needed. Otherwise END.
    agent.add_conditional_edges("researcher", router, {"continue": "chart_generator", "done": END})
    agent.add_conditional_edges("chart_generator", router, {"continue": "researcher", "done": END})

    return agent.compile()
