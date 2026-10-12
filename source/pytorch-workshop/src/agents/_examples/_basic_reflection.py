from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src import chat_settings, get_chat_model
from src.agents._api import register_agent

__all__ = ["AgentState", "basic_reflection"]

max_iterations: int = 6


class AgentState(MessagesState, total=False):
    """The state of the agent."""


generate_instructions = """You are an essay assistant tasked with writing excellent 5-paragraph
essays. Generate the best essay possible for the user's request. If the user provides critique,
respond with a revised version of your previous attempts.
"""

reflect_instructions = """You are a teacher grading an essay submission. Generate critique and
recommendations for the user's submission. Provide detailed recommendations, including requests for
length, depth, style, etc.
"""


def wrap_model(
    model: BaseChatModel, scope: Literal["generate", "reflect"]
) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model in a RunnableLambda.

    Args:
    ----
        model (BaseChatModel): The chat model to wrap.
        scope (Literal["generate", "reflect"]): The scope of the agent.

    """
    if scope == "generate":
        prompt = ChatPromptTemplate.from_messages(
            [("system", generate_instructions), MessagesPlaceholder(variable_name="messages")],
        )
        return prompt | model
    elif scope == "reflect":
        prompt = ChatPromptTemplate.from_messages(
            [("system", reflect_instructions), MessagesPlaceholder(variable_name="messages")]
        )
        return prompt | model
    else:
        raise ValueError("Invalid type for the agent.")


async def acall_generate_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the generate model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="generate")
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [response]}


async def acall_reflect_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the reflect model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="reflect")
    response = await model_runnable.ainvoke(state, config)

    messages = state["messages"]
    messages += [HumanMessage(content=response.content)]

    return {"messages": messages}


def router(state: AgentState) -> Literal["reflect", "done"]:
    """Route the agent based on the state.

    Args:
    ----
        state (dict): The current graph state

    """
    if len(state["messages"]) > max_iterations:
        return "done"
    return "reflect"


@register_agent(description="A basic reflection agent.")
def basic_reflection() -> CompiledStateGraph:
    """Get the basic reflection agent."""
    agent = StateGraph(AgentState)
    agent.add_node("generate", acall_generate_model)
    agent.add_node("reflect", acall_reflect_model)
    agent.set_entry_point("generate")

    # Always run "generate" after "reflect"
    agent.add_edge("reflect", "generate")

    # Reflect until we have enough messages
    agent.add_conditional_edges("generate", router, {"reflect": "reflect", "done": END})

    return agent.compile(checkpointer=MemorySaver())
