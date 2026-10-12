from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src import chat_settings, get_chat_model
from src.agents._api import register_agent

__all__ = ["AgentState", "chatbot"]


class AgentState(MessagesState, total=False):
    """The state of the agent."""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model in a RunnableLambda.

    Args:
    ----
        model (BaseChatModel): The chat model to wrap.

    """
    preprocessor = RunnableLambda(
        lambda state: state["messages"],
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

    return {"messages": [response]}


@register_agent(description="A simple chatbot.")
def chatbot() -> CompiledStateGraph:
    """Get the chatbot agent."""
    agent = StateGraph(AgentState)
    agent.add_node("model", acall_model)
    agent.set_entry_point("model")

    # Always END after blocking unsafe content
    agent.add_edge("model", END)

    return agent.compile(checkpointer=MemorySaver())
