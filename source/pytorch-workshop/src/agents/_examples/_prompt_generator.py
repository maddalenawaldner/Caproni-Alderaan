from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from src import chat_settings, get_chat_model
from src.agents._api import register_agent

__all__ = ["AgentState", "prompt_generator"]


class AgentState(MessagesState, total=False):
    """The state of the agent."""


class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""

    objective: str
    variables: list[str]
    constraints: list[str]
    requirements: list[str]


research_instructions = """Your job is to get information from a user about what type of prompt
template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool.
"""

template_instructions = """Write a prompt template for a language model. Use the following
requirements to guide you:

{reqs}
"""


def wrap_model(
    model: BaseChatModel, scope: Literal["research", "template"]
) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model in a RunnableLambda.

    Args:
    ----
        model (BaseChatModel): The chat model to wrap.
        scope (Literal["research", "template"]): The scope of the agent.

    """
    if scope == "research":
        model = model.bind_tools([PromptInstructions])
        preprocessor = RunnableLambda(
            lambda state: [SystemMessage(content=research_instructions)] + state["messages"],
            name="StateModifier",
        )

        return preprocessor | model
    elif scope == "template":
        preprocessor = RunnableLambda(
            lambda state: [SystemMessage(content=template_instructions)] + state["messages"],
            name="StateModifier",
        )
        return preprocessor | model
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

    return {"messages": [response]}


async def acall_template_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the template model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="template")

    # get the messages related to the prompt generation
    tool_call = None
    other_msgs = []
    for m in state["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)

    messages = {"messages": other_msgs}
    response = await model_runnable.ainvoke(messages, config)

    return {"messages": [response]}


def router(state: AgentState) -> Literal["tools", "info", "done"]:
    """Route the agent based on the state.

    Args:
    ----
        state (AgentState): The state of the agent.

    """
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "tools"
    elif not isinstance(messages[-1], HumanMessage):
        return "done"
    return "info"


@register_agent(description="A prompt generator agent.")
def prompt_generator() -> CompiledStateGraph:
    """Get the prompt generator agent."""
    agent = StateGraph(AgentState)
    agent.add_node("researcher", acall_research_model)
    agent.add_node("template_generator", acall_template_model)
    agent.set_entry_point("researcher")

    @agent.add_node
    async def add_tool_message(state: AgentState, config: RunnableConfig) -> AgentState | dict:
        """Add a tool message to the state.

        Args:
        ----
            state (AgentState): The state of the agent.
            config (RunnableConfig): The configuration for the agent.

        """
        return {
            "messages": [
                ToolMessage(
                    content="Prompt generated!",
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                )
            ]
        }

    # Always run "template_generator" after "add_tool_message"
    agent.add_edge("add_tool_message", "template_generator")

    # After "researcher", if the last message is an AIMessage with tool calls, run
    # "add_tool_message". If the last message is not a HumanMessage, END.
    # Otherwise, run "researcher".
    agent.add_conditional_edges(
        "researcher", router, {"tools": "add_tool_message", "info": "researcher", "done": END}
    )

    # Always end after "template_generator"
    agent.add_edge("template_generator", END)

    return agent.compile(checkpointer=MemorySaver())
