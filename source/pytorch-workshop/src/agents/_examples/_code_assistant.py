from typing import Literal

from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src import chat_settings, get_chat_model
from src.agents._api import register_agent

__all__ = ["AgentState", "code_assistant"]

max_iterations: int = 3
flag: Literal["reflect", "do not reflect"] = "do not reflect"


class AgentState(MessagesState, total=False):
    """The state of the agent."""

    error: str
    generation: str
    iterations: int


class Code(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


# Load the LCEL documentation
url = "https://python.langchain.com/docs/concepts/lcel/"
loader = RecursiveUrlLoader(url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text)
docs = loader.load()

# Sort the list based on the URLs and get the text
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join([doc.page_content for doc in d_reversed])

instructions = f"""You are a coding assistant with expertise in LCEL, LangChain expression
language. Here is a full set of LCEL documentation:
-------
{concatenated_content}
-------
Answer the user question based on the above provided documentation. Ensure any code you provide
can be executed with all required imports and variables defined. Structure your answer with a
description of the code solution. Then list the imports. And finally list the functioning code
block. Here is the user question:
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model in a RunnableLambda.

    Args:
    ----
        model (BaseChatModel): The chat model to wrap.

    """
    model = model.with_structured_output(Code, include_raw=True)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    postprocessor = RunnableLambda(
        lambda output: output["parsed"],
        name="OutputModifier",
    )

    return preprocessor | model | postprocessor


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m)

    messages = state["messages"]
    iterations = state.get("iterations", 0)
    error = state.get("error", "")

    if error == "yes":
        state["messages"] += [
            HumanMessage(
                "Now, try again. Invoke the code tool to structure the output with a prefix,"
                " imports, and code block:",
            )
        ]

    response = await model_runnable.ainvoke(state, config)
    ai_message = f"{response.prefix}\nImports: {response.imports}\nCode: {response.code}"
    messages += [AIMessage(content=ai_message)]

    iterations = iterations + 1

    return {"generation": response, "messages": messages, "iterations": iterations}


async def acheck_code(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Check the code for errors.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    messages = state["messages"]
    generation = state["generation"]
    iterations = state["iterations"]

    imports = generation.imports
    code = generation.code

    # Check imports
    try:
        exec(imports)  # noqa: S102 # nosec B102
    except ImportError as e:
        messages += [HumanMessage(f"Your solution failed the import test: {e}")]
        return {
            "generation": generation,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # Check execution
    try:
        exec(imports + "\n\n" + code)  # noqa: S102 # nosec B102
    except Exception as e:
        messages += [HumanMessage(f"Your solution failed the code execution test: {e}")]
        return {
            "generation": generation,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    return {
        "generation": generation,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }


async def areflect(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Reflect on the code solution.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m)

    messages = state["messages"]
    iterations = state["iterations"]
    generation = state["generation"]

    reflections = await model_runnable.ainvoke(state, config)
    messages += [AIMessage(content=f"Here are reflections on the error: {reflections}")]

    return {"generation": generation, "messages": messages, "iterations": iterations}


# After "check_code", if there is an error, run "reflect". Otherwise, decide to finish.
def router(state: AgentState) -> Literal["reflect", "model", "done"]:
    """Route the agent based on the state.

    Args:
    ----
        state (dict): The current graph state

    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        return "done"
    else:
        if flag == "reflect":
            return "reflect"
        else:
            return "model"


@register_agent(description="A code assistant agent.")
def code_assistant() -> CompiledStateGraph:
    """Get the code assistant agent."""
    agent = StateGraph(AgentState)
    agent.add_node("model", acall_model)
    agent.add_node("check_code", acheck_code)
    agent.add_node("reflect", areflect)
    agent.set_entry_point("model")

    # Always run "check_code" after "model"
    agent.add_edge("model", "check_code")

    agent.add_conditional_edges(
        "check_code",
        router,
        {"reflect": "reflect", "model": "model", "done": END},
    )

    # Always run "model" after "reflect"
    agent.add_edge("reflect", "model")

    return agent.compile(checkpointer=MemorySaver())
