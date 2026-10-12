from contextlib import suppress
from datetime import datetime
from typing import Any, Literal

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.tools import StructuredTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError

from src import chat_settings, get_chat_model
from src.agents._api import register_agent

__all__ = ["AgentState", "reflexion"]

max_iterations: int = 5


class AgentState(MessagesState, total=False):
    """The state of the agent."""


generate_template = """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.
"""

reflect_template = """Reflect on the user's original question and the actions taken thus far.
Respond using the {function_name} function.
"""

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the
        word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and
    make SURE it is not more than 250 words.
"""

web_search = None
with suppress(ValueError):
    web_search = TavilySearchResults(name="WebSearch", max_results=5)


class Reflection(BaseModel):
    """Reflection on the initial answer."""

    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question.

    It provides an answer, reflection, and then follow up with search queries to improve the
    answer.
    """

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description=(
            "1-3 search queries for researching improvements to address the critique of your"
            " current answer."
        )
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question.

    It provide an answer, reflection,  cite your reflection with references, and finally add search
    queries to improve the answer.
    """

    references: list[str] = Field(description="Citations motivating your updated answer.")


class RunnableWithRetries:
    """Runnable with validation and retries."""

    def __init__(self, runnable: RunnableSerializable, validator: RunnableSerializable) -> None:
        self.runnable = runnable
        self.validator = validator

    async def ainvoke(self, state: AgentState, config: RunnableConfig | None = None) -> Any:  # noqa: ANN401
        """Call ainvoke from a thread and validate the response with retries.

        Args:
        ----
            state (AgentState): The state of the agent.
            config (RunnableConfig): The configuration for the agent.

        """
        response = []
        for attempt in range(3):
            response = await self.runnable.ainvoke(
                {"messages": state["messages"]}, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                await self.validator.ainvoke(response)
                return response
            except ValidationError as e:
                state["messages"] += [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        return response


def run_queries(search_queries: list[str], **kwargs) -> list[dict]:
    """Run the generated queries.

    Args:
    ----
        search_queries (list[str]): The search queries
        kwargs: Additional keyword arguments

    """
    return web_search.batch([{"query": query} for query in search_queries])


def wrap_model(model: BaseChatModel, scope: Literal["draft", "revise"]) -> RunnableWithRetries:
    """Wrap the model in a  RunnableLambda.

    Args:
    ----
        model (BaseChatModel): The chat model to wrap.
        scope (Literal["draft", "revise"]): The scope of the agent.

    """
    if web_search is None:
        raise ValueError("To use the `web_search` tool, you must set the TAVILY_API_KEY!")

    if scope == "draft":
        prompt_messages = [
            ("system", generate_template),
            MessagesPlaceholder(variable_name="messages"),
            ("user", reflect_template),
        ]
        prompt = ChatPromptTemplate.from_messages(prompt_messages).partial(
            time=lambda: datetime.now().isoformat(),
            first_instruction="Provide a detailed ~250 word answer.",
            function_name=AnswerQuestion.__name__,
        )

        model = model.bind_tools(tools=[AnswerQuestion])
        validator = PydanticToolsParser(tools=[AnswerQuestion])

        return RunnableWithRetries(runnable=prompt | model, validator=validator)
    elif scope == "revise":
        prompt_messages = [
            ("system", generate_template),
            MessagesPlaceholder(variable_name="messages"),
            ("user", reflect_template),
        ]
        prompt = ChatPromptTemplate.from_messages(prompt_messages).partial(
            time=lambda: datetime.now().isoformat(),
            first_instruction="Provide a detailed ~250 word answer.",
            function_name=AnswerQuestion.__name__,
        )

        model = model.bind_tools(tools=[ReviseAnswer])
        validator = PydanticToolsParser(tools=[ReviseAnswer])

        return RunnableWithRetries(runnable=prompt | model, validator=validator)
    else:
        raise ValueError("Invalid type for the agent.")


async def acall_draft_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the draft model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="draft")
    responses = await model_runnable.ainvoke(state, config)

    return {"messages": responses}


async def acall_revise_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the revise model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="revise")
    responses = await model_runnable.ainvoke(state, config)

    return {"messages": responses}


tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)


def router(state: AgentState) -> Literal["execute_tools", "done"]:
    """Route the agent based on the state.

    Args:
    ----
        state (AgentState): The current state of the agent.

    """

    def _get_num_iterations(messages: list) -> int:
        """Get the number of iterations.

        Args:
        ----
            messages (list): The current state of the agent.

        """
        i = 0
        for m in messages[::-1]:
            if m.type not in {"tool", "ai"}:
                break
            i += 1
        return i

    num_iterations = _get_num_iterations(state["messages"])
    if num_iterations > max_iterations:
        return "done"
    return "execute_tools"


@register_agent(description="A reflexion agent.")
def reflexion() -> CompiledStateGraph:
    """Get the reflexion agent."""
    agent = StateGraph(AgentState)
    agent.add_node("draft", acall_draft_model)
    agent.add_node("execute_tools", tool_node)
    agent.add_node("revise", acall_revise_model)
    agent.set_entry_point("draft")

    # Always run "execute_tools" after "draft"
    agent.add_edge("draft", "execute_tools")

    # Always run "revise" after "execute_tools"
    agent.add_edge("execute_tools", "revise")

    # Execute tools until we have at least num_iterations iterations
    agent.add_conditional_edges("revise", router, {"execute_tools": "execute_tools", "done": END})

    return agent.compile()
