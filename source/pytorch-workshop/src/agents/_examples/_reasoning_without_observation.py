import re
from contextlib import suppress
from typing import Literal, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src import chat_settings, get_chat_model
from src.agents._api import register_agent

__all__ = ["AgentState", "reasoning_without_observation"]


class AgentState(MessagesState, total=False):
    """The state of the agent."""

    task: str
    plan_string: str
    steps: list[str]
    results: dict
    result: str


web_search = None
with suppress(ValueError):
    web_search = TavilySearchResults(max_results=3)

plan_template = """For the following task, make plans that can solve the problem step by step.
For each plan, indicate
which external tool together with tool input to retrieve evidence. You can store the evidence into
a variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

Tools can be one of the following:
(1) Google[input]: Worker that searches results from Google. Useful when you need to find short and
succinct answers about a specific topic. The input should be a search query.
(2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general world
knowledge and common sense. Prioritize it when you are confident in solving the problem yourself.
Input can be any instruction.

For example,
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x hours.
Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours less than Toby.
How many hours did Rebecca work?
Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve with
Wolfram Alpha. #E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Find out the number of hours Thomas worked. #E2 = LLM[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked. #E3 = Calculator[(2 ∗ #E2 − 10) − 8]

Begin!
Describe your plans with rich details. DO NOT add hashtags before the plan, DO NOT number the
plans. Each Plan should be followed by only one #E.

Task: {task}"""

solve_template = """Solve the following task or problem. To solve the problem, we have made
step-by-step Plan and retrieved corresponding Evidence to each Plan. Use them with caution since
long evidence might contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:"""


class Plan(BaseModel):
    """Plan to follow in future."""

    steps: list[str] = Field(description="different steps to follow, should be in sorted order")


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


def wrap_model(
    model: BaseChatModel, scope: Literal["plan", "execute", "solve"]
) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model in a RunnableLambda.

    Args:
    ----
        model (BaseChatModel): The chat model to wrap.
        scope (Literal["plan", "execute", "solve"]): The scope of the agent.

    """
    if web_search is None:
        raise ValueError("To use the `web_search` tool, you must set the TAVILY_API_KEY!")

    if scope == "plan":
        prompt_template = ChatPromptTemplate.from_messages([("user", plan_template)])
        return prompt_template | model
    elif scope == "execute":
        return model
    elif scope == "solve":
        prompt_template = ChatPromptTemplate.from_messages([("user", solve_template)])
        return prompt_template | model
    else:
        raise ValueError("Invalid type for the agent.")


def _get_current_task(state: AgentState) -> int | None:
    """Get the current task.

    Args:
    ----
        state (AgentState): The state of the agent.

    """
    if "results" not in state or state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


async def acall_plan_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the plan model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="plan")

    task = state["messages"][-1].content
    response = await model_runnable.ainvoke({"task": task}, config)

    regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
    matches = re.findall(regex_pattern, response.content)

    return {"steps": matches, "plan_string": response.content, "task": task}


async def acall_execute_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the execute model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="execute")

    step = _get_current_task(state)
    _, step_name, tool, tool_input = state["steps"][step - 1]
    results = (state["results"] or {}) if "results" in state else {}
    for k, v in results.items():
        tool_input = tool_input.replace(k, v)

    if tool == "Google":
        result = await web_search.ainvoke(tool_input, config)
    elif tool == "LLM":
        result = await model_runnable.ainvoke(tool_input, config)
    else:
        raise ValueError(f"Invalid tool: {tool}")

    results[step_name] = str(result)

    return {"results": results}


async def acall_solve_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the solve model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="solve")

    plan = ""
    for p, step_name, tool, tool_input in state["steps"]:
        results = (state["results"] or {}) if "results" in state else {}
        for k, v in results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan: {p}\n{step_name} = {tool}[{tool_input}]"

    response = await model_runnable.ainvoke({"task": state["task"], "plan": plan}, config)
    message = AIMessage(content=response.content)

    return {"messages": [message]}


def router(state: AgentState) -> Literal["execute", "solve"]:
    """Route the agent based on the state.

    Args:
    ----
        state (AgentState): The state of the agent.

    """
    step = _get_current_task(state)
    if step is None:
        return "solve"

    return "execute"


@register_agent(description="A reasoning agent without observations.")
def reasoning_without_observation() -> CompiledStateGraph:
    """Get the reasoning without observation agent."""
    agent = StateGraph(AgentState)
    agent.add_node("planner", acall_plan_model)
    agent.add_node("executer", acall_execute_model)
    agent.add_node("solver", acall_solve_model)
    agent.set_entry_point("planner")

    # Always run "executer" after "planner"
    agent.add_edge("planner", "executer")

    # Always end after "solver"
    agent.add_edge("solver", END)

    # After "executer", route to "executer" if there are more steps to execute
    # Otherwise, route to "solver"
    agent.add_conditional_edges("executer", router, {"execute": "executer", "solve": "solver"})

    return agent.compile()
