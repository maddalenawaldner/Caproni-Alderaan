import operator
from contextlib import suppress
from typing import Annotated, Literal, Union

from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src import chat_settings, get_chat_model
from src.agents._api import register_agent

__all__ = ["AgentState", "plan_and_execute"]


class AgentState(MessagesState, total=False):
    """The state of the agent."""

    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple], operator.add]
    response: str


web_search = None
with suppress(ValueError):
    web_search = TavilySearchResults(name="WebSearch", max_results=3)

plan_instructions = """For the given objective, come up with a simple step by step plan. This
plan should involve individual tasks, that if executed correctly will yield the correct answer. Do
not add any superfluous steps. The result of the final step should be the final answer. Make sure
that each step has all the information needed - do not skip steps.
"""

replan_instructions = """**Objective**:
{input}

**Original Plan**:
{plan}

**Completed Steps**:
{past_steps}

**Instructions**:
- If there are more steps needed to achieve the objective, return a **Plan** with the remaining
steps.
- If all necessary steps have been completed, return a **Response** to the user based on the
information gathered.
- Do **not** include any steps that have already been completed in the new plan.
- Do **not** return an empty plan; if no further steps are needed, you **must** return a
**Response**.
- Ensure your output is in the correct structured format as per the `Act` model.

**Remember**:
- The `Act` can be either a `Plan` or a `Response`.
- A `Plan` contains a list of steps that still need to be done.
- A `Response` contains the final answer to the user.

**Provide your output below:**
"""


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
    model: BaseChatModel, scope: Literal["answer", "execute", "plan", "replan"]
) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model in a RunnableLambda.

    Args:
    ----
        model (BaseChatModel): The chat model to wrap.
        scope (Literal["answer", "execute", "plan", "replan"]): The scope of the agent.

    """
    if web_search is None:
        raise ValueError("To use the `web_search` tool, you must set the TAVILY_API_KEY!")

    if scope == "answer":
        model = model.with_structured_output(Response)
        return model
    elif scope == "execute":
        execute_instructions = hub.pull("ih/ih-react-agent-executor")
        model = create_react_agent(model, tools=[web_search], state_modifier=execute_instructions)
        return model
    elif scope == "plan":
        model = model.bind(temperature=0)
        model = model.with_structured_output(Plan)
        plan_prompt = ChatPromptTemplate.from_template(plan_instructions)

        return plan_prompt | model
    elif scope == "replan":
        model = model.bind(temperature=0)
        model = model.with_structured_output(Act)
        replan_prompt = ChatPromptTemplate.from_template(replan_instructions)

        return replan_prompt | model
    else:
        raise ValueError("Invalid type for the agent.")


async def acall_execute_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the execute model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="execute")

    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task = plan[0]
    task_formatted = (
        f"For the following plan: {plan_str}\n\nYou are tasked with executing step {1}, {task}."
    )
    request = {"messages": [HumanMessage(content=task_formatted)]}
    response = await model_runnable.ainvoke(request, config)

    return {"past_steps": [(task, response["messages"][-1].content)]}


async def acall_plan_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the plan model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="plan")

    input_text = state["messages"][-1].content
    request = {"messages": [state["messages"][-1]]}
    response = await model_runnable.ainvoke(request, config)

    return {"input": input_text, "plan": response.steps}


async def acall_replan_model(state: AgentState, config: RunnableConfig) -> AgentState | dict:
    """Call the replan model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m, scope="replan")
    response = await model_runnable.ainvoke(state, config)

    if isinstance(response.action, Response):
        response_message = AIMessage(content=response.action.response)
        return {"messages": [response_message], "response": response.action.response}
    return {"plan": response.action.steps}


def router(state: AgentState) -> Literal["executer", "done"]:
    """Route the agent based on the state.

    Args:
    ----
        state (AgentState): The state of the agent.

    """
    if "response" in state and state["response"]:
        return "done"

    return "executer"


@register_agent(description="A plan and execute agent.")
def plan_and_execute() -> CompiledStateGraph:
    """Get the plan and execute agent."""
    agent = StateGraph(AgentState)
    agent.add_node("executer", acall_execute_model)
    agent.add_node("planner", acall_plan_model)
    agent.add_node("replanner", acall_replan_model)
    agent.set_entry_point("planner")

    # Always run "executer" after "planner"
    agent.add_edge("planner", "executer")

    # Always run "replanner" after "executer"
    agent.add_edge("executer", "replanner")

    # After "model", if there are tool calls, run "tools". Otherwise END.
    agent.add_conditional_edges("replanner", router, {"executer": "executer", "done": END})

    return agent.compile()
