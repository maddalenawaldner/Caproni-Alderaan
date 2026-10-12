import operator
from typing import Annotated, List, Literal, NamedTuple, Optional, TypedDict, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src import chat_settings, get_chat_model
from src.agents._api import register_agent

__all__ = ["AgentState", "tree_of_thoughts"]

OperatorType = Literal["+", "-", "*", "/"]
TokenType = Union[float, OperatorType]


class Equation(BaseModel):
    """Formula combining the provided numbers to reach the target of 24."""

    tokens: List[TokenType] = Field(
        description=(
            "The stack of tokens and operators in reverse-polish notation. Example:"
            " [3, 4, '+', -1, '*'] would evaluate to (3 + 4) * -1 = -7."
        ),
    )

    def compute(self) -> float | str:
        """Compute the result of the equation."""
        op_funcs = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }

        stack = []
        for token in self.tokens:
            if isinstance(token, float):
                stack.append(token)
            else:
                b, a = stack.pop(), stack.pop()
                stack.append(op_funcs[token](a, b))

        return stack[0]


class Candidate(NamedTuple):
    """Candidate equation to evaluate."""

    candidate: Equation
    score: Optional[float] = None
    feedback: Optional[str] = None

    def __str__(self) -> str:
        """Return the string representation of the candidate."""
        try:
            computed = self.candidate.compute()
        except Exception as e:
            computed = f"Invalid equation: {self.candidate.tokens}; Error: {repr(e)}"

        return f"Equation({self.candidate.tokens}) = {computed} (Reward: {self.score})"


class ScoredCandidate(Candidate):
    """Candidate equation with a score and feedback."""

    candidate: Equation
    score: float
    feedback: str


def _update_candidates(
    existing: list | None = None,
    updates: list | Literal["clear"] | None = None,
) -> List[str]:
    """Update the list of candidates.

    Args:
    ----
        existing (list, optional): The existing list of candidates.
        updates (list | Literal["clear"], optional): The updates to apply.

    """
    if existing is None:
        existing = []
    if updates is None:
        return existing
    if updates == "clear":
        return []

    return existing + updates


class AgentState(MessagesState, total=False):
    """The state of the agent."""

    problem: str
    candidates: Annotated[List[Candidate], _update_candidates]
    scored_candidates: Annotated[List[ScoredCandidate], _update_candidates]
    depth: Annotated[int, operator.add]


class ExpansionState(AgentState):
    """The state of the expansion agent."""

    seed: Optional[Candidate]


class Configuration(TypedDict, total=False):
    """Configuration for the search algorithm."""

    max_depth: int
    threshold: float
    k: int
    beam_size: int


def _ensure_configurable(config: RunnableConfig) -> Configuration:
    """Get params that configure the search algorithm.

    Args:
    ----
        config (RunnableConfig): The configuration for the agent.

    """
    configurable = config.get("configurable", {})
    return {
        **configurable,
        "max_depth": configurable.get("max_depth", 10),
        "threshold": config.get("threshold", 0.9),
        "k": configurable.get("k", 5),
        "beam_size": configurable.get("beam_size", 3),
    }


system_template = """You are playing the Game of 24. Using the provide numbers, create an equation
that evaluates to 24.\nSubmit exactly {k} guesses for this round.
"""

user_template = "Solve the 24 game for these numbers: {problem}.{candidate}"


class GuessEquations(BaseModel):
    """Submit multiple equations as guesses."""

    reasoning: str = Field(
        description=(
            "The reasoning behind the submitted guesses. Explain how you arrived at these"
            " equations."
        )
    )

    equations: List[Equation] = Field(description="The list of equations to submit as guesses.")


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model in a RunnableLambda.

    Args:
    ----
        model (BaseChatModel): The chat model to wrap.

    """
    model = model.with_structured_output(GuessEquations)
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", user_template)]
    ).partial(candidate="")

    return prompt | model


async def acall_model(
    state: ExpansionState, *, config: RunnableConfig
) -> dict[str, list[Candidate]]:
    """Call the generate model.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    m = get_chat_model(config["configurable"].get("model", chat_settings.DEFAULT_CHAT_MODEL))
    model_runnable = wrap_model(m)

    configurable = _ensure_configurable(config)
    candidate_str = "" if not state.get("seed") else "\n\n" + str(state["seed"])
    problem = state.get("problem", state["messages"][-1].content)
    equation_submission = await model_runnable.ainvoke(
        {
            "problem": problem,
            "candidate": candidate_str,
            "k": configurable["k"],
        },
        config=config,
    )
    new_candidates = [Candidate(candidate=equation) for equation in equation_submission.equations]

    return_data = {"candidates": new_candidates}
    if state.get("problem") is None:
        return_data["problem"] = problem

    return return_data


def _compute_score(problem: str, candidate: Candidate) -> ScoredCandidate:
    """Compute the score for the candidate equation.

    Args:
    ----
        problem (str): The problem to solve.
        candidate (Candidate): The candidate equation to evaluate.

    """
    numbers = list(map(int, problem.split()))

    # Check that the candidate equation uses all 4 numbers exactly once
    used_numbers = [token for token in candidate.candidate.tokens if isinstance(token, float)]
    if sorted(used_numbers) != sorted(numbers):
        score = 0
        feedback = "The equation must use all 4 numbers exactly once."
        return ScoredCandidate(candidate=candidate.candidate, score=score, feedback=feedback)
    try:
        result = candidate.candidate.compute()
        score = 1 / (1 + abs(24 - result))
        feedback = f"Result: {result}"
    except Exception as e:
        score = 0
        feedback = f"Invalid equation. Error: {repr(e)}"
    return ScoredCandidate(candidate=candidate.candidate, score=score, feedback=feedback)


def score(state: AgentState) -> dict[str, list[ScoredCandidate] | str]:
    """Evaluate the candidate generations.

    Args:
    ----
        state (AgentState): The state of the agent.

    """
    candidates = state["candidates"]
    scored = []
    for candidate in candidates:
        scored.append(_compute_score(state["problem"], candidate))
    return {"scored_candidates": scored, "candidates": "clear"}


def prune(state: AgentState, config: RunnableConfig) -> dict[str, list | str | int]:
    """Prune the candidates.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    scored_candidates = state["scored_candidates"]
    beam_size = _ensure_configurable(config)["beam_size"]
    organized = sorted(scored_candidates, key=lambda candidate: candidate[1], reverse=True)
    pruned = organized[:beam_size]

    return {
        "candidates": pruned,  # Update the starting point for the next iteration
        "scored_candidates": "clear",  # Clear the old memory
        "depth": 1,  # Increment the depth by 1
    }


def terminate(state: AgentState, config: RunnableConfig) -> dict[str, list[str]]:
    """Return the output to the user.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    winning_solution = state["candidates"][0]
    search_depth = state["depth"]
    if winning_solution[1] == 1:
        content = f"Found a winning solution in {search_depth} steps: {winning_solution}"
    else:
        content = (
            f"Failed to find a winning solution in {search_depth} steps. Best guess:"
            f" {winning_solution}"
        )
    message = AIMessage(content)

    return {"messages": [message]}


def should_terminate(state: AgentState, config: RunnableConfig) -> Literal["terminate"] | Send:
    """Check if the agent should terminate.

    Args:
    ----
        state (AgentState): The state of the agent.
        config (RunnableConfig): The configuration for the agent.

    """
    configurable = _ensure_configurable(config)
    solved = state["candidates"][0].score >= configurable["threshold"]
    if solved or state["depth"] >= configurable["max_depth"]:
        return "terminate"
    return [
        Send("expand", {**state, "somevalseed": candidate}) for candidate in state["candidates"]
    ]


@register_agent(description="A tree of thoughts agent.")
def tree_of_thoughts() -> CompiledStateGraph:
    """Get the tree of thoughts agent."""
    agent = StateGraph(AgentState)
    agent.add_node("expand", acall_model)
    agent.add_node("score", score)
    agent.add_node("prune", prune)
    agent.add_node("terminate", terminate)
    agent.set_entry_point("expand")

    # Always run "score" after "expand"
    agent.add_edge("expand", "score")

    # Always run "prune" after "score"
    agent.add_edge("score", "prune")

    # Always end after "terminate"
    agent.add_edge("terminate", END)

    # Terminate the agent if the threshold is reached or the max depth is reached
    agent.add_conditional_edges("prune", should_terminate, path_map=["expand", "terminate"])

    return agent.compile(checkpointer=MemorySaver())
