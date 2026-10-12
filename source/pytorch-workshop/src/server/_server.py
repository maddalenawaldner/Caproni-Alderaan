import json
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

import torch
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langsmith import Client as LangsmithClient

from src import chat_settings, utils
from src.agents import DEFAULT_AGENT, get_agent, get_agents_info
from src.models import get_model, get_models_info
from src.schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)

__all__ = [
    "verify_bearer",
    "lifespan",
    "app",
    "router",
    "invoke",
    "message_generator",
    "stream",
    "feedback",
    "history",
    "health_check",
]

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
log = utils.get_logger(__name__, rank_zero_only=True)

MODELS = {}


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    """Verify the bearer token for the API.

    Args:
    ----
        http_auth (Annotated[HTTPAuthorizationCredentials | None, Depends(HTTPBearer)]): The HTTP
            authorization credentials.

    """
    if not chat_settings.AUTH_SECRET:
        return
    auth_secret = chat_settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Context manager for the FastAPI app lifespan.

    Args:
    ----
        app (FastAPI): The FastAPI app.

    """
    models_info = get_models_info()
    for model_info in models_info:
        if model_info.name in MODELS:
            continue

        model = get_model(model_info.name)
        MODELS[model_info.name] = model

    # Construct agent with Sqlite checkpointer
    # TODO it's probably dangerous to share the same checkpointer on multiple agents
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        agents_info = get_agents_info()
        for a in agents_info:
            agent = get_agent(a.name)
            agent.checkpointer = saver
        yield
    # context manager will clean up the AsyncSqliteSaver on exit


app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=[Depends(verify_bearer)])


@router.get("/info")
async def info() -> ServiceMetadata:
    """Get service metadata."""
    chat_models = list(chat_settings.AVAILABLE_CHAT_MODELS)
    chat_models.sort()
    return ServiceMetadata(
        agents=get_agents_info(),
        chat_models=chat_models,
        models=get_models_info(),
        default_agent=DEFAULT_AGENT,
        default_chat_model=chat_settings.DEFAULT_CHAT_MODEL,
    )


def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], UUID]:
    """Parse user input into a format that can be passed to the agent.

    Args:
    ----
        user_input (UserInput): The user input.

    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    configurable = {"thread_id": thread_id, "model": user_input.model}

    if user_input.agent_config:
        if overlap := configurable.keys() & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422, detail=f"agent_config contains reserved keys: {overlap}"
            )
        configurable.update(user_input.agent_config)

    kwargs = {
        "input": {"messages": [HumanMessage(content=user_input.message)]},
        "config": RunnableConfig(
            configurable=configurable,
            run_id=run_id,
        ),
    }
    return kwargs, run_id


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.

    Args:
    ----
        user_input (UserInput): The user input.
        agent_id (str, optional): The agent ID.

    """
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs)
        output = utils.langchain_to_chat_message(response["messages"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        log.error("An exception occurred: %s", e)
        raise HTTPException(status_code=500, detail="Unexpected error") from e


@router.post("/{model_id}/transform")
@router.post("/transform")
async def transform(user_input: UserInput, model_id: str) -> ChatMessage:
    """Invoke a transformer with user input.

    Args:
    ----
        user_input (UserInput): The user input, including the prompt and optional config.
        model_id (str): The model identifier to use from MODELS.

    Returns:
    -------
        ChatMessage: A response message generated by the model.

    """
    model: torch.nn.Module = MODELS[model_id]
    user_input = {"messages": [("user", user_input.message)]}

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if model_id in ["ram"]:
            dtype = torch.float32

        model.to(device=device, dtype=dtype)
        output = await model.ainvoke(user_input)
        _, message = output["messages"][-1]
        model.to("cpu")

        # Convert to your application's `ChatMessage` format
        return ChatMessage(
            type="tool",
            content=message,
            run_id=str(uuid4()),
        )

    except Exception as e:
        log.error("Model invocation error for %s: %s", model_id, e)
        raise HTTPException(status_code=500, detail="Model invocation failed") from e


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.

    Args:
    ----
        user_input (StreamInput): The user input.
        agent_id (str, optional): The agent ID.

    """
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = _parse_input(user_input)

    # Process streamed events from the graph and yield messages over the SSE stream.
    async for event in agent.astream_events(**kwargs, version="v2"):
        if not event:
            continue

        new_messages = []
        # Yield messages written to the graph state after node execution finishes.
        if (
            event["event"] == "on_chain_end"
            # on_chain_end gets called a bunch of times in a graph execution
            # This filters out everything except for "graph node finished"
            and any(t.startswith("graph:step:") for t in event.get("tags", []))
        ):
            if isinstance(event["data"]["output"], Command):
                new_messages = event["data"]["output"].update.get("messages", [])
            elif "messages" in event["data"]["output"]:
                new_messages = event["data"]["output"]["messages"]

        # Also yield intermediate messages from agents.utils.CustomData.adispatch().
        if event["event"] == "on_custom_event" and "custom_data_dispatch" in event.get("tags", []):
            new_messages = [event["data"]]

        for message in new_messages:
            try:
                chat_message = utils.langchain_to_chat_message(message)
                chat_message.run_id = str(run_id)
            except Exception as e:
                log.error("Error parsing message: %s", e)
                yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                continue
            # LangGraph re-sends the input message, which feels weird, so drop it
            if chat_message.type == "human" and chat_message.content == user_input.message:
                continue
            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"  # noqa: E501

        # Yield tokens streamed from LLMs.
        if (
            event["event"] == "on_chat_model_stream"
            and user_input.stream_tokens
            and "llama_guard" not in event.get("tags", [])
        ):
            content = utils.remove_tool_calls(event["data"]["chunk"].content)
            if content:
                # Empty content in the context of OpenAI usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content.
                yield f"data: {json.dumps({'type': 'token', 'content': utils.convert_message_content_to_string(content)})}\n\n"  # noqa: E501
            continue

    yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int, Any]:
    """Return an example of a Server Sent Event response."""
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",  # noqa: E501
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.

    Args:
    ----
        user_input (StreamInput): The user input.
        agent_id (str, optional): The agent ID.

    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post

    Args:
    ----
        feedback (Feedback): The feedback.

    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    """Get chat history.

    Args:
    ----
        input (ChatHistoryInput): The chat history input

    """
    # TODO hard-coding DEFAULT_AGENT here is wonky
    agent: CompiledStateGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [utils.langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        log.error("An exception occurred: %s", e)
        raise HTTPException(status_code=500, detail="Unexpected error") from e


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


app.include_router(router)
