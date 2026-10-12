import json
import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import Request, Response

from src.client import Client, ClientError
from src.schema import (
    AgentInfo,
    ChatHistory,
    ChatMessage,
    ModelInfo,
    OpenAIChatModelName,
    ServiceMetadata,
)


def test_init() -> None:
    """Test client initialization with different parameters."""
    # Test default values
    client = Client(get_info=False)
    assert client.base_url == "http://localhost"
    assert client.timeout is None

    # Test custom values
    client = Client(
        base_url="http://test",
        timeout=30.0,
        get_info=False,
    )
    assert client.base_url == "http://test"
    assert client.timeout == 30.0
    client.update_agent("test-agent", verify=False)
    assert client.agent == "test-agent"


def test_headers() -> None:
    """Test header generation with and without auth."""
    # Test without auth
    client = Client(get_info=False)
    assert client._headers == {}

    # Test with auth
    with patch.dict(os.environ, {"AUTH_SECRET": "test-secret"}, clear=True):
        client = Client(get_info=False)
        assert client._headers == {"Authorization": "Bearer test-secret"}


def test_invoke(client: Client) -> None:
    """Test synchronous invocation."""
    QUESTION = "What is the weather?"
    ANSWER = "The weather is sunny."

    # Mock successful response
    mock_request = Request("POST", "http://test/invoke")
    mock_response = Response(
        200,
        json={"type": "ai", "content": ANSWER},
        request=mock_request,
    )
    with patch("httpx.post", return_value=mock_response):
        response = client.invoke(QUESTION)
        assert isinstance(response, ChatMessage)
        assert response.type == "ai"
        assert response.content == ANSWER

    # Test with model and thread_id
    with patch("httpx.post", return_value=mock_response) as mock_post:
        response = client.invoke(
            QUESTION,
            model="gpt-4o",
            thread_id="test-thread",
        )
        assert isinstance(response, ChatMessage)
        # Verify request
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["message"] == QUESTION
        assert kwargs["json"]["model"] == "gpt-4o"
        assert kwargs["json"]["thread_id"] == "test-thread"

    # Test error response
    error_response = Response(500, text="Internal Server Error", request=mock_request)
    with patch("httpx.post", return_value=error_response):
        with pytest.raises(ClientError) as exc:
            client.invoke(QUESTION)
        assert "500 Internal Server Error" in str(exc.value)


@pytest.mark.asyncio
async def test_ainvoke(client: Client) -> None:
    """Test asynchronous invocation."""
    QUESTION = "What is the weather?"
    ANSWER = "The weather is sunny."

    # Test successful response
    mock_request = Request("POST", "http://test/invoke")
    mock_response = Response(200, json={"type": "ai", "content": ANSWER}, request=mock_request)
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        response = await client.ainvoke(QUESTION)
        assert isinstance(response, ChatMessage)
        assert response.type == "ai"
        assert response.content == ANSWER

    # Test with model and thread_id
    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        response = await client.ainvoke(
            QUESTION,
            model="gpt-4o",
            thread_id="test-thread",
        )
        assert isinstance(response, ChatMessage)
        assert response.type == "ai"
        assert response.content == ANSWER
        # Verify request
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["message"] == QUESTION
        assert kwargs["json"]["model"] == "gpt-4o"
        assert kwargs["json"]["thread_id"] == "test-thread"

    # Test error response
    error_response = Response(500, text="Internal Server Error", request=mock_request)
    with patch("httpx.AsyncClient.post", return_value=error_response):
        with pytest.raises(ClientError) as exc:
            await client.ainvoke(QUESTION)
        assert "500 Internal Server Error" in str(exc.value)


def test_stream(client: Client) -> None:
    """Test synchronous streaming."""
    QUESTION = "What is the weather?"
    TOKENS = ["The", " weather", " is", " sunny", "."]
    FINAL_ANSWER = "The weather is sunny."

    # Create mock response with streaming events
    events = (
        [f"data: {json.dumps({'type': 'token', 'content': token})}" for token in TOKENS]
        + [
            f"data: {json.dumps({'type': 'message', 'content': {'type': 'ai', 'content': FINAL_ANSWER}})}"  # noqa: E501
        ]
        + ["data: [DONE]"]
    )

    # Mock the streaming response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = events
    mock_response.request = Request("POST", "http://test/stream")
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=None)

    with patch("httpx.stream", return_value=mock_response):
        # Collect all streamed responses
        responses = list(client.stream(QUESTION))

        # Verify tokens were streamed
        assert len(responses) == len(TOKENS) + 1  # tokens + final message
        for i, token in enumerate(TOKENS):
            assert responses[i] == token

        # Verify final message
        final_message = responses[-1]
        assert isinstance(final_message, ChatMessage)
        assert final_message.type == "ai"
        assert final_message.content == FINAL_ANSWER

    # Test error response
    error_response = Response(
        500, text="Internal Server Error", request=Request("POST", "http://test/stream")
    )
    error_response_mock = Mock()
    error_response_mock.__enter__ = Mock(return_value=error_response)
    error_response_mock.__exit__ = Mock(return_value=None)
    with patch("httpx.stream", return_value=error_response_mock):
        with pytest.raises(ClientError) as exc:
            list(client.stream(QUESTION))
        assert "500 Internal Server Error" in str(exc.value)


@pytest.mark.asyncio
async def test_astream(client: Client) -> None:
    """Test asynchronous streaming."""
    QUESTION = "What is the weather?"
    TOKENS = ["The", " weather", " is", " sunny", "."]
    FINAL_ANSWER = "The weather is sunny."

    # Create mock response with streaming events
    events = (
        [f"data: {json.dumps({'type': 'token', 'content': token})}" for token in TOKENS]
        + [
            f"data: {json.dumps({'type': 'message', 'content': {'type': 'ai', 'content': FINAL_ANSWER}})}"  # noqa: E501
        ]
        + ["data: [DONE]"]
    )

    # Create an async iterator for the events
    async def async_events() -> AsyncGenerator[str, None]:
        for event in events:
            yield event

    # Mock the streaming response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.request = Request("POST", "http://test/stream")
    mock_response.aiter_lines = Mock(return_value=async_events())
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.stream = Mock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        # Collect all streamed responses
        responses = []
        async for response in client.astream(QUESTION):
            responses.append(response)

        # Verify tokens were streamed
        assert len(responses) == len(TOKENS) + 1  # tokens + final message
        for i, token in enumerate(TOKENS):
            assert responses[i] == token

        # Verify final message
        final_message = responses[-1]
        assert isinstance(final_message, ChatMessage)
        assert final_message.type == "ai"
        assert final_message.content == FINAL_ANSWER

    # Test error response
    error_response = Response(
        500, text="Internal Server Error", request=Request("POST", "http://test/stream")
    )
    error_response_mock = AsyncMock()
    error_response_mock.__aenter__ = AsyncMock(return_value=error_response)

    mock_client.stream.return_value = error_response_mock

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ClientError) as exc:
            async for _ in client.astream(QUESTION):
                pass
        assert "500 Internal Server Error" in str(exc.value)


@pytest.mark.asyncio
async def test_acreate_feedback(client: Client) -> None:
    """Test asynchronous feedback creation."""
    RUN_ID = "test-run"
    KEY = "test-key"
    SCORE = 0.8
    KWARGS = {"comment": "Great response!"}

    # Test successful response
    mock_response = Response(200, json={}, request=Request("POST", "http://test/feedback"))
    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        await client.acreate_feedback(RUN_ID, KEY, SCORE, KWARGS)
        # Verify request
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["run_id"] == RUN_ID
        assert kwargs["json"]["key"] == KEY
        assert kwargs["json"]["score"] == SCORE
        assert kwargs["json"]["kwargs"] == KWARGS

    # Test error response
    error_response = Response(
        500, text="Internal Server Error", request=Request("POST", "http://test/feedback")
    )
    with patch("httpx.AsyncClient.post", return_value=error_response):
        with pytest.raises(ClientError) as exc:
            await client.acreate_feedback(RUN_ID, KEY, SCORE)
        assert "500 Internal Server Error" in str(exc.value)


def test_get_history(client: Client) -> None:
    """Test chat history retrieval."""
    THREAD_ID = "test-thread"
    HISTORY = {
        "messages": [
            {"type": "human", "content": "What is the weather?"},
            {"type": "ai", "content": "The weather is sunny."},
        ]
    }

    # Mock successful response
    mock_response = Response(200, json=HISTORY, request=Request("POST", "http://test/history"))
    with patch("httpx.post", return_value=mock_response):
        history = client.get_history(THREAD_ID)
        assert isinstance(history, ChatHistory)
        assert len(history.messages) == 2
        assert history.messages[0].type == "human"
        assert history.messages[1].type == "ai"

    # Test error response
    error_response = Response(
        500, text="Internal Server Error", request=Request("POST", "http://test/history")
    )
    with patch("httpx.post", return_value=error_response):
        with pytest.raises(ClientError) as exc:
            client.get_history(THREAD_ID)
        assert "500 Internal Server Error" in str(exc.value)


def test_info(client: Client) -> None:
    """Test agent info retrieval and updating."""
    assert client.info is None
    assert client.agent == "test-agent"

    # Mock info response
    test_info = ServiceMetadata(
        agents=[AgentInfo(name="custom-agent", description="Custom agent")],
        chat_models=[OpenAIChatModelName.GPT_4O_MINI],
        models=[ModelInfo(name="custom-model", description="Custom model")],
        default_agent="custom-agent",
        default_chat_model=OpenAIChatModelName.GPT_4O_MINI,
    )
    test_response = Response(
        200, json=test_info.model_dump(), request=Request("GET", "http://test/info")
    )

    # Update an existing client with info
    with patch("httpx.get", return_value=test_response):
        client.retrieve_info()

    assert client.info == test_info
    assert client.agent == "custom-agent"

    # Test invalid update_agent
    with pytest.raises(ClientError) as exc:
        client.update_agent("unknown-agent")
    assert "Agent unknown-agent not found in available agents: custom-agent" in str(exc.value)

    # Test a fresh client with info
    with patch("httpx.get", return_value=test_response):
        client = Client(base_url="http://test")
    assert client.info == test_info
    assert client.agent == "custom-agent"

    # Test error on invoke if no agent set
    client = Client(base_url="http://test", get_info=False)
    with pytest.raises(ClientError) as exc:
        client.invoke("test")
    assert "No agent selected. Use update_agent() to select an agent." in str(exc.value)
