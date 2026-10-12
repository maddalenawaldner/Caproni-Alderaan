from collections.abc import Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.server import app


@pytest.fixture
def test_client() -> TestClient:
    """Fixture to create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_agent() -> Generator:
    """Fixture to create a mock agent that can be configured for different test scenarios."""
    agent_mock = AsyncMock()
    agent_mock.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Test response")]})
    agent_mock.get_state = Mock()  # Default empty mock for get_state
    with patch("src.server._server.get_agent", Mock(return_value=agent_mock)):
        yield agent_mock


@pytest.fixture
def mock_settings() -> Generator:
    """Fixture to ensure chat settings are clean for each test."""
    with patch("src.server._server.chat_settings") as mock_settings:
        yield mock_settings


@pytest.fixture
def mock_httpx() -> Generator:
    """Patch httpx.stream and httpx.get to use our test client."""
    with TestClient(app) as client:

        def mock_stream(method: str, url: str, **kwargs) -> Mock:
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://localhost", "")
            return client.stream(method, path, **kwargs)

        def mock_get(url: str, **kwargs) -> Mock:
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://localhost", "")
            return client.get(path, **kwargs)

        with patch("httpx.stream", mock_stream):  # noqa: SIM117
            with patch("httpx.get", mock_get):
                yield
