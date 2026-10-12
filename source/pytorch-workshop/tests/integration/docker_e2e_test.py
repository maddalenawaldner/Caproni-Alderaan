import pytest

from src.client import Client


@pytest.mark.docker
def test_service_with_fake_model() -> None:
    """Test the service using the fake model.

    This test requires the service container to be running with USE_FAKE_CHAT_MODEL=true
    """
    client = Client("http://localhost", agent="chatbot")
    response = client.invoke("Tell me a joke?", model="fake")
    assert response.type == "ai"
    assert response.content == "This is a test response from the fake model."
