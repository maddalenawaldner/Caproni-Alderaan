import pytest

from src.client import Client


@pytest.fixture
def client() -> Client:
    """Fixture for creating a test client with a clean environment."""
    ac = Client(base_url="http://test", get_info=False)
    ac.update_agent("test-agent", verify=False)
    return ac
