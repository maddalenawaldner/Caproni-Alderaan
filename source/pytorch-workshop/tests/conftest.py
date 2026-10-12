import dotenv
import pytest


@pytest.fixture(scope="session", autouse=True)
def load_dotenv() -> None:
    """Load the environment variables from the .env file."""
    dotenv.load_dotenv(dotenv.find_dotenv())


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options for running tests."""
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow unit tests")


def pytest_configure(config: pytest.Config) -> None:
    """Add custom markers for tests."""
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")
    config.addinivalue_line("markers", "slow: mark as slow, skipping them by default")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests that require custom command-line options if the flags are not provided."""
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)

    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
