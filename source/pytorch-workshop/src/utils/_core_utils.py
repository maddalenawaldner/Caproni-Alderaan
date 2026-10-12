from pydantic import HttpUrl, TypeAdapter

from src.utils._logging_utils import get_logger

__all__ = [
    "check_str_is_http",
]

log = get_logger(__name__, rank_zero_only=True)


def check_str_is_http(x: str) -> str:
    """Check if a string is a valid HTTP URL.

    Args:
    ----
        x (str): The string to check.

    """
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))
