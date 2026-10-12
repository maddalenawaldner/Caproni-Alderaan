from src.models._tagging._ram import RAM

__all__ = [
    "TAGGING_MODELS",
    "DEFAULT_TAGGING_MODEL",
    "RAM",
]


TAGGING_MODELS = {
    "ram": RAM,
}

DEFAULT_TAGGING_MODEL = "ram"
