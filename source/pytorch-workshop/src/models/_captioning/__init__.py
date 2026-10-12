from src.models._captioning._kosmos import Kosmos

__all__ = [
    "CAPTIONING_MODELS",
    "DEFAULT_CAPTIONING_MODEL",
    "Kosmos",
]


CAPTIONING_MODELS = {
    "kosmos": Kosmos,
}

DEFAULT_CAPTIONING_MODEL = "kosmos"
