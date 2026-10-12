from src.models._generation._qwen2vl import Qwen2VL

__all__ = [
    "GENERATION_MODELS",
    "DEFAULT_GENERATION_MODEL",
    "Qwen2VL",
]


GENERATION_MODELS = {
    "qwen2vl": Qwen2VL,
}

DEFAULT_GENERATION_MODEL = "qwen2vl"
