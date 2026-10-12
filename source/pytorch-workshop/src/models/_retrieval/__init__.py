from src.models._retrieval._cased import CaSED
from src.models._retrieval._clip import CLIP

__all__ = [
    "RETRIEVAL_MODELS",
    "DEFAULT_RETRIEVAL_MODEL",
    "CaSED",
    "CLIP",
]


RETRIEVAL_MODELS = {
    "cased": CaSED,
    "clip": CLIP,
}

DEFAULT_RETRIEVAL_MODEL = "clip"
