from src.models._detection._owlvit import OwlViT
from src.models._detection._owlvit2 import OwlViT2

__all__ = [
    "DETECTION_MODELS",
    "DEFAULT_DETECTION_MODEL",
    "OwlViT",
]


DETECTION_MODELS = {
    "owlvit": OwlViT,
    "owlvit2": OwlViT2,
}

DEFAULT_DETECTION_MODEL = "owlvit2"
