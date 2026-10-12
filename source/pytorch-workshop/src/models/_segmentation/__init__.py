from src.models._segmentation._sam import SAM

__all__ = [
    "SEGMENTATION_MODELS",
    "DEFAULT_SEGMENTATION_MODEL",
    "SAM",
]


SEGMENTATION_MODELS = {
    "sam": SAM,
}

DEFAULT_SEGMENTATION_MODEL = "sam"
