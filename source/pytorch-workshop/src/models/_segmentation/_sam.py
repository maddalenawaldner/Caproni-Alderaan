import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from matplotlib.axes import Axes
from PIL import Image
from transformers import SamModel, SamProcessor

from src.models._api import register_model

__all__ = ["SAM"]


def _show_mask(mask: torch.Tensor, ax: Axes, random_color: bool = False) -> None:
    """Apply a mask on the image.

    Args:
    ----
        mask (torch.Tensor): The mask to apply.
        ax (Axes): The ax to apply the mask on.
        random_color (bool): Whether to select a random color for the mask. Defaults
            to True.

    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def _show_masks_on_image(
    image_path: str,
    masks: list[torch.Tensor] | torch.Tensor,
    scores: list[float] | torch.Tensor | None = None,
) -> str:
    """Apply the masks on the input image.

    Args:
    ----
        image_path (str): The input image.
        masks (list(torch.Tensor) | torch.Tensor): The masks.
        scores (list(torch.Tensor) | torch.Tensor): The scores of the masks.

    """
    if image_path.startswith(("http://", "https://")):
        image = Image.open(requests.get(image_path, stream=True, timeout=3000).raw)
    else:
        image = Image.open(image_path)

    if isinstance(masks, torch.Tensor) and masks.ndim == 4:
        masks = [m for group in masks for m in group]

    if isinstance(scores, torch.Tensor) and scores.ndim == 2:
        scores = [s.item() for group in scores for s in group]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(np.array(image))

    for _, mask in enumerate(masks):
        _show_mask(mask.cpu().detach(), ax, random_color=True)

    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64


@register_model(name="sam")
class SAM(torch.nn.Module):
    """SAM segmentation model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier lifrom
            huggingface.co/models. Defaults to "facebook/sam-vit-huge".
        device (str): Device type for model weights. Defaults to "cpu".
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.float16".
        **kwargs (dict): Extra model arguments.

    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/sam-vit-huge",
        device: str = "cpu",
        device_map: str = "auto",
        dtype: str | torch.dtype = "float16",
        **kwargs: dict,
    ) -> None:
        super().__init__()

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self._model_name_or_path = model_name_or_path
        self._grid_size = kwargs.get("grid_size", 4)
        self.device = torch.device(device)
        self.device_map = device_map
        self.dtype = dtype

        self.model = SamModel.from_pretrained(self._model_name_or_path)
        self.processor = SamProcessor.from_pretrained(self._model_name_or_path)
        self.model.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def __call__(self, image_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Caption the image.

        Args:
        ----
            image_path (str): The path to the input image.

        """
        if image_path.startswith(("http://", "https://")):
            image = Image.open(requests.get(image_path, stream=True, timeout=3000).raw)
        else:
            image = Image.open(image_path)

        input_points = []
        for row in range(self._grid_size):
            for col in range(self._grid_size):
                center_x = (col + 0.5) * (image.size[1] / self._grid_size)
                center_y = (row + 0.5) * (image.size[0] / self._grid_size)
                input_points.append([[center_x, center_y]])

        inputs = self.processor(image, input_points=[input_points], return_tensors="pt")
        inputs.to(device=self.device, dtype=self.dtype)
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        scores = outputs.iou_scores

        return masks, scores

    async def ainvoke(self, user_input: dict) -> dict:
        """Run the segmentation model on the user input.

        Args:
        ----
            user_input (dict): The user input, containing the image to segment.

        """
        message = user_input["messages"][-1]

        conversation = None
        if isinstance(message[1], str) == 1:
            conversation = message[1]
        elif isinstance(message[1], tuple):
            content_type, content = message[1]

            if content_type != "image":
                raise ValueError(
                    f"Invalid type for model message. Expected 'image', got '{content_type}'."
                )

            conversation = content
        elif isinstance(message[1], list):
            for content_type, content in message[1]:
                if content_type != "image":
                    continue

                if conversation is not None:
                    raise ValueError("Multiple input image passed to the model. Expected one.")

                conversation = content
        else:
            raise ValueError(f"Too many values in the content tuple {message[1]}")

        masks, scores = self.__call__(conversation)
        image_base64 = _show_masks_on_image(message, masks[0], scores[0])

        user_input["messages"].append(("sam", image_base64))
        return user_input

    def to(self, *args, **kwargs) -> "SAM":
        """Perform Tensor dtype and/or device conversion.

        Args:
        ----
            *args: Arguments.
            **kwargs: Keyword arguments.

        """
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.device = device
        self.dtype = dtype
        super().to(*args, **kwargs)
