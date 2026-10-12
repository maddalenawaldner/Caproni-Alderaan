import os
import subprocess
from functools import partial

import requests
import torch
from PIL import Image
from ram import get_transform
from ram import inference_ram as inference
from ram.models import ram

from src.models._api import register_model

__all__ = ["RAM"]

_WEIGHTS = {
    "ram": "https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/ram_swin_large_14m.pth",
}


@register_model(name="ram")
class RAM(torch.nn.Module):
    """RAM tagging model.

    Args:
    ----
        device (str): Device type for model weights. Defaults to "cpu".
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.float32".
        **kwargs (dict): Extra model arguments.

    """

    def __init__(
        self,
        device: str = "cpu",
        device_map: str = "auto",
        dtype: str | torch.dtype = "float32",
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        file_url = _WEIGHTS["ram"]
        file_name = os.path.basename(file_url)
        target_dir = ".cache/models"
        os.makedirs(target_dir, exist_ok=True)

        file_path = os.path.join(target_dir, file_name)
        if not os.path.exists(file_path):
            subprocess.run(["wget", "-c", file_url, "-O", file_path])  # noqa: S603, S607  # nosec B607

        self._image_size = kwargs.get("image_size", 384)
        self.device = torch.device(device)
        self.device_map = device_map
        self.dtype = dtype

        self.model = ram(pretrained=file_path, image_size=self._image_size, vit="swin_l")
        self.processor = get_transform(image_size=self._image_size)
        self.model.to(device=self.device)
        self.model.eval()
        self._forward = partial(inference, model=self.model)

    @torch.no_grad()
    def __call__(self, image_path: str) -> str:
        """Caption the image.

        Args:
        ----
            image_path (str): The path to the input image.

        """
        if image_path.startswith(("http://", "https://")):
            image = Image.open(requests.get(image_path, stream=True, timeout=3000).raw)
        else:
            image = Image.open(image_path)

        inputs = self.processor(image).unsqueeze(0).to(self.device)
        outputs = self._forward(inputs)
        outputs = outputs[0].split(" | ")

        return outputs

    async def ainvoke(self, user_input: dict) -> str:
        """Run the tagging model on the user input.

        Args:
        ----
            user_input (dict): The user input, containing the image to tag.

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

        tags = self.__call__(conversation)
        text = ", ".join(tags)

        user_input["messages"].append(("ram", text))
        return user_input

    def to(self, *args, **kwargs) -> "RAM":
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
