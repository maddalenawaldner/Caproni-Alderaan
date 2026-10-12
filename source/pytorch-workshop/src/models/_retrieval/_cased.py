import requests
import torch
from PIL import Image
from transformers import AutoModel, CLIPProcessor

from src.models._api import register_model

__all__ = ["CaSED"]


@register_model(name="cased")
class CaSED(torch.nn.Module):
    """CaSED retrieval model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "altndrr/cased".
        device (str): Device type for model weights. Defaults to "cpu".
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.float16".
        **kwargs (dict): Extra model arguments.

    """

    def __init__(
        self,
        model_name_or_path: str = "altndrr/cased",
        device: str = "cpu",
        device_map: str = "auto",
        dtype: str | torch.dtype = "float16",
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self._model_name_or_path = model_name_or_path
        self._num_samples = kwargs.get("num_samples", 10)
        self.device = torch.device(device)
        self.device_map = device_map
        self.dtype = dtype

        self.model = AutoModel.from_pretrained("altndrr/cased", trust_remote_code=True)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.to(device=self.device, dtype=self.dtype)

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

        images = self.processor(images=[image], return_tensors="pt", padding=True)
        outputs = self.model(images, alpha=0.7)

        captions = []
        for i in range(len(outputs["vocabularies"])):
            labels, scores = outputs["vocabularies"][i], outputs["scores"][i]
            _, indices = scores.topk(self._num_samples)
            captions.append([labels[index] for index in indices])

        return captions

    async def ainvoke(self, user_input: dict) -> dict:
        """Run the retrieval model on the user input.

        Args:
        ----
            user_input (dict): The user input, containing the image for retrieval.

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

        captions = self.__call__(conversation)[0]
        text = ", ".join(captions)

        user_input["messages"].append(("cased", text))
        return user_input

    def to(self, *args, **kwargs) -> "CaSED":
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
