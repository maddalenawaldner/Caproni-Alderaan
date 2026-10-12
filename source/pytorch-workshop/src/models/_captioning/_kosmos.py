import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

from src.models._api import register_model

__all__ = ["Kosmos"]


@register_model(name="kosmos")
class Kosmos(torch.nn.Module):
    """Kosmos captioning model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "microsoft/kosmos-2-patch14-224".
        device (str): Device type for model weights. Defaults to "cpu".
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.float16".
        **kwargs (dict): Extra model arguments.

    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/kosmos-2-patch14-224",
        device: str = "cpu",
        device_map: str = "auto",
        dtype: str | torch.dtype = "float16",
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self._model_name_or_path = model_name_or_path
        self.device = torch.device(device)
        self.device_map = device_map
        self.dtype = dtype

        self.model = Kosmos2ForConditionalGeneration.from_pretrained(self._model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(self._model_name_or_path)
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

        prompt = "<grounding> An image of"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.device),
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"].to(self.device),
            use_cache=True,
            max_new_tokens=64,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        caption, _ = self.processor.post_process_generation(generated_text)

        return caption

    async def ainvoke(self, user_input: dict) -> dict:
        """Run the captioning model on the user input.

        Args:
        ----
            user_input (dict): The user input, containing the image to caption.

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

        caption = self.__call__(conversation)

        user_input["messages"].append(("kosmos", caption))
        return user_input

    def to(self, *args, **kwargs) -> "Kosmos":
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
