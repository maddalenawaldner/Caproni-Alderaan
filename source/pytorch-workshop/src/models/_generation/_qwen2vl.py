import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.models._api import register_model

__all__ = ["Qwen2VL"]


@register_model(name="qwen2vl")
class Qwen2VL(torch.nn.Module):
    """Qwen2VL generation model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "Qwen/Qwen2-VL-2B-Instruct".
        device (str): Device type for model weights. Defaults to "cpu".
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.float16".
        **kwargs (dict): Extra model arguments.

    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cpu",
        device_map: str = "auto",
        dtype: str | torch.dtype = "float16",
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self._model_name_or_path = model_name_or_path
        self._max_new_tokens = kwargs.get("max_new_tokens", 128)
        self.device = torch.device(device)
        self.device_map = device_map
        self.dtype = dtype

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self._model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(self._model_name_or_path)
        self.model.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def __call__(self, conversation: list[dict]) -> str:
        """Generate text.

        Args:
        ----
            conversation (list[dict]): The conversation with the user.

        """
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids, strict=True)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return output_text

    async def ainvoke(self, user_input: dict) -> dict:
        """Run the generation model on the user input.

        Args:
        ----
            user_input (dict): The user input.

        """
        conversation = []
        for message in user_input["messages"]:
            data = {
                "role": message[0] if message[0] in ["user", "system"] else "assistant",
                "content": [],
            }

            if isinstance(message[1], str) == 1:
                data["content"].append({"type": "text", "text": message[1]})
            elif isinstance(message[1], tuple):
                content_type, content = message[1]
                data["content"].append(
                    {"type": content_type, "url" if content_type == "image" else "text": content}
                )
            elif isinstance(message[1], list):
                for content_type, content in message[1]:
                    data["content"].append(
                        {
                            "type": content_type,
                            "url" if content_type == "image" else "text": content,
                        }
                    )
            else:
                raise ValueError(f"Too many values in the content tuple {message[1]}")
            conversation.append(data)

        generation = self.__call__(conversation)[0]

        user_input["messages"].append(("qwen2vl", generation))
        return user_input

    def to(self, *args, **kwargs) -> "Qwen2VL":
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
