import requests
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from src.models._api import register_model

__all__ = ["OwlViT"]


_CLASS_NAMES = {
    "coco": [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ],
}


@register_model(name="owlvit")
class OwlViT(torch.nn.Module):
    """OwlViT detection model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "google/owlvit-base-patch32".
        device (str): Device type for model weights. Defaults to "cpu".
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.float16".
        **kwargs (dict): Extra model arguments.

    """

    def __init__(
        self,
        model_name_or_path: str = "google/owlvit-base-patch32",
        device: str = "cpu",
        device_map: str = "auto",
        dtype: str | torch.dtype = "float16",
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self._model_name_or_path = model_name_or_path
        self._class_names = _CLASS_NAMES[kwargs.get("dataset_class_names", "coco")]
        self.device = torch.device(device)
        self.device_map = device_map
        self.dtype = dtype

        self.model = OwlViTForObjectDetection.from_pretrained(self._model_name_or_path)
        self.processor = OwlViTProcessor.from_pretrained(self._model_name_or_path)
        self.model.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def __call__(self, image_path: str) -> str:
        """Detect objects in the image.

        Args:
        ----
            image_path (str): The path to the input image.

        """
        if image_path.startswith(("http://", "https://")):
            image = Image.open(requests.get(image_path, stream=True, timeout=3000).raw)
        else:
            image = Image.open(image_path)

        texts = [[f"a photo of a {class_name}" for class_name in self._class_names]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=0.1, target_sizes=target_sizes
        )

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        detections = []
        for box, score, label in zip(boxes, scores, labels, strict=True):
            box = [round(i, 2) for i in box.tolist()]
            label = text[label].split("a photo of a ")[1]
            detections.append(f"{label} (confidence {round(score.item(), 3)}, location {box})")

        return detections

    async def ainvoke(self, user_input: dict) -> dict:
        """Run the detection model on the user input.

        Args:
        ----
            user_input (dict): The user input, containing the image to detection.

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

        detections = self.__call__(conversation)
        text = "\n".join(detections)

        user_input["messages"].append(("owlvit", text))
        return user_input

    def to(self, *args, **kwargs) -> "OwlViT":
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
