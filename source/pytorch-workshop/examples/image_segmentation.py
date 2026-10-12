import asyncio
import base64
import io
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from src.models import DEFAULT_SEGMENTATION_MODEL, SEGMENTATION_MODELS, get_model


async def run_serverless_segmentation_model(args: Namespace) -> None:
    """Run a segmentation model in serverless mode.

    Args:
    ----
        args (Namespace): The parsed arguments.

    """
    if args.model not in SEGMENTATION_MODELS:
        avail_models = list(SEGMENTATION_MODELS.keys())
        raise ValueError(
            f"{args.model} not available. Available models are: {', '.join(avail_models)}"
        )
    model = get_model(args.model, device=args.device)

    user_input = args.input
    user_input = {"messages": [("user", user_input)]}
    output = await model.ainvoke(user_input)
    _, image_base64 = output["messages"][-1]

    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_SEGMENTATION_MODEL,
        choices=SEGMENTATION_MODELS,
        help=f"The model to execute (default: {DEFAULT_SEGMENTATION_MODEL})",
    )
    parser.add_argument("--input", "-i", required=True, help="The image to use for inference")
    parser.add_argument(
        "--device",
        default="cpu",
        help="The device to use for inference.",
        choices=["cpu", "cuda", "mps"],
    )
    args = parser.parse_args()

    asyncio.run(run_serverless_segmentation_model(args))
