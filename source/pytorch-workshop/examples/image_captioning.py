import asyncio
from argparse import ArgumentParser, Namespace

from dotenv import load_dotenv

load_dotenv()

from src.models import CAPTIONING_MODELS, DEFAULT_CAPTIONING_MODEL, get_model


async def run_serverless_captioning_model(args: Namespace) -> None:
    """Run a captioning model in serverless mode.

    Args:
    ----
        args (Namespace): The parsed arguments.

    """
    if args.model not in CAPTIONING_MODELS:
        avail_models = list(CAPTIONING_MODELS.keys())
        raise ValueError(
            f"{args.model} not available. Available models are: {', '.join(avail_models)}"
        )
    model = get_model(args.model, device=args.device)

    user_input = args.input
    user_input = {"messages": [("user", user_input)]}
    output = await model.ainvoke(user_input)
    _, message = output["messages"][-1]
    print(message)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_CAPTIONING_MODEL,
        choices=CAPTIONING_MODELS,
        help=f"The model to execute (default: {DEFAULT_CAPTIONING_MODEL})",
    )
    parser.add_argument("--input", "-i", required=True, help="The image to use for inference")
    parser.add_argument(
        "--device",
        default="cpu",
        help="The device to use for inference.",
        choices=["cpu", "cuda", "mps"],
    )
    args = parser.parse_args()

    asyncio.run(run_serverless_captioning_model(args))
