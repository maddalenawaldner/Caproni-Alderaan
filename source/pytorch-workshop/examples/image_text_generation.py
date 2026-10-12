import asyncio
from argparse import ArgumentParser, Namespace

from dotenv import load_dotenv

load_dotenv()

from src.models import DEFAULT_GENERATION_MODEL, GENERATION_MODELS, get_model


async def run_serverless_image_text_generation_model(args: Namespace) -> None:
    """Run a image-text generation model in serverless mode.

    Args:
    ----
        args (Namespace): The parsed arguments.

    """
    if args.model not in GENERATION_MODELS:
        avail_models = list(GENERATION_MODELS.keys())
        raise ValueError(
            f"{args.model} not available. Available models are: {', '.join(avail_models)}"
        )
    model = get_model(args.model, device=args.device)

    conversation = []
    user_input = args.input
    while True:
        if not user_input:
            user_input = input("User: ")

        if user_input.lower() in ["/bye"]:
            break

        attached_image = input("Attach image (optional): ")

        # Add new message to the conversation history.
        content = []
        if attached_image:
            content.append(("image", attached_image))
        content.append(("text", user_input))
        conversation.append(("user", content))

        user_input = {"messages": conversation}
        output = await model.ainvoke(user_input)
        user_input, attached_image = None, None

        _, message = output["messages"][-1]
        print(message)

        if args.input:
            break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_GENERATION_MODEL,
        choices=GENERATION_MODELS,
        help=f"The model to execute (default: {DEFAULT_GENERATION_MODEL})",
    )
    parser.add_argument("--input", "-i", help="The query to pass to the model (default: None)")
    parser.add_argument(
        "--device",
        default="cpu",
        help="The device to use for inference.",
        choices=["cpu", "cuda", "mps"],
    )
    args = parser.parse_args()

    asyncio.run(run_serverless_image_text_generation_model(args))
