from argparse import ArgumentParser, Namespace

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from src import chat_settings


def run_server(args: Namespace) -> None:
    """Run the FastAPI server."""
    dev_configs = dict(reload=chat_settings.is_dev(), reload_dirs=["src", "tests"])
    uvicorn.run("src.server:app", host=args.host, port=args.port, **dev_configs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--host",
        default=chat_settings.HOST,
        help=f"The host to run the server on (default: {chat_settings.HOST})",
    )
    parser.add_argument(
        "--port",
        default=chat_settings.PORT,
        help=f"The port to run the server on (default: {chat_settings.PORT})",
    )
    args = parser.parse_args()

    run_server(args)
