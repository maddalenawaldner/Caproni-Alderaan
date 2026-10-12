import asyncio
from argparse import ArgumentParser, Namespace
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

load_dotenv()

from src.agents import AGENTS, DEFAULT_AGENT, get_agent


async def run_serverless_text_generation(args: Namespace) -> None:
    """Run an agent in serverless mode.

    Args:
    ----
        args (Namespace): The parsed arguments.

    """
    if args.agent not in AGENTS:
        avail_agents = list(AGENTS.keys())
        raise ValueError(
            f"{args.agent} not available. Available agents are: {', '.join(avail_agents)}"
        )
    agent = get_agent(args.agent)
    config = RunnableConfig(recursion_limit=150, configurable={"thread_id": args.thread_id})

    print("Thread ID:", args.thread_id)
    user_input = args.input
    while True:
        if not user_input:
            user_input = input("User: ")

        if user_input.lower() in ["/bye"]:
            break

        user_input = {"messages": [("user", user_input)]}
        output = await agent.ainvoke(user_input, config=config)
        output["messages"][-1].pretty_print()
        user_input = None

        if args.input:
            break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        choices=AGENTS,
        help=f"The agent to interact with (default: {DEFAULT_AGENT})",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="The thread ID for the conversation (default: random UUID)",
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="The query to pass to the agent (default: None)",
    )
    args = parser.parse_args()

    args.thread_id = args.thread_id or str(uuid4())

    asyncio.run(run_serverless_text_generation(args))
