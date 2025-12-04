"""Main entry point for the application."""

import asyncio
import logging
import os

from dotenv import load_dotenv
from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient
from langgraph_sdk.schema import Config, Context

from agents.deep.agent import Context as DeepContext

logger = logging.getLogger(__name__)

load_dotenv()
LANGGRAPH_REMOTE_URL = os.getenv("LANGGRAPH_REMOTE_URL")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")


async def get_or_create_assistant(
    client: LangGraphClient,
    name: str,
    graph_id: str,
    config: Config | None = None,
    context: Context | None = None,
):
    """Get an existing assistant by name or create a new one."""
    existing = await client.assistants.search(metadata={"name": name})
    if existing:
        logger.info("Found existing assistant: %s", name)
        return existing[0]

    logger.info("Creating new assistant: %s", name)
    return await client.assistants.create(
        graph_id=graph_id,
        config=config,
        context=context,
        name=name,
    )


async def main() -> None:
    """Run the main application."""
    client = get_client(url=LANGGRAPH_REMOTE_URL, api_key=LANGSMITH_API_KEY)

    # Get or create two assistants with different model configurations
    creative_assistant = await get_or_create_assistant(
        client,
        name="Creative Assistant",
        graph_id="deep",
        context=DeepContext(
            model_name="claude-sonnet-4-5-20250929",
            temperature=0.9,
        ),
    )
    precise_assistant = await get_or_create_assistant(
        client,
        name="Precise Assistant",
        graph_id="deep",
        context=DeepContext(
            model_name="gpt-4o-mini",
            temperature=0.1,
        ),
    )

    logger.info("Using assistants: %s, %s", creative_assistant, precise_assistant)

    question = {"messages": [{"role": "human", "content": "What is LangGraph?"}]}

    # Run both assistants
    logger.info("--- Creative Assistant (temperature=0.9) ---")
    async for chunk in client.runs.stream(
        None,
        creative_assistant["assistant_id"],
        input=question,
        stream_mode="updates",
    ):
        logger.info("Event: %s | Data: %s", chunk.event, chunk.data)

    logger.info("--- Precise Assistant (temperature=0.1) ---")
    async for chunk in client.runs.stream(
        None,
        precise_assistant["assistant_id"],
        input=question,
        stream_mode="updates",
    ):
        logger.info("Event: %s | Data: %s", chunk.event, chunk.data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
