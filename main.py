"""Main entry point for the application."""

import asyncio
import logging
import os

from dotenv import load_dotenv
from langgraph_sdk import get_client

logger = logging.getLogger(__name__)

load_dotenv()
LANGGRAPH_REMOTE_URL = os.getenv("LANGGRAPH_REMOTE_URL")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")


async def main() -> None:
    """Run the main application."""
    client = get_client(url=LANGGRAPH_REMOTE_URL, api_key=LANGSMITH_API_KEY)

    # Create two assistants with different model configurations
    creative_assistant = await client.assistants.create(
        graph_id="agent",
        config={"configurable": {"model_name": "gpt-4o-mini", "temperature": 0.9}},
        name="Creative Assistant",
    )
    precise_assistant = await client.assistants.create(
        graph_id="agent",
        config={"configurable": {"model_name": "gpt-4o-mini", "temperature": 0.1}},
        name="Precise Assistant",
    )

    logger.info("Created assistants: %s, %s", creative_assistant, precise_assistant)

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
