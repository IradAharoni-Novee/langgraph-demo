"""Browser Tools.

This module provides browser automation tools using Browserbase MCP server
with Stagehand for AI-powered web navigation, extraction, and interaction.
"""

import os

from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from tavily import TavilyClient


def get_tavily_client() -> TavilyClient:
    """Get the Tavily client instance."""
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def get_browserbase_mcp_client() -> MultiServerMCPClient:
    """Get the Browserbase MCP client instance.

    Returns:
        MultiServerMCPClient configured for Browserbase Stagehand tools.
    """
    return MultiServerMCPClient(
        {
            "browserbase": {
                "command": "npx",
                "args": ["@browserbasehq/mcp"],
                "env": {
                    "BROWSERBASE_API_KEY": os.environ["BROWSERBASE_API_KEY"],
                    "BROWSERBASE_PROJECT_ID": os.environ["BROWSERBASE_PROJECT_ID"],
                    "GEMINI_API_KEY": os.environ["GOOGLE_API_KEY"],
                },
                "transport": "stdio",
            },
        }
    )


async def get_browserbase_tools() -> list:
    """Get browser automation tools from Browserbase MCP server.

    This provides access to Stagehand-powered tools:
    - browserbase_stagehand_navigate: Navigate to any URL
    - browserbase_stagehand_act: Perform actions using natural language
    - browserbase_stagehand_extract: Extract text content from pages
    - browserbase_stagehand_observe: Find actionable elements
    - browserbase_screenshot: Capture screenshots
    - browserbase_stagehand_get_url: Get current URL
    - browserbase_session_create: Create a browser session
    - browserbase_session_close: Close the session

    Returns:
        List of LangChain tools from the Browserbase MCP server.
    """
    client = get_browserbase_mcp_client()
    return await client.get_tools()


def internet_search(query: str, max_results: int = 5):
    """Run a web search."""
    return get_tavily_client().search(query, max_results=max_results)


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


# Static tools (non-MCP)
static_tools = [
    think_tool,
    internet_search,
]


async def get_all_tools() -> list:
    """Get all tools including Browserbase MCP tools and static tools.

    Returns:
        Combined list of MCP browser tools and static tools.
    """
    client = get_browserbase_mcp_client()
    mcp_tools = await client.get_tools()
    return mcp_tools + static_tools
