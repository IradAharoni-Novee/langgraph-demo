"""Research Agent - Standalone script for LangGraph deployment.

This module creates a deep research agent with custom tools and prompts
for conducting web research with strategic thinking and context management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents_cli.integrations.sandbox_factory import create_sandbox
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from typing_extensions import Annotated, TypedDict

from agents.deep.research_agent.prompts import (
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from agents.deep.research_agent.tools import get_all_tools

# Default configuration values
DEFAULT_MODEL_NAME = "claude-sonnet-4-5-20250929"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS = 3
DEFAULT_MAX_RESEARCHER_ITERATIONS = 3


@dataclass
class Context:
    """Context parameters for the agent."""

    model_name: str = field(default=DEFAULT_MODEL_NAME)
    temperature: float = field(default=DEFAULT_TEMPERATURE)
    max_concurrent_research_units: int = field(
        default=DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS
    )
    max_researcher_iterations: int = field(default=DEFAULT_MAX_RESEARCHER_ITERATIONS)


class State(TypedDict):
    """State for the research agent graph."""

    messages: Annotated[list, add_messages]
    query: str
    research_complete: bool


def _get_current_date() -> str:
    """Get current date formatted for prompts."""
    return datetime.now().strftime("%Y-%m-%d")


def _build_instructions(max_concurrent: int, max_iterations: int) -> str:
    """Build combined orchestrator instructions."""
    return (
        RESEARCH_WORKFLOW_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
            max_concurrent_research_units=max_concurrent,
            max_researcher_iterations=max_iterations,
        )
    )


def _create_research_subagent(current_date: str, tools: list) -> dict:
    """Create the research sub-agent configuration."""
    return {
        "name": "research-agent",
        "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
        "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
        "tools": tools,
    }


async def prepare_research(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Prepare the research query and initialize state."""
    query = state.get("query", "")
    messages = state.get("messages", [])

    # If query provided but no messages, create initial message
    if query and not messages:
        messages = [HumanMessage(content=query)]

    return {
        "messages": messages,
        "research_complete": False,
    }


async def run_deep_agent(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Run the deep research agent."""
    model_name = runtime.context.model_name
    temperature = runtime.context.temperature
    max_concurrent = runtime.context.max_concurrent_research_units
    max_iterations = runtime.context.max_researcher_iterations

    # Build components
    current_date = _get_current_date()
    instructions = _build_instructions(max_concurrent, max_iterations)

    # Get all tools including Browserbase MCP tools
    tools = await get_all_tools()
    research_sub_agent = _create_research_subagent(current_date, tools)

    # Initialize model
    model = init_chat_model(model=model_name, temperature=temperature)

    # Create and run the deep agent within sandbox context
    with create_sandbox(
        provider="daytona",
        setup_script_path="scripts/sandbox_setup.sh",
    ) as sandbox_backend:
        deep_agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=instructions,
            middleware=[
                ModelCallLimitMiddleware(thread_limit=10),
                ToolCallLimitMiddleware(thread_limit=10),
                ModelFallbackMiddleware(first_model="gpt-4o-mini"),
                ModelRetryMiddleware(),
                ToolRetryMiddleware(),
            ],
            subagents=[research_sub_agent],
            backend=CompositeBackend(default=sandbox_backend, routes={}),
        )

        # Invoke the deep agent with current messages (must be inside with block)
        result = await deep_agent.ainvoke({"messages": state["messages"]})

    return {
        "messages": result.get("messages", []),
        "research_complete": True,
    }


async def finalize_research(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Finalize and post-process research results."""
    # Placeholder for any post-processing logic
    return {"research_complete": True}


def build_graph(config: RunnableConfig) -> CompiledStateGraph:
    """Build the research agent graph with deterministic nodes."""
    debug = bool(config.get("configurable", {}).get("debug", False))

    # Create the graph
    graph = StateGraph(State, context_schema=Context)

    # Add deterministic nodes
    graph.add_node(prepare_research)
    graph.add_node(run_deep_agent)
    graph.add_node(finalize_research)

    # Define edges
    graph.add_edge(START, "prepare_research")
    graph.add_edge("prepare_research", "run_deep_agent")
    graph.add_edge("run_deep_agent", "finalize_research")
    graph.add_edge("finalize_research", END)

    return graph.compile(name="Research Agent", debug=debug)
