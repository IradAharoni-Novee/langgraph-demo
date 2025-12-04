"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

import logging
from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "claude-sonnet-4-5-20250929"
DEFAULT_TEMPERATURE = 0.0


@dataclass
class Context:
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    model_name: str = field(default=DEFAULT_MODEL_NAME)
    temperature: float = field(default=DEFAULT_TEMPERATURE)


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    messages: Annotated[list[BaseMessage], add] = field(default_factory=list)


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    model_name = runtime.context.model_name
    temperature = runtime.context.temperature
    model = init_chat_model(model=model_name, temperature=temperature)
    response = await model.ainvoke(state.messages)
    return {"messages": [response]}


def build_graph(config: RunnableConfig) -> CompiledStateGraph:
    """Build the graph."""
    debug = bool(config.get("configurable", {}).get("debug", False))
    graph = StateGraph(State, context_schema=Context)
    graph.add_node(call_model)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", END)
    return graph.compile(debug=debug)
