"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from dataclasses import dataclass
from typing import Any, Dict

from langchain import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    model_name: str
    temperature: float


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    changeme: str = "example"


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    model_name = runtime.context.get("model_name", "gpt-4o-mini")
    temperature = runtime.context.get("temperature", 0.7)
    model = init_chat_model(model=model_name, temperature=temperature)
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}


def build_graph(config: RunnableConfig) -> CompiledStateGraph:
    """Build the graph."""
    debug = bool(config.get("configurable", {}).get("debug", False))
    graph = StateGraph(State, context_schema=Context)
    graph.add_node(call_model)
    graph.add_edge("__start__", "call_model")
    return graph.compile(name="New Graph", debug=debug)
