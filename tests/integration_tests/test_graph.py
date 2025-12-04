import pytest

from agents.simple.graph import build_graph

pytestmark = pytest.mark.anyio

config = {"configurable": {"debug": False}}


@pytest.mark.langsmith
async def test_agent_simple_passthrough() -> None:
    inputs = {"changeme": "some_val"}
    res = await build_graph(config).ainvoke(inputs)
    assert res is not None
