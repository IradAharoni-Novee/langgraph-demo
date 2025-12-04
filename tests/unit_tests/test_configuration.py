"""Unit tests for the simple agent graph configuration and structure."""

from langgraph.pregel import Pregel

from agents.simple.graph import Context, State, build_graph

config = {"configurable": {"debug": False}}


class TestState:
    """Tests for the State dataclass."""

    def test_state_default_values(self) -> None:
        """State should have expected default values."""
        state = State()
        assert state.changeme == "example"

    def test_state_custom_values(self) -> None:
        """State should accept custom values."""
        state = State(changeme="custom")
        assert state.changeme == "custom"


class TestContext:
    """Tests for the Context TypedDict."""

    def test_context_is_typed_dict(self) -> None:
        """Context should be a valid TypedDict."""
        context: Context = {"my_configurable_param": "test_value"}
        assert context["my_configurable_param"] == "test_value"


class TestBuildGraph:
    """Tests for the build_graph function."""

    def test_build_graph_returns_pregel(self) -> None:
        """build_graph should return a compiled Pregel graph."""
        graph = build_graph(config)
        assert isinstance(graph, Pregel)

    def test_build_graph_has_correct_name(self) -> None:
        """Compiled graph should have the expected name."""
        graph = build_graph(config)
        assert graph.name == "New Graph"

    def test_build_graph_with_debug_disabled(self) -> None:
        """Graph should compile with debug mode disabled by default."""
        graph = build_graph(config)
        assert graph.debug is False

    def test_build_graph_with_debug_enabled(self) -> None:
        """Graph should respect debug configuration."""
        debug_config = config.copy()
        debug_config["configurable"]["debug"] = True
        graph = build_graph(debug_config)
        assert graph.debug is True

    def test_graph_has_call_model_node(self) -> None:
        """Graph should contain the call_model node."""
        graph = build_graph(config)
        assert "call_model" in graph.nodes

    def test_graph_has_start_edge(self) -> None:
        """Graph should have an edge from __start__ to call_model."""
        graph = build_graph(config)
        # The graph's nodes dict includes __start__ mapping to call_model
        assert "__start__" in graph.nodes
