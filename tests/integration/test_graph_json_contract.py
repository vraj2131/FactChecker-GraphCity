"""
Phase 11 Integration Tests: GraphResponse JSON contract.

Verifies that the JSON produced by GraphBuilderService (and returned by the
API) conforms exactly to what the React frontend expects:
- Required top-level keys: metadata, nodes, edges
- Node required fields: node_id, node_type, text, verdict, confidence,
  size, color, is_main_claim, source_count
- Edge required fields: source, target, edge_type, weight, color, width, dashed
- Field types and value ranges
- Exactly one is_main_claim=true node
- All edge source/target IDs reference existing nodes
- Metadata counts match actual node lists

No network or LLM calls — builds GraphResponse objects directly via Pydantic.
"""

import json

import pytest

from backend.app.schemas.edge_schema import Edge
from backend.app.schemas.node_schema import Node
from backend.app.schemas.response_schema import GraphMetadata, GraphResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(
    node_id: str,
    node_type: str = "direct_support",
    verdict: str = "supports",
    is_main: bool = False,
    confidence: float = 0.70,
) -> Node:
    return Node(
        node_id=node_id,
        node_type=node_type,
        text="Test node text",
        verdict=verdict,
        confidence=confidence,
        size=24.0 if is_main else 10.0,
        color="#2E7D32",
        is_main_claim=is_main,
        source_count=1,
    )


def _make_edge(source: str, target: str, edge_type: str = "supports") -> Edge:
    return Edge(
        source=source,
        target=target,
        edge_type=edge_type,
        weight=0.70,
        color="#1E88E5",
        width=4.0,
        dashed=False,
    )


def _make_full_response() -> GraphResponse:
    """Build a representative 4-node GraphResponse covering all node types."""
    nodes = [
        _make_node("node_main", "main_claim", "verified", is_main=True, confidence=0.75),
        _make_node("node_ev_01", "direct_support", "supports"),
        _make_node("node_ev_02", "direct_refute", "refutes"),
        _make_node("node_ev_03", "context_signal", "correlated"),
    ]
    edges = [
        _make_edge("node_main", "node_ev_01", "supports"),
        _make_edge("node_main", "node_ev_02", "refutes"),
        _make_edge("node_main", "node_ev_03", "correlated"),
    ]
    metadata = GraphMetadata(
        claim_text="The Eiffel Tower is in Paris.",
        overall_verdict="verified",
        overall_confidence=0.75,
        total_nodes=4,
        total_edges=3,
        support_node_count=1,
        refute_node_count=1,
        insufficient_node_count=0,
        context_node_count=1,
        factcheck_node_count=0,
        top_support_score=0.70,
        top_refute_score=0.70,
    )
    return GraphResponse(metadata=metadata, nodes=nodes, edges=edges)


@pytest.fixture
def graph() -> GraphResponse:
    return _make_full_response()


@pytest.fixture
def graph_dict(graph) -> dict:
    return json.loads(graph.model_dump_json())


# ---------------------------------------------------------------------------
# Top-level structure
# ---------------------------------------------------------------------------

class TestTopLevelStructure:

    def test_has_metadata_key(self, graph_dict):
        assert "metadata" in graph_dict

    def test_has_nodes_key(self, graph_dict):
        assert "nodes" in graph_dict

    def test_has_edges_key(self, graph_dict):
        assert "edges" in graph_dict

    def test_nodes_is_list(self, graph_dict):
        assert isinstance(graph_dict["nodes"], list)

    def test_edges_is_list(self, graph_dict):
        assert isinstance(graph_dict["edges"], list)

    def test_metadata_is_dict(self, graph_dict):
        assert isinstance(graph_dict["metadata"], dict)


# ---------------------------------------------------------------------------
# Metadata contract
# ---------------------------------------------------------------------------

class TestMetadataContract:
    REQUIRED_FIELDS = {
        "claim_text", "overall_verdict", "overall_confidence",
        "total_nodes", "total_edges",
        "support_node_count", "refute_node_count",
        "insufficient_node_count", "context_node_count", "factcheck_node_count",
    }

    def test_all_required_fields_present(self, graph_dict):
        meta = graph_dict["metadata"]
        for field in self.REQUIRED_FIELDS:
            assert field in meta, f"metadata missing field: {field}"

    def test_claim_text_is_string(self, graph_dict):
        assert isinstance(graph_dict["metadata"]["claim_text"], str)

    def test_overall_verdict_is_valid(self, graph_dict):
        assert graph_dict["metadata"]["overall_verdict"] in {"verified", "rejected", "not_enough_info"}

    def test_overall_confidence_in_range(self, graph_dict):
        conf = graph_dict["metadata"]["overall_confidence"]
        assert 0.0 <= conf <= 1.0

    def test_total_nodes_is_int(self, graph_dict):
        assert isinstance(graph_dict["metadata"]["total_nodes"], int)

    def test_total_edges_is_int(self, graph_dict):
        assert isinstance(graph_dict["metadata"]["total_edges"], int)

    def test_node_counts_non_negative(self, graph_dict):
        meta = graph_dict["metadata"]
        for key in ("support_node_count", "refute_node_count", "insufficient_node_count",
                    "context_node_count", "factcheck_node_count"):
            assert meta[key] >= 0

    def test_total_nodes_matches_node_list(self, graph_dict):
        assert graph_dict["metadata"]["total_nodes"] == len(graph_dict["nodes"])

    def test_total_edges_matches_edge_list(self, graph_dict):
        assert graph_dict["metadata"]["total_edges"] == len(graph_dict["edges"])


# ---------------------------------------------------------------------------
# Node contract
# ---------------------------------------------------------------------------

class TestNodeContract:
    REQUIRED_FIELDS = {
        "node_id", "node_type", "text", "verdict",
        "confidence", "size", "color", "is_main_claim", "source_count",
    }
    VALID_NODE_TYPES = {
        "main_claim", "direct_support", "direct_refute",
        "insufficient_evidence", "context_signal", "factcheck_review",
    }
    VALID_VERDICTS = {
        "verified", "rejected", "not_enough_info",
        "supports", "refutes", "insufficient", "correlated", "neutral",
    }

    def test_all_required_fields_on_each_node(self, graph_dict):
        for node in graph_dict["nodes"]:
            for field in self.REQUIRED_FIELDS:
                assert field in node, f"node '{node.get('node_id')}' missing field: {field}"

    def test_node_ids_are_unique(self, graph_dict):
        ids = [n["node_id"] for n in graph_dict["nodes"]]
        assert len(ids) == len(set(ids))

    def test_node_type_valid(self, graph_dict):
        for node in graph_dict["nodes"]:
            assert node["node_type"] in self.VALID_NODE_TYPES

    def test_verdict_valid(self, graph_dict):
        for node in graph_dict["nodes"]:
            assert node["verdict"] in self.VALID_VERDICTS

    def test_confidence_in_range(self, graph_dict):
        for node in graph_dict["nodes"]:
            assert 0.0 <= node["confidence"] <= 1.0

    def test_size_positive(self, graph_dict):
        for node in graph_dict["nodes"]:
            assert node["size"] > 0

    def test_color_non_empty(self, graph_dict):
        for node in graph_dict["nodes"]:
            assert node["color"] and len(node["color"]) >= 4

    def test_exactly_one_main_claim(self, graph_dict):
        main_nodes = [n for n in graph_dict["nodes"] if n["is_main_claim"]]
        assert len(main_nodes) == 1

    def test_main_node_type_is_main_claim(self, graph_dict):
        main = next(n for n in graph_dict["nodes"] if n["is_main_claim"])
        assert main["node_type"] == "main_claim"

    def test_main_node_id_is_node_main(self, graph_dict):
        main = next(n for n in graph_dict["nodes"] if n["is_main_claim"])
        assert main["node_id"] == "node_main"

    def test_source_count_non_negative(self, graph_dict):
        for node in graph_dict["nodes"]:
            assert node["source_count"] >= 0


# ---------------------------------------------------------------------------
# Edge contract
# ---------------------------------------------------------------------------

class TestEdgeContract:
    REQUIRED_FIELDS = {"source", "target", "edge_type", "weight", "color", "width", "dashed"}
    VALID_EDGE_TYPES = {
        "supports", "refutes", "insufficient", "correlated",
        "shared_source", "shared_topic", "same_publisher", "causal_hint", "temporal_relation",
    }

    def test_all_required_fields_on_each_edge(self, graph_dict):
        for edge in graph_dict["edges"]:
            for field in self.REQUIRED_FIELDS:
                assert field in edge, f"edge missing field: {field}"

    def test_edge_type_valid(self, graph_dict):
        for edge in graph_dict["edges"]:
            assert edge["edge_type"] in self.VALID_EDGE_TYPES

    def test_weight_in_range(self, graph_dict):
        for edge in graph_dict["edges"]:
            assert 0.0 <= edge["weight"] <= 1.0

    def test_width_positive(self, graph_dict):
        for edge in graph_dict["edges"]:
            assert edge["width"] > 0

    def test_dashed_is_bool(self, graph_dict):
        for edge in graph_dict["edges"]:
            assert isinstance(edge["dashed"], bool)

    def test_edge_source_ids_exist_in_nodes(self, graph_dict):
        node_ids = {n["node_id"] for n in graph_dict["nodes"]}
        for edge in graph_dict["edges"]:
            assert edge["source"] in node_ids, f"edge source '{edge['source']}' not in nodes"

    def test_edge_target_ids_exist_in_nodes(self, graph_dict):
        node_ids = {n["node_id"] for n in graph_dict["nodes"]}
        for edge in graph_dict["edges"]:
            assert edge["target"] in node_ids, f"edge target '{edge['target']}' not in nodes"

    def test_all_edges_from_main_node(self, graph_dict):
        for edge in graph_dict["edges"]:
            assert edge["source"] == "node_main"


# ---------------------------------------------------------------------------
# Serialisation contract
# ---------------------------------------------------------------------------

class TestSerialisationContract:

    def test_model_dump_json_round_trips(self, graph):
        raw = graph.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["metadata"]["claim_text"] == graph.metadata.claim_text

    def test_no_python_objects_in_output(self, graph_dict):
        # All values should be JSON-native types (no datetime, UUID, etc.)
        raw = json.dumps(graph_dict)
        assert isinstance(raw, str)

    def test_graphresponse_validates_correctly(self):
        # Re-instantiate from dict to confirm Pydantic validation passes
        g = _make_full_response()
        d = json.loads(g.model_dump_json())
        g2 = GraphResponse.model_validate(d)
        assert g2.metadata.overall_verdict == g.metadata.overall_verdict

    def test_empty_edges_allowed(self):
        node = _make_node("node_main", "main_claim", "not_enough_info", is_main=True)
        meta = GraphMetadata(
            claim_text="Obscure claim with no evidence.",
            overall_verdict="not_enough_info",
            overall_confidence=0.10,
            total_nodes=1,
            total_edges=0,
        )
        g = GraphResponse(metadata=meta, nodes=[node], edges=[])
        d = json.loads(g.model_dump_json())
        assert d["edges"] == []
        assert d["metadata"]["total_edges"] == 0
