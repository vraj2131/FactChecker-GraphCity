"""
Phase 11 Integration Tests: POST /api/v1/verify-claim

All tests use FastAPI TestClient with mocked VerifyClaimService and
GraphBuilderService so no real LLM, NLI, or network calls are made.

Test groups:
- TestVerifyClaimHappyPath   : valid requests → 200 + GraphResponse
- TestVerifyClaimValidation  : invalid requests → 422
- TestVerifyClaimErrors      : pipeline errors → 503
- TestHealthEndpoint         : /health → 200
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.dependencies import get_verify_service, get_graph_service
from backend.app.schemas.edge_schema import Edge
from backend.app.schemas.node_schema import Node
from backend.app.schemas.response_schema import GraphMetadata, GraphResponse


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_node(node_id: str, is_main: bool = False) -> Node:
    return Node(
        node_id=node_id,
        node_type="main_claim" if is_main else "direct_support",
        text="Test node text",
        verdict="verified" if is_main else "supports",
        confidence=0.75,
        size=24.0 if is_main else 10.0,
        color="#2E7D32",
        is_main_claim=is_main,
        source_count=1,
    )


def _make_edge(target: str) -> Edge:
    return Edge(
        source="node_main",
        target=target,
        edge_type="supports",
        weight=0.75,
        color="#1E88E5",
        width=4.0,
        dashed=False,
    )


def _make_graph_response(claim: str = "The Eiffel Tower is in Paris.") -> GraphResponse:
    main_node = _make_node("node_main", is_main=True)
    ev_node = _make_node("node_ev_01")
    edge = _make_edge("node_ev_01")
    metadata = GraphMetadata(
        claim_text=claim,
        overall_verdict="verified",
        overall_confidence=0.75,
        total_nodes=2,
        total_edges=1,
        support_node_count=1,
        refute_node_count=0,
        insufficient_node_count=0,
        context_node_count=0,
        factcheck_node_count=0,
        top_support_score=0.75,
        top_refute_score=None,
        retrieval_notes="2 sources retrieved from: wikipedia.",
    )
    return GraphResponse(metadata=metadata, nodes=[main_node, ev_node], edges=[edge])


# ---------------------------------------------------------------------------
# Fixtures — override FastAPI dependencies with mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_verify_svc():
    svc = MagicMock()
    return svc


@pytest.fixture
def mock_graph_svc():
    svc = MagicMock()
    return svc


@pytest.fixture
def client(mock_verify_svc, mock_graph_svc):
    """TestClient with mocked services injected."""
    app.dependency_overrides[get_verify_service] = lambda: mock_verify_svc
    app.dependency_overrides[get_graph_service] = lambda: mock_graph_svc
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# TestVerifyClaimHappyPath
# ---------------------------------------------------------------------------

class TestVerifyClaimHappyPath:

    def test_returns_200(self, client, mock_verify_svc, mock_graph_svc):
        mock_verify_svc.verify.return_value = MagicMock()
        mock_graph_svc.build.return_value = _make_graph_response()

        resp = client.post("/api/v1/verify-claim", json={"claim_text": "The Eiffel Tower is in Paris."})
        assert resp.status_code == 200

    def test_response_has_metadata(self, client, mock_verify_svc, mock_graph_svc):
        mock_verify_svc.verify.return_value = MagicMock()
        mock_graph_svc.build.return_value = _make_graph_response()

        resp = client.post("/api/v1/verify-claim", json={"claim_text": "Water boils at 100C."})
        body = resp.json()
        assert "metadata" in body
        assert "overall_verdict" in body["metadata"]
        assert "overall_confidence" in body["metadata"]
        assert "claim_text" in body["metadata"]

    def test_response_has_nodes(self, client, mock_verify_svc, mock_graph_svc):
        mock_verify_svc.verify.return_value = MagicMock()
        mock_graph_svc.build.return_value = _make_graph_response()

        resp = client.post("/api/v1/verify-claim", json={"claim_text": "Water boils at 100C."})
        body = resp.json()
        assert "nodes" in body
        assert len(body["nodes"]) >= 1

    def test_response_has_edges(self, client, mock_verify_svc, mock_graph_svc):
        mock_verify_svc.verify.return_value = MagicMock()
        mock_graph_svc.build.return_value = _make_graph_response()

        resp = client.post("/api/v1/verify-claim", json={"claim_text": "Water boils at 100C."})
        body = resp.json()
        assert "edges" in body

    def test_exactly_one_main_claim_node(self, client, mock_verify_svc, mock_graph_svc):
        mock_verify_svc.verify.return_value = MagicMock()
        mock_graph_svc.build.return_value = _make_graph_response()

        resp = client.post("/api/v1/verify-claim", json={"claim_text": "The Eiffel Tower is in Paris."})
        nodes = resp.json()["nodes"]
        main_nodes = [n for n in nodes if n["is_main_claim"]]
        assert len(main_nodes) == 1

    def test_claim_text_reflected_in_metadata(self, client, mock_verify_svc, mock_graph_svc):
        claim = "Barack Obama was the 44th US president."
        mock_verify_svc.verify.return_value = MagicMock()
        mock_graph_svc.build.return_value = _make_graph_response(claim)

        resp = client.post("/api/v1/verify-claim", json={"claim_text": claim})
        assert resp.json()["metadata"]["claim_text"] == claim

    def test_verify_service_called_with_stripped_claim(self, client, mock_verify_svc, mock_graph_svc):
        mock_verify_svc.verify.return_value = MagicMock()
        mock_graph_svc.build.return_value = _make_graph_response()

        client.post("/api/v1/verify-claim", json={"claim_text": "  The Eiffel Tower is in Paris.  "})
        mock_verify_svc.verify.assert_called_once_with(
            "The Eiffel Tower is in Paris.", use_cache=True
        )

    def test_graph_service_called_with_verify_result(self, client, mock_verify_svc, mock_graph_svc):
        fake_result = MagicMock()
        mock_verify_svc.verify.return_value = fake_result
        mock_graph_svc.build.return_value = _make_graph_response()

        client.post("/api/v1/verify-claim", json={"claim_text": "The Eiffel Tower is in Paris."})
        mock_graph_svc.build.assert_called_once_with(fake_result)

    def test_confidence_in_range(self, client, mock_verify_svc, mock_graph_svc):
        mock_verify_svc.verify.return_value = MagicMock()
        mock_graph_svc.build.return_value = _make_graph_response()

        resp = client.post("/api/v1/verify-claim", json={"claim_text": "The Eiffel Tower is in Paris."})
        conf = resp.json()["metadata"]["overall_confidence"]
        assert 0.0 <= conf <= 1.0

    def test_verdict_is_valid(self, client, mock_verify_svc, mock_graph_svc):
        mock_verify_svc.verify.return_value = MagicMock()
        mock_graph_svc.build.return_value = _make_graph_response()

        resp = client.post("/api/v1/verify-claim", json={"claim_text": "The Eiffel Tower is in Paris."})
        verdict = resp.json()["metadata"]["overall_verdict"]
        assert verdict in {"verified", "rejected", "not_enough_info"}

    def test_response_is_json_serialisable(self, client, mock_verify_svc, mock_graph_svc):
        mock_verify_svc.verify.return_value = MagicMock()
        mock_graph_svc.build.return_value = _make_graph_response()

        resp = client.post("/api/v1/verify-claim", json={"claim_text": "The Eiffel Tower is in Paris."})
        # If json() succeeds without raising, the body is valid JSON
        body = resp.json()
        assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# TestVerifyClaimValidation
# ---------------------------------------------------------------------------

class TestVerifyClaimValidation:

    def test_missing_claim_text_returns_422(self, client):
        resp = client.post("/api/v1/verify-claim", json={})
        assert resp.status_code == 422

    def test_empty_string_claim_returns_422_or_400(self, client):
        # Pydantic min_length=3 gives 422; our strip check gives 400
        resp = client.post("/api/v1/verify-claim", json={"claim_text": ""})
        assert resp.status_code in {400, 422}

    def test_whitespace_only_claim_returns_422(self, client, mock_verify_svc, mock_graph_svc):
        # "   " passes min_length=3 but the field_validator strips and raises ValueError → 422
        resp = client.post("/api/v1/verify-claim", json={"claim_text": "   "})
        assert resp.status_code == 422

    def test_claim_too_short_returns_422(self, client):
        resp = client.post("/api/v1/verify-claim", json={"claim_text": "ab"})
        assert resp.status_code == 422

    def test_claim_too_long_returns_422(self, client):
        resp = client.post("/api/v1/verify-claim", json={"claim_text": "x" * 1001})
        assert resp.status_code == 422

    def test_max_nodes_out_of_range_returns_422(self, client):
        resp = client.post("/api/v1/verify-claim", json={"claim_text": "Valid claim.", "max_nodes": 300})
        assert resp.status_code == 422

    def test_non_json_body_returns_422(self, client):
        resp = client.post("/api/v1/verify-claim", content="not json", headers={"Content-Type": "application/json"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# TestVerifyClaimErrors
# ---------------------------------------------------------------------------

class TestVerifyClaimErrors:

    def test_pipeline_exception_returns_503(self, client, mock_verify_svc, mock_graph_svc):
        mock_verify_svc.verify.side_effect = RuntimeError("LLM crashed")

        resp = client.post("/api/v1/verify-claim", json={"claim_text": "The Eiffel Tower is in Paris."})
        assert resp.status_code == 503

    def test_503_detail_mentions_pipeline(self, client, mock_verify_svc, mock_graph_svc):
        mock_verify_svc.verify.side_effect = RuntimeError("timeout")

        resp = client.post("/api/v1/verify-claim", json={"claim_text": "The Eiffel Tower is in Paris."})
        assert "Pipeline error" in resp.json()["detail"]

    def test_graph_build_exception_returns_503(self, client, mock_verify_svc, mock_graph_svc):
        mock_verify_svc.verify.return_value = MagicMock()
        mock_graph_svc.build.side_effect = ValueError("node factory failed")

        resp = client.post("/api/v1/verify-claim", json={"claim_text": "The Eiffel Tower is in Paris."})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self):
        with TestClient(app) as c:
            resp = c.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok(self):
        with TestClient(app) as c:
            resp = c.get("/health")
        assert resp.json() == {"status": "ok"}

    def test_health_ready_returns_200(self):
        with TestClient(app) as c:
            resp = c.get("/health/ready")
        assert resp.status_code == 200

    def test_health_ready_returns_status_field(self):
        with TestClient(app) as c:
            resp = c.get("/health/ready")
        assert "status" in resp.json()
