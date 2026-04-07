"""
Unit tests for Phase 10: Graph Builder.

Tests cover:
- node_factory: build_main_node, build_evidence_nodes
- edge_factory: build_edges
- GraphBuilderService.build() end-to-end

All tests are pure unit tests — no network, no LLM, no FAISS, no NLI model.
Inputs are constructed directly from dataclasses and Pydantic models.
"""

import pytest
from typing import Dict, List, Optional

from backend.app.graph.node_factory import build_main_node, build_evidence_nodes
from backend.app.graph.edge_factory import build_edges
from backend.app.models.llm_model import LLMResult, SourceClassification
from backend.app.models.nli_model import NLIResult
from backend.app.schemas.node_schema import Node
from backend.app.schemas.edge_schema import Edge
from backend.app.schemas.source_schema import Source
from backend.app.services.confidence_service import ConfidenceOutput, ConfidenceService
from backend.app.services.graph_builder_service import GraphBuilderService
from backend.app.services.verify_claim_service import VerifyClaimResult
from backend.app.utils.constants import (
    EDGE_COLOR_CORRELATED,
    EDGE_COLOR_INSUFFICIENT,
    EDGE_COLOR_REFUTES,
    EDGE_COLOR_SUPPORTS,
    NODE_COLOR_DIRECT_REFUTE,
    NODE_COLOR_DIRECT_SUPPORT,
    NODE_COLOR_FACTCHECK_REVIEW,
    NODE_COLOR_INSUFFICIENT,
    NODE_COLOR_MAIN_NEI,
    NODE_COLOR_MAIN_REJECTED,
    NODE_COLOR_MAIN_VERIFIED,
    NODE_SIZE_DIRECT_EVIDENCE,
    NODE_SIZE_MAIN_CLAIM,
    NODE_SIZE_WEAK_EVIDENCE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_source(
    i: int,
    source_type: str = "wikipedia",
    snippet: str = "Some evidence text.",
    url: str = None,
) -> Source:
    return Source(
        source_id=f"src_{i:02d}",
        source_type=source_type,
        title=f"Source {i} title",
        url=url or f"https://example.com/source-{i}",
        publisher="Test Publisher",
        snippet=snippet,
        trust_score=0.75,
        relevance_score=0.80,
    )


def _make_llm_result(
    verdict: str = "supported",
    confidence: float = 0.9,
    best_source_index: int = 1,
    classifications: List[tuple] = None,
) -> LLMResult:
    """classifications: list of (index, classification, rationale)"""
    if classifications is None:
        classifications = [(1, "direct_support", "Strong evidence.")]
    return LLMResult(
        overall_verdict=verdict,
        confidence=confidence,
        best_source_index=best_source_index,
        short_explanation="Test explanation.",
        sources=[
            SourceClassification(index=idx, classification=cls, rationale=rat)
            for idx, cls, rat in classifications
        ],
    )


def _make_nli(label: str = "supports", confidence: float = 0.85) -> NLIResult:
    return NLIResult(
        label=label,
        confidence=confidence,
        scores={"supports": 0.85, "refutes": 0.05, "not_enough_info": 0.10},
    )


def _make_confidence_output(
    verdict: str = "verified",
    confidence: float = 0.72,
) -> ConfidenceOutput:
    return ConfidenceOutput(
        overall_confidence=confidence,
        overall_verdict=verdict,
        support_score=0.8,
        refute_score=0.1,
        evidence_quality=0.75,
        corroboration=0.5,
        coverage=0.6,
        raw_confidence=0.65,
    )


def _make_verify_result(
    claim: str = "The Eiffel Tower is in Paris.",
    sources: List[Source] = None,
    llm_result: LLMResult = None,
    nli_results: Dict[int, NLIResult] = None,
    confidence_output: ConfidenceOutput = None,
) -> VerifyClaimResult:
    if sources is None:
        sources = [_make_source(1), _make_source(2)]
    if llm_result is None:
        llm_result = _make_llm_result(
            classifications=[(1, "direct_support", "Good evidence."),
                             (2, "correlated_context", "Topically related.")]
        )
    if nli_results is None:
        nli_results = {0: _make_nli("supports"), 1: _make_nli("not_enough_info", 0.6)}
    if confidence_output is None:
        confidence_output = _make_confidence_output()
    return VerifyClaimResult(
        claim_text=claim,
        sources=sources,
        nli_results=nli_results,
        llm_result=llm_result,
        confidence_output=confidence_output,
        llm_input_sources=sources,
    )


# ---------------------------------------------------------------------------
# node_factory: build_main_node
# ---------------------------------------------------------------------------

class TestBuildMainNode:

    def test_node_id_is_node_main(self):
        node = build_main_node("Claim.", _make_confidence_output(), _make_llm_result(), [_make_source(1)])
        assert node.node_id == "node_main"

    def test_is_main_claim_flag(self):
        node = build_main_node("Claim.", _make_confidence_output(), _make_llm_result(), [_make_source(1)])
        assert node.is_main_claim is True

    def test_node_type_is_main_claim(self):
        node = build_main_node("Claim.", _make_confidence_output(), _make_llm_result(), [_make_source(1)])
        assert node.node_type == "main_claim"

    def test_text_matches_claim(self):
        claim = "Water boils at 100 degrees."
        node = build_main_node(claim, _make_confidence_output(), _make_llm_result(), [_make_source(1)])
        assert node.text == claim

    def test_verified_verdict_green_color(self):
        conf = _make_confidence_output(verdict="verified")
        node = build_main_node("Claim.", conf, _make_llm_result(), [_make_source(1)])
        assert node.color == NODE_COLOR_MAIN_VERIFIED
        assert node.verdict == "verified"

    def test_rejected_verdict_red_color(self):
        conf = _make_confidence_output(verdict="rejected")
        node = build_main_node("Claim.", conf, _make_llm_result(verdict="refuted"), [_make_source(1)])
        assert node.color == NODE_COLOR_MAIN_REJECTED
        assert node.verdict == "rejected"

    def test_nei_verdict_grey_color(self):
        conf = _make_confidence_output(verdict="not_enough_info", confidence=0.10)
        node = build_main_node("Claim.", conf, _make_llm_result(verdict="insufficient"), [_make_source(1)])
        assert node.color == NODE_COLOR_MAIN_NEI
        assert node.verdict == "not_enough_info"

    def test_size_is_main_claim_size(self):
        node = build_main_node("Claim.", _make_confidence_output(), _make_llm_result(), [_make_source(1)])
        assert node.size == NODE_SIZE_MAIN_CLAIM

    def test_confidence_rounded(self):
        conf = _make_confidence_output(confidence=0.74123456)
        node = build_main_node("Claim.", conf, _make_llm_result(), [_make_source(1)])
        assert node.confidence == 0.741

    def test_best_source_url_from_best_source_index(self):
        sources = [_make_source(1, url="https://best.com/article"), _make_source(2)]
        llm = _make_llm_result(best_source_index=1)
        node = build_main_node("Claim.", _make_confidence_output(), llm, sources)
        assert "best.com" in str(node.best_source_url)

    def test_best_source_url_fallback_to_first_source(self):
        sources = [_make_source(1, url="https://first.com/article")]
        llm = _make_llm_result(best_source_index=None)
        llm.best_source_index = None
        node = build_main_node("Claim.", _make_confidence_output(), llm, sources)
        assert node.best_source_url is not None

    def test_top_sources_capped_at_5(self):
        sources = [_make_source(i) for i in range(1, 9)]  # 8 sources
        node = build_main_node("Claim.", _make_confidence_output(), _make_llm_result(), sources)
        assert len(node.top_sources) <= 5

    def test_source_count_matches_input(self):
        sources = [_make_source(i) for i in range(1, 4)]
        node = build_main_node("Claim.", _make_confidence_output(), _make_llm_result(), sources)
        assert node.source_count == 3

    def test_short_explanation_populated(self):
        node = build_main_node("Claim.", _make_confidence_output(), _make_llm_result(), [_make_source(1)])
        assert node.short_explanation is not None
        assert len(node.short_explanation) > 0


# ---------------------------------------------------------------------------
# node_factory: build_evidence_nodes
# ---------------------------------------------------------------------------

class TestBuildEvidenceNodes:

    def _build(self, sources, classifications, nli_labels=None):
        llm = _make_llm_result(classifications=classifications)
        nli = {}
        if nli_labels:
            for i, label in enumerate(nli_labels):
                nli[i] = _make_nli(label)
        return build_evidence_nodes(sources, llm, nli, ConfidenceService())

    def test_one_node_per_source(self):
        sources = [_make_source(i) for i in range(1, 4)]
        nodes = self._build(sources, [(1, "direct_support", ""), (2, "direct_refute", ""), (3, "insufficient", "")])
        assert len(nodes) == 3

    def test_node_ids_are_sequential(self):
        sources = [_make_source(i) for i in range(1, 4)]
        nodes = self._build(sources, [(1, "direct_support", ""), (2, "correlated_context", ""), (3, "insufficient", "")])
        ids = [n.node_id for n in nodes]
        assert ids == ["node_ev_01", "node_ev_02", "node_ev_03"]

    def test_direct_support_node_type_and_color(self):
        nodes = self._build([_make_source(1)], [(1, "direct_support", "Evidence.")])
        assert nodes[0].node_type == "direct_support"
        assert nodes[0].color == NODE_COLOR_DIRECT_SUPPORT

    def test_direct_refute_node_type_and_color(self):
        nodes = self._build([_make_source(1)], [(1, "direct_refute", "Counter evidence.")])
        assert nodes[0].node_type == "direct_refute"
        assert nodes[0].color == NODE_COLOR_DIRECT_REFUTE

    def test_correlated_context_becomes_context_signal(self):
        nodes = self._build([_make_source(1)], [(1, "correlated_context", "Related.")])
        assert nodes[0].node_type == "context_signal"

    def test_insufficient_node_type(self):
        nodes = self._build([_make_source(1)], [(1, "insufficient", "Not relevant.")])
        assert nodes[0].node_type == "insufficient_evidence"
        assert nodes[0].color == NODE_COLOR_INSUFFICIENT

    def test_factcheck_source_always_factcheck_review(self):
        """Factcheck source_type overrides LLM classification for node_type."""
        fc_source = _make_source(1, source_type="factcheck")
        nodes = self._build([fc_source], [(1, "direct_support", "Fact checked.")])
        assert nodes[0].node_type == "factcheck_review"
        assert nodes[0].color == NODE_COLOR_FACTCHECK_REVIEW

    def test_direct_evidence_node_size(self):
        nodes = self._build([_make_source(1)], [(1, "direct_support", "")])
        assert nodes[0].size == NODE_SIZE_DIRECT_EVIDENCE

    def test_weak_evidence_node_size(self):
        nodes = self._build([_make_source(1)], [(1, "correlated_context", "")])
        assert nodes[0].size == NODE_SIZE_WEAK_EVIDENCE

    def test_best_source_url_set_on_evidence_node(self):
        source = _make_source(1, url="https://wiki.org/evidence")
        nodes = self._build([source], [(1, "direct_support", "")])
        assert "wiki.org" in str(nodes[0].best_source_url)

    def test_top_sources_contains_the_source(self):
        source = _make_source(1)
        nodes = self._build([source], [(1, "direct_support", "")])
        assert len(nodes[0].top_sources) == 1
        assert nodes[0].top_sources[0].source_id == source.source_id

    def test_nli_verdict_overrides_type_default(self):
        """NLI refutes label should produce refutes verdict even on a context_signal."""
        nodes = self._build(
            [_make_source(1)],
            [(1, "correlated_context", "")],
            nli_labels=["refutes"],
        )
        assert nodes[0].verdict == "refutes"

    def test_is_main_claim_false_on_evidence_nodes(self):
        nodes = self._build([_make_source(1)], [(1, "direct_support", "")])
        assert nodes[0].is_main_claim is False

    def test_confidence_is_float_in_range(self):
        nodes = self._build([_make_source(1)], [(1, "direct_support", "")])
        assert 0.0 <= nodes[0].confidence <= 1.0

    def test_source_count_is_one(self):
        nodes = self._build([_make_source(1)], [(1, "direct_support", "")])
        assert nodes[0].source_count == 1


# ---------------------------------------------------------------------------
# edge_factory: build_edges
# ---------------------------------------------------------------------------

class TestBuildEdges:

    def _build(self, sources, classifications, nli_labels=None):
        llm = _make_llm_result(classifications=classifications)
        nli = {}
        if nli_labels:
            for i, label in enumerate(nli_labels):
                nli[i] = _make_nli(label)
        return build_edges(sources, llm, nli, ConfidenceService())

    def test_one_edge_per_source(self):
        sources = [_make_source(i) for i in range(1, 4)]
        edges = self._build(sources, [(1, "direct_support", ""), (2, "direct_refute", ""), (3, "insufficient", "")])
        assert len(edges) == 3

    def test_all_edges_source_from_node_main(self):
        sources = [_make_source(i) for i in range(1, 3)]
        edges = self._build(sources, [(1, "direct_support", ""), (2, "direct_refute", "")])
        assert all(e.source == "node_main" for e in edges)

    def test_edge_targets_are_sequential(self):
        sources = [_make_source(i) for i in range(1, 4)]
        edges = self._build(sources, [(1, "direct_support", ""), (2, "correlated_context", ""), (3, "insufficient", "")])
        targets = [e.target for e in edges]
        assert targets == ["node_ev_01", "node_ev_02", "node_ev_03"]

    def test_supports_edge_type_and_color(self):
        edges = self._build([_make_source(1)], [(1, "direct_support", "")])
        assert edges[0].edge_type == "supports"
        assert edges[0].color == EDGE_COLOR_SUPPORTS
        assert edges[0].dashed is False

    def test_refutes_edge_type_and_color(self):
        edges = self._build([_make_source(1)], [(1, "direct_refute", "")])
        assert edges[0].edge_type == "refutes"
        assert edges[0].color == EDGE_COLOR_REFUTES
        assert edges[0].dashed is False

    def test_correlated_edge_is_dashed(self):
        edges = self._build([_make_source(1)], [(1, "correlated_context", "")])
        assert edges[0].edge_type == "correlated"
        assert edges[0].color == EDGE_COLOR_CORRELATED
        assert edges[0].dashed is True

    def test_insufficient_edge_is_dashed(self):
        edges = self._build([_make_source(1)], [(1, "insufficient", "")])
        assert edges[0].edge_type == "insufficient"
        assert edges[0].color == EDGE_COLOR_INSUFFICIENT
        assert edges[0].dashed is True

    def test_edge_weight_in_range(self):
        edges = self._build([_make_source(1)], [(1, "direct_support", "")])
        assert 0.0 <= edges[0].weight <= 1.0

    def test_edge_width_in_range(self):
        edges = self._build([_make_source(1)], [(1, "direct_support", "")])
        from backend.app.utils.constants import EDGE_WIDTH_MAX, EDGE_WIDTH_MIN
        assert EDGE_WIDTH_MIN <= edges[0].width <= EDGE_WIDTH_MAX

    def test_direct_evidence_has_higher_weight_than_insufficient(self):
        sources = [_make_source(1), _make_source(2)]
        edges = self._build(
            sources,
            [(1, "direct_support", "Strong."), (2, "insufficient", "Weak.")],
            nli_labels=["supports", "not_enough_info"],
        )
        assert edges[0].weight > edges[1].weight

    def test_correlated_lower_weight_than_direct(self):
        sources = [_make_source(1), _make_source(2)]
        edges = self._build(
            sources,
            [(1, "direct_support", "Direct."), (2, "correlated_context", "Context.")],
            nli_labels=["supports", "not_enough_info"],
        )
        assert edges[0].weight > edges[1].weight

    def test_edge_label_matches_edge_type(self):
        edges = self._build([_make_source(1)], [(1, "direct_support", "")])
        assert edges[0].label == edges[0].edge_type

    def test_rationale_goes_into_explanation(self):
        edges = self._build([_make_source(1)], [(1, "direct_support", "This is the rationale.")])
        assert edges[0].explanation is not None
        assert "rationale" in edges[0].explanation


# ---------------------------------------------------------------------------
# GraphBuilderService.build() — end-to-end
# ---------------------------------------------------------------------------

class TestGraphBuilderService:

    def _build_graph(self, result=None):
        if result is None:
            result = _make_verify_result()
        return GraphBuilderService().build(result)

    def test_returns_graph_response(self):
        from backend.app.schemas.response_schema import GraphResponse
        graph = self._build_graph()
        assert isinstance(graph, GraphResponse)

    def test_exactly_one_main_node(self):
        graph = self._build_graph()
        main_nodes = [n for n in graph.nodes if n.is_main_claim]
        assert len(main_nodes) == 1

    def test_main_node_id_is_node_main(self):
        graph = self._build_graph()
        main = next(n for n in graph.nodes if n.is_main_claim)
        assert main.node_id == "node_main"

    def test_total_nodes_equals_1_plus_sources(self):
        sources = [_make_source(i) for i in range(1, 4)]
        result = _make_verify_result(
            sources=sources,
            llm_result=_make_llm_result(
                classifications=[(1, "direct_support", ""), (2, "direct_refute", ""), (3, "insufficient", "")]
            ),
            nli_results={0: _make_nli(), 1: _make_nli("refutes"), 2: _make_nli("not_enough_info")},
        )
        graph = self._build_graph(result)
        assert graph.metadata.total_nodes == 4  # 1 main + 3 evidence
        assert len(graph.nodes) == 4

    def test_total_edges_equals_source_count(self):
        sources = [_make_source(i) for i in range(1, 4)]
        result = _make_verify_result(
            sources=sources,
            llm_result=_make_llm_result(
                classifications=[(1, "direct_support", ""), (2, "direct_refute", ""), (3, "insufficient", "")]
            ),
        )
        graph = self._build_graph(result)
        assert graph.metadata.total_edges == 3
        assert len(graph.edges) == 3

    def test_no_missing_node_ids(self):
        graph = self._build_graph()
        assert all(n.node_id for n in graph.nodes)

    def test_no_missing_colors(self):
        graph = self._build_graph()
        assert all(n.color for n in graph.nodes)

    def test_no_missing_urls(self):
        """Every node must have best_source_url populated."""
        graph = self._build_graph()
        for node in graph.nodes:
            assert node.best_source_url is not None, f"{node.node_id} missing best_source_url"

    def test_node_type_matches_edge_type(self):
        """Each edge type must be consistent with its target node type."""
        type_to_edge = {
            "direct_support":        "supports",
            "direct_refute":         "refutes",
            "factcheck_review":      "refutes",
            "context_signal":        "correlated",
            "insufficient_evidence": "insufficient",
        }
        graph = self._build_graph()
        node_map = {n.node_id: n.node_type for n in graph.nodes}
        for edge in graph.edges:
            target_type = node_map[edge.target]
            expected = type_to_edge.get(target_type)
            assert edge.edge_type == expected, (
                f"{edge.target}: node_type={target_type} but edge_type={edge.edge_type}"
            )

    def test_metadata_verdict_matches_confidence_output(self):
        conf = _make_confidence_output(verdict="rejected", confidence=0.65)
        result = _make_verify_result(confidence_output=conf)
        graph = self._build_graph(result)
        assert graph.metadata.overall_verdict == "rejected"
        assert graph.metadata.overall_confidence == 0.65

    def test_metadata_node_counts_correct(self):
        sources = [_make_source(1), _make_source(2), _make_source(3, source_type="factcheck")]
        result = _make_verify_result(
            sources=sources,
            llm_result=_make_llm_result(
                classifications=[
                    (1, "direct_support", ""),
                    (2, "correlated_context", ""),
                    (3, "direct_support", ""),   # factcheck overrides to factcheck_review
                ]
            ),
            nli_results={0: _make_nli("supports"), 1: _make_nli("not_enough_info"), 2: _make_nli("refutes")},
        )
        graph = self._build_graph(result)
        assert graph.metadata.support_node_count == 1
        assert graph.metadata.context_node_count == 1
        assert graph.metadata.factcheck_node_count == 1

    def test_retrieval_notes_populated(self):
        graph = self._build_graph()
        assert graph.metadata.retrieval_notes is not None
        assert len(graph.metadata.retrieval_notes) > 0

    def test_empty_sources_returns_valid_graph(self):
        """No sources → single main node, no edges, NEI verdict."""
        result = _make_verify_result(
            sources=[],
            llm_result=_make_llm_result(verdict="insufficient", confidence=0.0, classifications=[]),
            nli_results={},
            confidence_output=_make_confidence_output(verdict="not_enough_info", confidence=0.05),
        )
        graph = self._build_graph(result)
        assert graph.metadata.total_nodes == 1
        assert graph.metadata.total_edges == 0
        assert graph.metadata.overall_verdict == "not_enough_info"

    def test_all_edges_connect_to_valid_node_ids(self):
        graph = self._build_graph()
        node_ids = {n.node_id for n in graph.nodes}
        for edge in graph.edges:
            assert edge.source in node_ids, f"edge source {edge.source} not in nodes"
            assert edge.target in node_ids, f"edge target {edge.target} not in nodes"

    def test_graph_response_is_json_serialisable(self):
        graph = self._build_graph()
        json_str = graph.model_dump_json()
        assert len(json_str) > 100
        import json
        data = json.loads(json_str)
        assert "metadata" in data
        assert "nodes" in data
        assert "edges" in data
