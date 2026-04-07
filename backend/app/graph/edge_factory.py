"""
Phase 10: Edge Factory.

Builds Edge objects connecting the main claim node to each evidence node.
Edge type, color, width, and dashed style are all derived from the LLM
classification and edge confidence score.
"""

from typing import Dict, List, Optional

from backend.app.models.llm_model import LLMResult, SourceClassification
from backend.app.models.nli_model import NLIResult
from backend.app.schemas.edge_schema import Edge
from backend.app.schemas.source_schema import Source
from backend.app.services.confidence_service import ConfidenceService
from backend.app.utils.constants import (
    EDGE_COLOR_CORRELATED,
    EDGE_COLOR_INSUFFICIENT,
    EDGE_COLOR_REFUTES,
    EDGE_COLOR_SUPPORTS,
    EDGE_WIDTH_MAX,
    EDGE_WIDTH_MIN,
)

# LLM classification → edge_type
_CLASSIFICATION_TO_EDGE_TYPE: Dict[str, str] = {
    "direct_support":     "supports",
    "direct_refute":      "refutes",
    "correlated_context": "correlated",
    "insufficient":       "insufficient",
}

# edge_type → hex color
_EDGE_TYPE_TO_COLOR: Dict[str, str] = {
    "supports":     EDGE_COLOR_SUPPORTS,
    "refutes":      EDGE_COLOR_REFUTES,
    "correlated":   EDGE_COLOR_CORRELATED,
    "insufficient": EDGE_COLOR_INSUFFICIENT,
}

# edge_type → dashed (correlated and insufficient are rendered dashed)
_EDGE_TYPE_DASHED: Dict[str, bool] = {
    "supports":     False,
    "refutes":      False,
    "correlated":   True,
    "insufficient": True,
}


def _edge_width(edge_confidence: float) -> float:
    """Scale edge confidence [0, 1] to visual width [EDGE_WIDTH_MIN, EDGE_WIDTH_MAX]."""
    clamped = max(0.0, min(1.0, edge_confidence))
    return round(EDGE_WIDTH_MIN + clamped * (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN), 2)


def build_edges(
    sources: List[Source],
    llm_result: LLMResult,
    nli_results: Dict[int, NLIResult],
    confidence_svc: ConfidenceService,
) -> List[Edge]:
    """
    Build one edge per source, connecting node_main → node_ev_XX.

    Edge properties:
    - edge_type  : from LLM classification (factcheck sources use their NLI label)
    - color      : edge_type → color map
    - weight     : edge confidence score [0, 1]
    - width      : scaled from weight for visual thickness
    - dashed     : True for correlated/insufficient (weaker evidence)
    - label      : short human-readable type string
    - explanation: LLM rationale for this source
    """
    class_by_idx: Dict[int, SourceClassification] = {
        sc.index: sc for sc in llm_result.sources
    }

    edges: List[Edge] = []
    for i, source in enumerate(sources, start=1):
        sc: Optional[SourceClassification] = class_by_idx.get(i)
        llm_class = sc.classification if sc else "insufficient"
        rationale = (sc.rationale or "") if sc else ""

        edge_type = _CLASSIFICATION_TO_EDGE_TYPE.get(llm_class, "insufficient")

        nli = nli_results.get(i - 1)
        edge_conf = confidence_svc.compute_edge_confidence(source, llm_class, nli)

        edges.append(Edge(
            source="node_main",
            target=f"node_ev_{i:02d}",
            edge_type=edge_type,
            weight=round(edge_conf, 3),
            color=_EDGE_TYPE_TO_COLOR.get(edge_type, EDGE_COLOR_INSUFFICIENT),
            width=_edge_width(edge_conf),
            dashed=_EDGE_TYPE_DASHED.get(edge_type, True),
            label=edge_type,
            explanation=rationale[:300] if rationale else None,
        ))

    return edges
