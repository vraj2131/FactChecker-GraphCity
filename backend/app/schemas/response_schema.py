from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from backend.app.schemas.edge_schema import Edge
from backend.app.schemas.node_schema import Node


class GraphMetadata(BaseModel):
    """
    Additional graph-level metadata for debugging, UI display,
    and future analytics.
    """

    claim_text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Original user claim."
    )

    overall_verdict: str = Field(
        ...,
        description="Final verdict for the main claim."
    )

    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final confidence score for the main claim verdict."
    )

    total_nodes: int = Field(
        ...,
        ge=1,
        description="Total number of nodes in the graph."
    )

    total_edges: int = Field(
        ...,
        ge=0,
        description="Total number of edges in the graph."
    )

    support_node_count: int = Field(
        default=0,
        ge=0,
        description="Number of direct support nodes."
    )

    refute_node_count: int = Field(
        default=0,
        ge=0,
        description="Number of direct refute nodes."
    )

    insufficient_node_count: int = Field(
        default=0,
        ge=0,
        description="Number of insufficient evidence nodes."
    )

    context_node_count: int = Field(
        default=0,
        ge=0,
        description="Number of correlated/context nodes."
    )

    factcheck_node_count: int = Field(
        default=0,
        ge=0,
        description="Number of fact-check review nodes."
    )

    top_support_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Highest support score among support nodes."
    )

    top_refute_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Highest refute score among refute nodes."
    )

    retrieval_notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional notes about retrieval conditions or limitations."
    )

    @field_validator("claim_text", "overall_verdict", "retrieval_notes", mode="before")
    @classmethod
    def strip_string_fields(cls, value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("claim_text")
    @classmethod
    def validate_claim_text(cls, value: Optional[str]) -> str:
        if value is None:
            raise ValueError("claim_text cannot be empty.")
        return value

    @field_validator("overall_verdict")
    @classmethod
    def validate_overall_verdict(cls, value: Optional[str]) -> str:
        allowed_verdicts = {"verified", "rejected", "not_enough_info"}
        if value is None:
            raise ValueError("overall_verdict cannot be empty.")
        lowered = value.lower()
        if lowered not in allowed_verdicts:
            raise ValueError(f"overall_verdict must be one of: {sorted(allowed_verdicts)}")
        return lowered

    @field_validator("retrieval_notes")
    @classmethod
    def normalize_retrieval_notes(cls, value: Optional[str]) -> Optional[str]:
        return value if value else None


class GraphResponse(BaseModel):
    """
    Full response schema returned by the backend
    for a single verified claim graph.
    """

    metadata: GraphMetadata = Field(
        ...,
        description="Graph-level metadata and summary information."
    )

    nodes: List[Node] = Field(
        ...,
        min_length=1,
        description="All graph nodes including the main claim node."
    )

    edges: List[Edge] = Field(
        default_factory=list,
        description="All edges connecting nodes in the graph."
    )

    @field_validator("nodes")
    @classmethod
    def validate_nodes(cls, value: List[Node]) -> List[Node]:
        if not value:
            raise ValueError("nodes cannot be empty.")

        main_claim_nodes = [node for node in value if node.is_main_claim]
        if len(main_claim_nodes) != 1:
            raise ValueError("There must be exactly one main claim node in the graph.")

        return value

    @field_validator("edges")
    @classmethod
    def validate_edges(cls, value: List[Edge]) -> List[Edge]:
        return value

# {
#   "metadata": {
#     "claim_text": "Amazon stock rose by 5% today.",
#     "overall_verdict": "verified",
#     "overall_confidence": 0.84,
#     "total_nodes": 12,
#     "total_edges": 11,
#     "support_node_count": 5,
#     "refute_node_count": 1,
#     "insufficient_node_count": 2,
#     "context_node_count": 3,
#     "factcheck_node_count": 0,
#     "top_support_score": 0.92,
#     "top_refute_score": 0.41,
#     "retrieval_notes": "Strong direct article evidence found from multiple publishers."
#   },
#   "nodes": [...],
#   "edges": [...]
# }