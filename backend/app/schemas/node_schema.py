from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator

from backend.app.schemas.source_schema import Source


class Node(BaseModel):
    """
    Graph node schema.

    A node can represent:
    - the main user claim
    - direct supporting evidence
    - direct refuting evidence
    - insufficient / unclear evidence
    - correlated context
    - fact-check review content
    """

    node_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique node identifier within the graph."
    )

    node_type: str = Field(
        ...,
        description=(
            "Type of node. Allowed values: "
            "main_claim, direct_support, direct_refute, "
            "insufficient_evidence, context_signal, factcheck_review."
        )
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="Primary text shown for this node."
    )

    verdict: str = Field(
        ...,
        description=(
            "Node-level verdict or stance. Allowed values: "
            "verified, rejected, not_enough_info, supports, refutes, "
            "insufficient, correlated, neutral."
        )
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized confidence score for this node."
    )

    size: float = Field(
        default=10.0,
        gt=0.0,
        le=100.0,
        description="Node size to be used by the frontend renderer."
    )

    color: str = Field(
        ...,
        min_length=4,
        max_length=20,
        description="Hex color code or frontend-resolvable color string."
    )

    best_source_url: Optional[HttpUrl] = Field(
        default=None,
        description="Best source URL to open when the node is clicked."
    )

    top_sources: List[Source] = Field(
        default_factory=list,
        max_length=5,
        description="Top sources shown on hover for this node."
    )

    short_explanation: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional short explanation shown in the UI."
    )

    source_count: int = Field(
        default=0,
        ge=0,
        description="Number of supporting sources attached to this node."
    )

    is_main_claim: bool = Field(
        default=False,
        description="Whether this node is the root/main claim node."
    )

    @field_validator(
        "node_id",
        "node_type",
        "text",
        "verdict",
        "color",
        "short_explanation",
        mode="before",
    )
    @classmethod
    def strip_string_fields(cls, value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, value: Optional[str]) -> str:
        if value is None:
            raise ValueError("node_id cannot be empty.")
        return value

    @field_validator("node_type")
    @classmethod
    def validate_node_type(cls, value: Optional[str]) -> str:
        allowed_types = {
            "main_claim",
            "direct_support",
            "direct_refute",
            "insufficient_evidence",
            "context_signal",
            "factcheck_review",
        }
        if value is None:
            raise ValueError("node_type cannot be empty.")
        lowered = value.lower()
        if lowered not in allowed_types:
            raise ValueError(f"node_type must be one of: {sorted(allowed_types)}")
        return lowered

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: Optional[str]) -> str:
        if value is None:
            raise ValueError("text cannot be empty.")
        return value

    @field_validator("verdict")
    @classmethod
    def validate_verdict(cls, value: Optional[str]) -> str:
        allowed_verdicts = {
            "verified",
            "rejected",
            "not_enough_info",
            "supports",
            "refutes",
            "insufficient",
            "correlated",
            "neutral",
        }
        if value is None:
            raise ValueError("verdict cannot be empty.")
        lowered = value.lower()
        if lowered not in allowed_verdicts:
            raise ValueError(f"verdict must be one of: {sorted(allowed_verdicts)}")
        return lowered

    @field_validator("color")
    @classmethod
    def validate_color(cls, value: Optional[str]) -> str:
        if value is None:
            raise ValueError("color cannot be empty.")
        return value

    @field_validator("top_sources")
    @classmethod
    def validate_top_sources(cls, value: List[Source]) -> List[Source]:
        if len(value) > 5:
            raise ValueError("top_sources cannot contain more than 5 items.")
        return value

    @field_validator("short_explanation")
    @classmethod
    def normalize_short_explanation(cls, value: Optional[str]) -> Optional[str]:
        return value if value else None

    @field_validator("source_count")
    @classmethod
    def validate_source_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("source_count cannot be negative.")
        return value

# {
#   "node_id": "node_main_001",
#   "node_type": "main_claim",
#   "text": "Amazon stock rose by 5% today.",
#   "verdict": "verified",
#   "confidence": 0.84,
#   "size": 24,
#   "color": "#2E7D32",
#   "best_source_url": "https://example.com/article",
#   "top_sources": [],
#   "short_explanation": "Multiple sources directly support the claim.",
#   "source_count": 4,
#   "is_main_claim": true
# }