from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Edge(BaseModel):
    """
    Graph edge schema.

    An edge connects two nodes and describes the relationship between them.
    """

    source: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Source node_id."
    )

    target: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target node_id."
    )

    edge_type: str = Field(
        ...,
        description=(
            "Type of edge. Allowed values: "
            "supports, refutes, insufficient, correlated, "
            "shared_source, shared_topic, same_publisher, causal_hint, temporal_relation."
        )
    )

    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized edge strength."
    )

    color: str = Field(
        ...,
        min_length=4,
        max_length=20,
        description="Hex color code or frontend-resolvable color string."
    )

    width: float = Field(
        default=1.0,
        gt=0.0,
        le=20.0,
        description="Visual edge width for frontend rendering."
    )

    dashed: bool = Field(
        default=False,
        description="Whether the edge should be rendered as dashed/inferred."
    )

    label: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional short label for UI or debugging."
    )

    explanation: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional explanation for why this edge exists."
    )

    @field_validator("source", "target", "edge_type", "color", "label", "explanation", mode="before")
    @classmethod
    def strip_string_fields(cls, value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: Optional[str]) -> str:
        if value is None:
            raise ValueError("source cannot be empty.")
        return value

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: Optional[str]) -> str:
        if value is None:
            raise ValueError("target cannot be empty.")
        return value

    @field_validator("edge_type")
    @classmethod
    def validate_edge_type(cls, value: Optional[str]) -> str:
        allowed_types = {
            "supports",
            "refutes",
            "insufficient",
            "correlated",
            "shared_source",
            "shared_topic",
            "same_publisher",
            "causal_hint",
            "temporal_relation",
        }
        if value is None:
            raise ValueError("edge_type cannot be empty.")
        lowered = value.lower()
        if lowered not in allowed_types:
            raise ValueError(f"edge_type must be one of: {sorted(allowed_types)}")
        return lowered

    @field_validator("color")
    @classmethod
    def validate_color(cls, value: Optional[str]) -> str:
        if value is None:
            raise ValueError("color cannot be empty.")
        return value

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: Optional[str]) -> Optional[str]:
        return value if value else None

    @field_validator("explanation")
    @classmethod
    def normalize_explanation(cls, value: Optional[str]) -> Optional[str]:
        return value if value else None

# {
#   "source": "node_main_001",
#   "target": "node_ev_004",
#   "edge_type": "supports",
#   "weight": 0.91,
#   "color": "#1E88E5",
#   "width": 3.5,
#   "dashed": false,
#   "label": "supports",
#   "explanation": "This evidence directly confirms the main claim."
# }