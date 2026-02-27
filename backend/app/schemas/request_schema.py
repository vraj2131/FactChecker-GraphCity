from typing import Optional

from pydantic import BaseModel, Field, field_validator


class VerifyClaimRequest(BaseModel):
    """
    Request schema for verifying a user-provided claim
    and generating a graph response.
    """

    claim_text: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The main user claim to verify."
    )

    max_nodes: int = Field(
        default=50,
        ge=5,
        le=200,
        description="Maximum number of evidence/context nodes to return."
    )

    include_context: bool = Field(
        default=True,
        description="Whether to include correlated/context nodes in addition to direct support/refute evidence."
    )

    include_related_edges: bool = Field(
        default=True,
        description="Whether to include evidence-to-evidence related edges such as shared_source or correlated."
    )

    top_k_sources_per_node: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Maximum number of sources to attach to each node."
    )

    @field_validator("claim_text")
    @classmethod
    def validate_claim_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("claim_text cannot be empty or only whitespace.")
        return cleaned

# {
#   "claim_text": "Amazon stock rose by 5% today.",
#   "max_nodes": 75,
#   "include_context": true,
#   "include_related_edges": true,
#   "top_k_sources_per_node": 5
# }