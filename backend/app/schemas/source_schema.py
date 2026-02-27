from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Source(BaseModel):
    """
    Source attached to a graph node.

    This can represent:
    - a Wikipedia sentence/page
    - a news article
    - a fact-check review
    - a GDELT-derived article reference
    """

    source_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique source identifier within the system."
    )

    source_type: str = Field(
        ...,
        description="Type of source, e.g. wikipedia, newsapi, guardian, factcheck, gdelt."
    )

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable source title."
    )

    url: HttpUrl = Field(
        ...,
        description="Canonical URL for the source."
    )

    publisher: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Publisher or source provider name."
    )

    snippet: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Relevant snippet or extracted evidence text from the source."
    )

    published_at: Optional[str] = Field(
        default=None,
        description="Publication datetime as an ISO-like string if available."
    )

    trust_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Normalized trust score for this source."
    )

    relevance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Normalized relevance score of this source to the current claim or node."
    )

    stance_hint: Optional[str] = Field(
        default=None,
        description="Optional lightweight stance hint such as supports, refutes, insufficient, or correlated."
    )

    @field_validator("source_id", "source_type", "title", "publisher", "snippet", "published_at", "stance_hint", mode="before")
    @classmethod
    def strip_string_fields(cls, value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if value is None:
            raise ValueError("source_id cannot be empty.")
        return value

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: Optional[str]) -> str:
        allowed_types = {"wikipedia", "newsapi", "guardian", "factcheck", "gdelt", "other"}
        if value is None:
            raise ValueError("source_type cannot be empty.")
        lowered = value.lower()
        if lowered not in allowed_types:
            raise ValueError(f"source_type must be one of: {sorted(allowed_types)}")
        return lowered

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: Optional[str]) -> str:
        if value is None:
            raise ValueError("title cannot be empty.")
        return value

    @field_validator("publisher")
    @classmethod
    def normalize_publisher(cls, value: Optional[str]) -> Optional[str]:
        return value if value else None

    @field_validator("snippet")
    @classmethod
    def normalize_snippet(cls, value: Optional[str]) -> Optional[str]:
        return value if value else None

    @field_validator("published_at")
    @classmethod
    def normalize_published_at(cls, value: Optional[str]) -> Optional[str]:
        return value if value else None

    @field_validator("stance_hint")
    @classmethod
    def validate_stance_hint(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        allowed_hints = {"supports", "refutes", "insufficient", "correlated", "neutral"}
        lowered = value.lower()
        if lowered not in allowed_hints:
            raise ValueError(f"stance_hint must be one of: {sorted(allowed_hints)}")
        return lowered

# {
#   "source_id": "src_001",
#   "source_type": "guardian",
#   "title": "Government boosts technology funding",
#   "url": "https://www.theguardian.com/example-article",
#   "publisher": "The Guardian",
#   "snippet": "The government announced increased funding for technology initiatives...",
#   "published_at": "2026-02-26T10:30:00Z",
#   "trust_score": 0.88,
#   "relevance_score": 0.71,
#   "stance_hint": "correlated"
# }