from typing import List, Optional, Set, Tuple

from backend.app.preprocessing.entity_extractor import (
    extract_claim_entities,
    entity_overlap_score,
    anchor_present,
)
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import (
    RANKING_WEIGHT_RELEVANCE,
    RANKING_WEIGHT_TRUST,
    RANKING_WEIGHT_TYPE_PRIORITY,
    SOURCE_TYPE_PRIORITY,
    SOURCE_TYPE_PRIORITY_DEFAULT,
)

# Tier 1 — anchor missing: source mentions none of the claim's named subjects
_ANCHOR_MISSING_PENALTY = 0.2

# Tier 2 — low general overlap: source shares few entities overall
_ENTITY_OVERLAP_THRESHOLD = 0.20
_ENTITY_OVERLAP_PENALTY = 0.5


class RankingService:
    """
    Ranks a list of Source objects using a weighted scoring formula with
    entity-aware penalties to filter off-topic sources.

    Score = (trust_score * RANKING_WEIGHT_TRUST)
          + (relevance_score * RANKING_WEIGHT_RELEVANCE)
          + (type_priority * RANKING_WEIGHT_TYPE_PRIORITY)

    Penalties applied when claim is provided:
      - Anchor missing (0 named entities from claim in source): × 0.2
      - Low overall entity overlap (< 20%): × 0.5
    """

    def _type_priority(self, source_type: str) -> float:
        return SOURCE_TYPE_PRIORITY.get(
            source_type.strip().lower(), SOURCE_TYPE_PRIORITY_DEFAULT
        )

    def _score(
        self,
        source: Source,
        anchor_entities: Optional[Set[str]] = None,
        all_entities: Optional[Set[str]] = None,
    ) -> float:
        type_p = self._type_priority(source.source_type)
        base = (
            source.trust_score * RANKING_WEIGHT_TRUST
            + source.relevance_score * RANKING_WEIGHT_RELEVANCE
            + type_p * RANKING_WEIGHT_TYPE_PRIORITY
        )

        source_text = f"{source.title or ''} {source.snippet or ''}"

        # Tier 1: hard anchor check — source mentions none of the claim's named subjects
        if anchor_entities and not anchor_present(anchor_entities, source_text):
            base *= _ANCHOR_MISSING_PENALTY
            return base  # No need for tier 2 check — already heavily penalized

        # Tier 2: general entity overlap below threshold
        if all_entities:
            overlap = entity_overlap_score(all_entities, source_text)
            if overlap < _ENTITY_OVERLAP_THRESHOLD:
                base *= _ENTITY_OVERLAP_PENALTY

        return base

    def rank(self, sources: List[Source], claim: Optional[str] = None) -> List[Source]:
        """
        Return sources sorted by descending weighted score.
        When claim is provided, applies entity-overlap penalties to off-topic sources.
        Does not mutate the input list.
        """
        if not sources:
            return []

        anchor_entities: Optional[Set[str]] = None
        all_entities: Optional[Set[str]] = None

        if claim:
            anchors, numbers = extract_claim_entities(claim)
            anchor_entities = anchors if anchors else None
            all_entities = anchors | numbers if (anchors or numbers) else None

        return sorted(
            sources,
            key=lambda s: self._score(s, anchor_entities, all_entities),
            reverse=True,
        )
