from typing import List

from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import (
    RANKING_WEIGHT_RELEVANCE,
    RANKING_WEIGHT_TRUST,
    RANKING_WEIGHT_TYPE_PRIORITY,
    SOURCE_TYPE_PRIORITY,
    SOURCE_TYPE_PRIORITY_DEFAULT,
)


class RankingService:
    """
    Ranks a list of Source objects using a weighted scoring formula.

    Score = (trust_score   * RANKING_WEIGHT_TRUST)
          + (relevance_score * RANKING_WEIGHT_RELEVANCE)
          + (type_priority  * RANKING_WEIGHT_TYPE_PRIORITY)

    Source type priorities (defined in constants):
        factcheck  → 1.0
        guardian   → 0.8
        newsapi    → 0.6
        wikipedia  → 0.5
        gdelt      → 0.3
        unknown    → 0.1 (SOURCE_TYPE_PRIORITY_DEFAULT)

    Input list is not mutated. Returns a new list sorted descending by score.
    """

    def _type_priority(self, source_type: str) -> float:
        return SOURCE_TYPE_PRIORITY.get(
            source_type.strip().lower(), SOURCE_TYPE_PRIORITY_DEFAULT
        )

    def _score(self, source: Source) -> float:
        type_p = self._type_priority(source.source_type)
        return (
            source.trust_score * RANKING_WEIGHT_TRUST
            + source.relevance_score * RANKING_WEIGHT_RELEVANCE
            + type_p * RANKING_WEIGHT_TYPE_PRIORITY
        )

    def rank(self, sources: List[Source]) -> List[Source]:
        """
        Return sources sorted by descending weighted score.
        Does not mutate the input list.
        """
        if not sources:
            return []

        return sorted(sources, key=self._score, reverse=True)
