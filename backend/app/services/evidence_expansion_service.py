import logging
from typing import List

from backend.app.preprocessing.normalize_text import normalize_snippet_text
from backend.app.preprocessing.snippet_extractor import (
    extract_best_snippet,
    score_sentence_relevance,
)
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import (
    DEFAULT_SNIPPET_MAX_CHARS,
    DEFAULT_SNIPPET_MAX_SENTENCES,
    SNIPPET_FALLBACK_TO_TITLE,
    SNIPPET_MIN_RELEVANCE_SCORE,
)

logger = logging.getLogger(__name__)


class EvidenceExpansionService:
    """
    Enriches a list of Source objects with clean, claim-relevant snippets.

    For each source:
    1. If snippet exists (any length): run extract_best_snippet to pick the
       most claim-relevant sentence(s). If extraction fails (e.g. single very
       short sentence), fall back to the normalized original.
    2. If no snippet: fall back to the source title (if SNIPPET_FALLBACK_TO_TITLE).

    After snippet resolution, sources whose final snippet scores below
    SNIPPET_MIN_RELEVANCE_SCORE against the claim are dropped. This requires
    at least 2 meaningful claim terms to overlap, filtering single-term noise
    (e.g. "Crystal Palace" matching "palace") before NLI inference.
    """

    def expand(self, claim: str, sources: List[Source]) -> List[Source]:
        """
        Return a new list of Source objects with clean, claim-relevant snippets.

        Sources whose resolved snippet has zero relevance to the claim are
        dropped. All other sources preserve their original order.

        Args:
            claim:   The main claim text used to score sentence relevance.
            sources: List of Source objects from the retrieval pipeline.

        Returns:
            List[Source] with snippets cleaned and zero-relevance sources removed.
        """
        if not sources:
            return []

        expanded: List[Source] = []
        dropped = 0

        for source in sources:
            new_snippet = self._resolve_snippet(claim, source)

            # Relevance filter: drop sources whose snippet does not meet the
            # minimum claim-term overlap threshold (requires at least 2 key terms)
            if new_snippet and score_sentence_relevance(claim, new_snippet) < SNIPPET_MIN_RELEVANCE_SCORE:
                logger.debug(
                    "Dropping zero-relevance source source_id=%s", source.source_id
                )
                dropped += 1
                continue

            if new_snippet == source.snippet:
                expanded.append(source)
            else:
                expanded.append(source.model_copy(update={"snippet": new_snippet}))

        logger.debug(
            "EvidenceExpansionService: %d sources kept, %d dropped (zero relevance) for claim='%s'",
            len(expanded),
            dropped,
            claim[:60],
        )

        return expanded

    def _resolve_snippet(self, claim: str, source: Source) -> str | None:
        """
        Determine the best snippet for a single source.

        Priority:
        1. If snippet exists: run extract_best_snippet to pick the most
           claim-relevant sentence(s) regardless of snippet length.
           If extraction returns nothing (e.g. single very short sentence),
           fall back to the normalized raw snippet.
        2. If no snippet: use title as fallback (if SNIPPET_FALLBACK_TO_TITLE).
        """
        raw = source.snippet

        # --- Case 1: snippet exists (any length) ---
        if raw and raw.strip():
            # Always attempt sentence-level extraction for claim relevance
            extracted = extract_best_snippet(
                claim=claim,
                text=raw,
                max_sentences=DEFAULT_SNIPPET_MAX_SENTENCES,
                max_chars=DEFAULT_SNIPPET_MAX_CHARS,
            )
            if extracted:
                return extracted
            # Extraction failed (e.g. single sentence below min-word threshold)
            # Fall back to the normalized original if it fits within char limit
            if len(raw) <= DEFAULT_SNIPPET_MAX_CHARS:
                return normalize_snippet_text(raw)
            logger.debug(
                "extract_best_snippet returned None for source_id=%s; trying title fallback.",
                source.source_id,
            )

        # --- Case 2: no snippet or extraction+fallback both failed ---
        if SNIPPET_FALLBACK_TO_TITLE and source.title and source.title.strip():
            fallback = normalize_snippet_text(source.title)
            if fallback:
                logger.debug(
                    "Using title fallback for source_id=%s", source.source_id
                )
                return fallback

        return None
