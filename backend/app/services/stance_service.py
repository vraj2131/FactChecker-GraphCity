import logging
from typing import Dict, List, Optional

from backend.app.models.nli_model import NLIModel, NLIResult
from backend.app.schemas.source_schema import Source
from backend.app.services.cache_service import CacheService
from backend.app.utils.constants import NLI_CACHE_NAMESPACE
from backend.app.utils.hashing import build_cache_key

logger = logging.getLogger(__name__)

# Map NLI output labels to Source.stance_hint allowed values
_NLI_TO_STANCE: dict = {
    "supports": "supports",
    "refutes": "refutes",
    "not_enough_info": "insufficient",
}


class StanceService:
    """
    Classifies each evidence source's snippet against the main claim using NLI.

    Supports an optional cascade mode:
    - Fast model classifies all snippets.
    - Snippets labelled supports/refutes are sent to a stronger confirm_model.
    - The confirm_model result is authoritative for those cases.
    - Snippets already labelled not_enough_info skip the confirm_model entirely.

    This saves compute on the ~60-70% of snippets that are clearly insufficient
    while using the best model quality for the labels that matter (graph edges).

    Flow:
    1. Sources without snippets pass through unchanged.
    2. Check cache for already-classified snippets.
    3. Batch uncached snippets → fast model predict().
    4. [cascade] Collect supports/refutes → confirm_model predict().
    5. Map final NLI label → stance_hint, persist to cache.
    6. Return new List[Source] with stance_hint populated.

    Input sources are never mutated.
    """

    def __init__(
        self,
        model: NLIModel,
        cache: CacheService,
        confirm_model: Optional[NLIModel] = None,
    ) -> None:
        self._model = model
        self._cache = cache
        self._confirm_model = confirm_model
        if confirm_model:
            logger.info(
                "StanceService: cascade mode enabled (%s → %s)",
                model.model_name,
                confirm_model.model_name,
            )

    def classify(self, claim: str, sources: List[Source]) -> List[Source]:
        """
        Run NLI over all sources with snippets and set stance_hint.

        In cascade mode, supports/refutes from the fast model are confirmed
        by the stronger model before being finalised.

        Args:
            claim:   The main claim text.
            sources: List of Source objects from EvidenceExpansionService.

        Returns:
            List[Source] of the same length, with stance_hint set on sources
            that have snippets. Sources without snippets pass through unchanged.

        Raises:
            ValueError: If claim is empty.
        """
        if not claim or not claim.strip():
            raise ValueError("claim cannot be empty.")

        if not sources:
            return []

        # --- Step 1: cache lookup ---
        needs_nli: List[int] = []
        cached_results: Dict[int, NLIResult] = {}

        for i, source in enumerate(sources):
            if not source.snippet or not source.snippet.strip():
                continue

            cache_key = self._build_nli_cache_key(claim, source)
            if self._cache.exists(NLI_CACHE_NAMESPACE, cache_key):
                raw = self._cache.load(NLI_CACHE_NAMESPACE, cache_key)
                if raw is not None:
                    cached_results[i] = NLIResult(
                        label=raw["label"],
                        confidence=raw["confidence"],
                        scores=raw["scores"],
                    )
                    continue

            needs_nli.append(i)

        logger.info(
            "StanceService: %d need NLI, %d from cache, %d no snippet",
            len(needs_nli),
            len(cached_results),
            len(sources) - len(needs_nli) - len(cached_results),
        )

        # --- Step 2: fast model inference ---
        nli_results: Dict[int, NLIResult] = {}
        if needs_nli:
            snippets = [sources[i].snippet for i in needs_nli]
            predictions = self._model.predict(claim, snippets)
            for source_idx, result in zip(needs_nli, predictions):
                nli_results[source_idx] = result

        # --- Step 3: cascade confirmation ---
        if self._confirm_model and nli_results:
            nli_results = self._run_cascade(claim, sources, nli_results)

        # --- Step 4: persist fresh results to cache ---
        for source_idx, result in nli_results.items():
            cache_key = self._build_nli_cache_key(claim, sources[source_idx])
            self._cache.save(NLI_CACHE_NAMESPACE, cache_key, {
                "label": result.label,
                "confidence": result.confidence,
                "scores": result.scores,
            })

        # --- Step 5: merge and build output ---
        all_nli: Dict[int, NLIResult] = {**cached_results, **nli_results}

        output: List[Source] = []
        for i, source in enumerate(sources):
            result: Optional[NLIResult] = all_nli.get(i)
            if result is None:
                output.append(source)
            else:
                stance = _NLI_TO_STANCE.get(result.label, "insufficient")
                output.append(source.model_copy(update={"stance_hint": stance}))

        logger.info(
            "StanceService: classified %d sources for claim='%s'",
            len([s for s in output if s.stance_hint]),
            claim[:60],
        )

        return output

    def _run_cascade(
        self,
        claim: str,
        sources: List[Source],
        initial: Dict[int, NLIResult],
    ) -> Dict[int, NLIResult]:
        """
        Run confirm_model on snippets where the fast model said supports/refutes.

        not_enough_info results skip the confirm_model (no point spending compute
        confirming an already-weak result).

        Returns an updated copy of initial with confirmed labels where applicable.
        """
        to_confirm = [
            idx for idx, result in initial.items()
            if result.label in ("supports", "refutes")
        ]

        if not to_confirm:
            logger.info("Cascade: no supports/refutes to confirm.")
            return initial

        logger.info(
            "Cascade: confirming %d result(s) with %s",
            len(to_confirm),
            self._confirm_model.model_name,
        )

        confirm_snippets = [sources[idx].snippet for idx in to_confirm]
        confirm_preds = self._confirm_model.predict(claim, confirm_snippets)

        updated = dict(initial)
        for source_idx, confirm_result in zip(to_confirm, confirm_preds):
            fast_result = initial[source_idx]
            if fast_result.label != confirm_result.label:
                logger.info(
                    "Cascade override source_id=%s: %s (%.2f) → %s (%.2f)",
                    sources[source_idx].source_id,
                    fast_result.label, fast_result.confidence,
                    confirm_result.label, confirm_result.confidence,
                )
            updated[source_idx] = confirm_result

        return updated

    def _build_nli_cache_key(self, claim: str, source: Source) -> str:
        """
        Build a stable cache key for this (claim, snippet) pair.
        Includes confirm_model name so cascade results are cached separately
        from single-model results.
        """
        kwargs: dict = {"snippet": source.snippet or ""}
        if self._confirm_model:
            kwargs["confirm_model"] = self._confirm_model.model_name
        return build_cache_key(
            source_name=NLI_CACHE_NAMESPACE,
            query=claim,
            **kwargs,
        )
