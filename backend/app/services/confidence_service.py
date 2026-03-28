"""
Phase 9: Confidence Service.

Computes calibrated confidence scores for the main claim and per-source
edge weights.  Correlated context is deliberately excluded from
directional scoring so it cannot inflate the truth score.

Two public methods:
- compute_main_confidence  → ConfidenceOutput  (overall claim verdict)
- compute_edge_confidence  → float              (per-source edge weight)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.models.calibration_model import calibrate
from backend.app.models.llm_model import LLMResult
from backend.app.models.nli_model import NLIResult
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import (
    CONFIDENCE_NEI_CEILING,
    CONFIDENCE_REJECTED_THRESHOLD,
    CONFIDENCE_VERIFIED_THRESHOLD,
    CONFIDENCE_WEIGHT_CORROBORATION,
    CONFIDENCE_WEIGHT_COVERAGE,
    CONFIDENCE_WEIGHT_DIRECTIONAL,
    CONFIDENCE_WEIGHT_EVIDENCE_QUALITY,
    CONFIDENCE_WEIGHT_LLM,
    EDGE_WEIGHT_LLM_CLASS,
    EDGE_WEIGHT_NLI,
    EDGE_WEIGHT_RELEVANCE,
    EDGE_WEIGHT_TRUST,
    LLM_CLASSIFICATION_STRENGTH,
    MIN_INDEPENDENT_SOURCES_FOR_BONUS,
)

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceOutput:
    """Full confidence scoring result for a claim."""

    overall_confidence: float  # calibrated [0, 1]
    overall_verdict: str  # verified / rejected / not_enough_info
    support_score: float  # raw support signal strength
    refute_score: float  # raw refute signal strength
    evidence_quality: float  # quality sub-score
    corroboration: float  # corroboration sub-score
    coverage: float  # coverage sub-score
    raw_confidence: float  # pre-calibration weighted score
    debug: Dict[str, Any] = field(default_factory=dict)


class ConfidenceService:
    """
    Aggregates retrieval, NLI, and LLM signals into calibrated confidence
    scores.

    Main claim confidence formula (5 weighted components):
        directional     (0.35)  — support vs refute signal, direct sources only
        llm_confidence  (0.25)  — LLM's own confidence estimate
        evidence_quality(0.20)  — avg trust × relevance of direct sources
        corroboration   (0.15)  — independent source type agreement
        coverage        (0.05)  — breadth of retriever types (only place
                                  correlated context contributes)
    """

    # ------------------------------------------------------------------
    # Main claim confidence
    # ------------------------------------------------------------------

    def compute_main_confidence(
        self,
        llm_result: LLMResult,
        sources: List[Source],
        nli_results: Dict[int, NLIResult],
    ) -> ConfidenceOutput:
        """
        Compute a calibrated confidence score for the overall claim verdict.

        Args:
            llm_result:  LLMResult from Phase 8 (Groq or local).
            sources:     Classified sources (same list sent to LLM).
            nli_results: Dict mapping source index → NLIResult from StanceService.

        Returns:
            ConfidenceOutput with calibrated verdict and sub-scores.
        """
        verdict = llm_result.overall_verdict  # supported / refuted / insufficient / mixed

        # Build a lookup: source index → LLM classification string
        class_by_idx: Dict[int, str] = {
            sc.index: sc.classification for sc in llm_result.sources
        }

        # --- Partition sources by LLM classification ---
        direct_support_indices: List[int] = []
        direct_refute_indices: List[int] = []
        correlated_indices: List[int] = []

        for i in range(len(sources)):
            llm_class = class_by_idx.get(i + 1, "insufficient")  # 1-indexed
            if llm_class == "direct_support":
                direct_support_indices.append(i)
            elif llm_class == "direct_refute":
                direct_refute_indices.append(i)
            elif llm_class == "correlated_context":
                correlated_indices.append(i)

        # --- 1. Directional score (only direct sources) ---
        support_score = self._weighted_nli_avg(
            direct_support_indices, sources, nli_results
        )
        refute_score = self._weighted_nli_avg(
            direct_refute_indices, sources, nli_results
        )

        if verdict == "supported":
            directional = support_score * (1.0 - 0.5 * refute_score)
        elif verdict == "refuted":
            directional = refute_score * (1.0 - 0.5 * support_score)
        else:  # insufficient or mixed
            directional = max(support_score, refute_score) * 0.4

        # --- 2. LLM confidence ---
        # When LLM correctly identifies a verdict with direct evidence but
        # reports 0.0 confidence (common with Groq/Llama), apply a floor
        # derived from the directional evidence strength.
        llm_conf = max(0.0, min(1.0, llm_result.confidence))
        has_direct = len(direct_support_indices) > 0 or len(direct_refute_indices) > 0
        if llm_conf < 0.1 and has_direct and verdict in ("supported", "refuted"):
            llm_conf = max(llm_conf, directional * 0.5)

        # --- 3. Evidence quality (direct sources only) ---
        direct_indices = direct_support_indices + direct_refute_indices
        if direct_indices:
            quality = sum(
                sources[i].trust_score * sources[i].relevance_score
                for i in direct_indices
            ) / len(direct_indices)
        else:
            quality = 0.0

        # --- 4. Corroboration (distinct source types among directional agreement) ---
        if verdict == "supported":
            agreeing_indices = direct_support_indices
        elif verdict == "refuted":
            agreeing_indices = direct_refute_indices
        else:
            agreeing_indices = []

        agreeing_types = set(sources[i].source_type for i in agreeing_indices)
        if len(agreeing_types) >= 1:
            corroboration = min(
                1.0,
                (len(agreeing_types) - 1) / max(MIN_INDEPENDENT_SOURCES_FOR_BONUS, 1),
            )
        else:
            corroboration = 0.0

        # --- 5. Coverage (all source types, including correlated) ---
        all_types = set(s.source_type for s in sources)
        coverage = min(1.0, len(all_types) / 4)

        # --- Weighted combination ---
        raw = (
            CONFIDENCE_WEIGHT_DIRECTIONAL * directional
            + CONFIDENCE_WEIGHT_LLM * llm_conf
            + CONFIDENCE_WEIGHT_EVIDENCE_QUALITY * quality
            + CONFIDENCE_WEIGHT_CORROBORATION * corroboration
            + CONFIDENCE_WEIGHT_COVERAGE * coverage
        )

        # --- NEI capping ---
        if verdict in ("insufficient", "mixed"):
            raw = min(raw, CONFIDENCE_NEI_CEILING)

        # --- Calibrate ---
        calibrated = calibrate(raw)

        # --- Map to final verdict ---
        if verdict == "supported" and calibrated >= CONFIDENCE_VERIFIED_THRESHOLD:
            final_verdict = "verified"
        elif verdict == "refuted" and calibrated >= CONFIDENCE_REJECTED_THRESHOLD:
            final_verdict = "rejected"
        else:
            final_verdict = "not_enough_info"

        return ConfidenceOutput(
            overall_confidence=round(calibrated, 4),
            overall_verdict=final_verdict,
            support_score=round(support_score, 4),
            refute_score=round(refute_score, 4),
            evidence_quality=round(quality, 4),
            corroboration=round(corroboration, 4),
            coverage=round(coverage, 4),
            raw_confidence=round(raw, 4),
            debug={
                "directional": round(directional, 4),
                "llm_conf": round(llm_conf, 4),
                "llm_verdict": verdict,
                "direct_support_count": len(direct_support_indices),
                "direct_refute_count": len(direct_refute_indices),
                "correlated_count": len(correlated_indices),
                "agreeing_types": sorted(agreeing_types),
                "all_types": sorted(all_types),
            },
        )

    # ------------------------------------------------------------------
    # Edge / neighbor confidence
    # ------------------------------------------------------------------

    def compute_edge_confidence(
        self,
        source: Source,
        llm_classification: str,
        nli_result: Optional[NLIResult] = None,
    ) -> float:
        """
        Compute a confidence weight for a single source edge.

        Correlated context sources naturally get lower scores because
        LLM_CLASSIFICATION_STRENGTH["correlated_context"] = 0.3 vs 1.0
        for direct_support / direct_refute.

        Args:
            source:             The Source object.
            llm_classification: LLM classification string for this source.
            nli_result:         NLIResult for this source (if available).

        Returns:
            Edge confidence score clamped to [0.0, 1.0].
        """
        nli_conf = nli_result.confidence if nli_result else 0.3
        class_strength = LLM_CLASSIFICATION_STRENGTH.get(llm_classification, 0.1)

        edge = (
            EDGE_WEIGHT_NLI * nli_conf
            + EDGE_WEIGHT_TRUST * source.trust_score
            + EDGE_WEIGHT_RELEVANCE * source.relevance_score
            + EDGE_WEIGHT_LLM_CLASS * class_strength
        )

        return round(max(0.0, min(1.0, edge)), 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_nli_avg(
        indices: List[int],
        sources: List[Source],
        nli_results: Dict[int, NLIResult],
    ) -> float:
        """
        Weighted average of NLI confidence over a set of source indices.

        Weight per source = trust_score × relevance_score.
        Falls back to 0.5 confidence for sources without NLI results.
        """
        if not indices:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for i in indices:
            nli = nli_results.get(i)
            conf = nli.confidence if nli else 0.5
            w = sources[i].trust_score * sources[i].relevance_score
            w = max(w, 0.01)  # floor so zero-relevance sources still contribute
            weighted_sum += conf * w
            total_weight += w

        if total_weight == 0.0:
            return 0.0

        return weighted_sum / total_weight
