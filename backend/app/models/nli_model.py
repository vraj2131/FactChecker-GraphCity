import logging
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.app.utils.constants import (
    NLI_BATCH_SIZE,
    NLI_CONFIDENCE_THRESHOLD,
    NLI_DEVICE,
    NLI_MODEL_NAME,
)

logger = logging.getLogger(__name__)

# Mapping from HuggingFace model label names → project label names
_LABEL_MAP = {
    "entailment": "supports",
    "contradiction": "refutes",
    "neutral": "not_enough_info",
}


@dataclass
class NLIResult:
    """
    Result of one NLI inference call for a single (claim, snippet) pair.
    """
    label: str           # "supports" / "refutes" / "not_enough_info"
    confidence: float    # softmax probability of the winning label [0, 1]
    scores: Dict[str, float] = field(default_factory=dict)  # all three label probabilities


class NLIModel:
    """
    Thin wrapper around a HuggingFace NLI cross-encoder model.

    Responsibilities:
    - load tokenizer + model
    - accept a claim and a batch of evidence snippets
    - return NLIResult per snippet (label + confidence + all scores)

    Label mapping:
        entailment    → supports
        contradiction → refutes
        neutral       → not_enough_info

    If the highest-scoring label has confidence < NLI_CONFIDENCE_THRESHOLD,
    the result is overridden to not_enough_info.
    """

    def __init__(
        self,
        model_name: str = NLI_MODEL_NAME,
        device: str = NLI_DEVICE,
    ) -> None:
        self.model_name = model_name
        self.device = device

        logger.info("Loading NLI model: %s on device=%s", model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, dtype=torch.float32
        )
        self.model.to(device)
        self.model.eval()

        # Read label order from model config — never hardcode indices
        self._id2label: Dict[int, str] = self.model.config.id2label
        logger.info("NLI model labels: %s", self._id2label)

    def predict(self, claim: str, snippets: List[str]) -> List[NLIResult]:
        """
        Run NLI inference for a claim against a list of evidence snippets.

        Args:
            claim:    The claim text (hypothesis).
            snippets: List of evidence snippet strings (premises).

        Returns:
            List[NLIResult] of the same length as snippets.

        Raises:
            ValueError: If claim is empty or snippets is empty.
        """
        if not claim or not claim.strip():
            raise ValueError("claim cannot be empty.")
        if not snippets:
            raise ValueError("snippets cannot be empty.")

        results: List[NLIResult] = []

        for batch_start in range(0, len(snippets), NLI_BATCH_SIZE):
            batch = snippets[batch_start: batch_start + NLI_BATCH_SIZE]
            batch_results = self._predict_batch(claim, batch)
            results.extend(batch_results)

        return results

    def _predict_batch(self, claim: str, snippets: List[str]) -> List[NLIResult]:
        """Run NLI on a single batch of snippets."""
        # NLI convention: (premise, hypothesis) — snippet is premise, claim is hypothesis
        encoding = self.tokenizer(
            snippets,
            [claim] * len(snippets),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        probs = torch.softmax(outputs.logits, dim=-1).cpu().tolist()

        results: List[NLIResult] = []
        for prob_row in probs:
            # Build scores dict using label names from model config
            scores: Dict[str, float] = {}
            for idx, prob in enumerate(prob_row):
                hf_label = self._id2label[idx].lower()
                project_label = _LABEL_MAP.get(hf_label, "not_enough_info")
                scores[project_label] = round(float(prob), 4)

            best_label = max(scores, key=lambda k: scores[k])
            confidence = scores[best_label]

            # Apply confidence threshold
            if confidence < NLI_CONFIDENCE_THRESHOLD:
                best_label = "not_enough_info"

            results.append(NLIResult(
                label=best_label,
                confidence=round(confidence, 4),
                scores=scores,
            ))

        return results
