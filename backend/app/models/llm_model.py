import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from huggingface_hub import login as hf_login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load HF_TOKEN from .env and authenticate if present
load_dotenv()
_hf_token = os.getenv("HF_TOKEN", "").strip()
if _hf_token:
    hf_login(token=_hf_token, add_to_git_credential=False)
    logging.getLogger(__name__).info("HuggingFace: authenticated via HF_TOKEN")

from backend.app.schemas.source_schema import Source
from backend.app.services.cache_service import CacheService
from backend.app.utils.constants import (
    DEFAULT_CACHE_DIR,
    GROQ_BACKOFF_BASE_S,
    GROQ_BACKOFF_JITTER_S,
    GROQ_BACKOFF_MAX_S,
    GROQ_CACHE_NAMESPACE,
    GROQ_DAILY_QUOTA_RESET_MIN_S,
    GROQ_MAX_TOKENS,
    GROQ_MODEL_NAME,
    GROQ_RATE_LIMIT_MAX_ATTEMPTS,
    GROQ_REQUEST_TIMEOUT_S,
    GROQ_SDK_MAX_RETRIES,
    LLM_CACHE_NAMESPACE,
    LLM_DEVICE,
    LLM_FALLBACK_MODEL_NAME,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    LLM_PROMPT_VERSION,
)
from backend.app.utils.hashing import build_cache_key, stable_hash_object

# Absolute path to data/cache — resolved relative to this source file so it
# works regardless of the working directory (scripts, notebooks, FastAPI).
_CACHE_BASE = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cache"

logger = logging.getLogger(__name__)


class GroqDailyQuotaExhausted(Exception):
    """Raised when Groq's per-DAY token budget is spent (not the per-minute one).

    Carries `reset_seconds` so a long-running batch can sleep until the quota
    window rolls over instead of burning its retry attempts on a wait that
    cannot succeed for hours.
    """

    def __init__(self, message: str, reset_seconds: float) -> None:
        super().__init__(message)
        self.reset_seconds = reset_seconds


def _parse_duration(value: str) -> Optional[float]:
    """Parse Groq duration strings like '7.66s', '2m59.56s', '1h2m3s', '500ms'."""
    if not value:
        return None
    text = str(value).strip().lower()
    # Bare number => seconds (the plain `retry-after` form)
    try:
        return float(text)
    except ValueError:
        pass
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", text)
    if not matches:
        return None
    unit_seconds = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(num) * unit_seconds[unit] for num, unit in matches)


def _rate_limit_wait(exc: Exception) -> tuple[Optional[float], bool]:
    """Extract (wait_seconds, is_daily) from a Groq RateLimitError.

    Groq reports its own reset window in response headers; honouring that is
    what turns a hard failure into a short, correctly-sized pause. Returns
    (None, False) when no usable hint is present so the caller falls back to
    exponential backoff.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}

    def _get(name: str) -> str:
        try:
            return headers.get(name) or ""
        except Exception:
            return ""

    waits = []
    for header in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        parsed = _parse_duration(_get(header))
        if parsed is not None:
            waits.append(parsed)
    retry_after_ms = _parse_duration(_get("retry-after-ms"))
    if retry_after_ms is not None:
        waits.append(retry_after_ms / 1000.0 if retry_after_ms > 1000 else retry_after_ms)

    if not waits:
        # Fall back to the message body, which usually restates the window.
        parsed = _parse_duration_from_message(str(exc))
        if parsed is None:
            return None, False
        waits.append(parsed)

    wait = max(waits)
    # A multi-minute reset means the daily budget is gone, not the per-minute one.
    is_daily = wait >= GROQ_DAILY_QUOTA_RESET_MIN_S or "per day" in str(exc).lower()
    return wait, is_daily


def _parse_duration_from_message(message: str) -> Optional[float]:
    """Pull a 'try again in 2m59.56s' style hint out of an error message."""
    match = re.search(r"try again in\s+([0-9hms.\s]+)", message, re.IGNORECASE)
    return _parse_duration(match.group(1)) if match else None


_VALID_CLASSIFICATIONS = {
    "direct_support",
    "direct_refute",
    "correlated_context",
    "insufficient",
}
_VALID_VERDICTS = {"supported", "refuted", "insufficient", "mixed"}


@dataclass
class SourceClassification:
    """LLM classification result for a single source."""
    index: int
    classification: str   # direct_support / direct_refute / correlated_context / insufficient
    rationale: str


@dataclass
class NodeLink:
    """LLM-identified relationship between two evidence nodes (source-to-source)."""
    from_index: int   # 1-based source index
    to_index: int     # 1-based source index
    relation: str     # corroborates / contradicts / provides_context


@dataclass
class LLMResult:
    """Full LLM response for a claim against a set of sources."""
    sources: List[SourceClassification] = field(default_factory=list)
    overall_verdict: str = "insufficient"   # supported / refuted / insufficient / mixed
    confidence: float = 0.0
    short_explanation: str = ""
    best_source_index: int = 1
    node_links: List[NodeLink] = field(default_factory=list)


class LLMModel:
    """
    Thin wrapper around a HuggingFace causal LLM for fact-checking classification.

    Responsibilities:
    - Load tokenizer + model once at init
    - Build a structured prompt (system + user) for each claim
    - Generate and parse a JSON response into LLMResult
    - Retry once on JSON parse failure with a stricter instruction

    The system prompt is loaded from:
        backend/app/prompts/classify_sources_{LLM_PROMPT_VERSION}.txt

    NLI stance hints from Phase 7 are included in the source listing so the
    LLM has prior signal but can override it with finer-grained labels.
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        device: str = LLM_DEVICE,
        fallback_model_name: str = LLM_FALLBACK_MODEL_NAME,
    ) -> None:
        self.device = device

        # Load system prompt from the versioned .txt file
        prompt_path = (
            Path(__file__).parent.parent
            / "prompts"
            / f"classify_sources_{LLM_PROMPT_VERSION}.txt"
        )
        self.system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        logger.info("LLMModel: loaded system prompt from %s", prompt_path)

        # Try primary model, fall back automatically on any load failure
        self.tokenizer, self.model, self.model_name = self._load_with_fallback(
            model_name, fallback_model_name, device
        )
        self.model.eval()
        print(f"[LLM] Active model: {self.model_name}")

        self._cache = CacheService(_CACHE_BASE)
        self._cache_namespace = LLM_CACHE_NAMESPACE

    def _load_with_fallback(
        self,
        primary: str,
        fallback: str,
        device: str,
    ):
        """Try to load primary model; on failure load fallback instead."""
        for attempt, name in enumerate(
            [m for m in [primary, fallback] if m], start=1
        ):
            try:
                tokenizer, model = self._load_model(name, device)
                if attempt > 1:
                    print(f"[LLM] Using fallback model: {name}")
                return tokenizer, model, name
            except Exception as exc:
                label = "Primary" if attempt == 1 else "Fallback"
                print(f"[LLM] {label} model '{name}' failed to load: {exc}")
                logger.warning("LLMModel: %s model '%s' failed: %s", label.lower(), name, exc)
                if attempt == 1 and fallback:
                    print(f"[LLM] Switching to fallback model: {fallback}")
                else:
                    raise RuntimeError(
                        f"Both primary ('{primary}') and fallback ('{fallback}') models failed to load."
                    ) from exc

    @staticmethod
    def _load_model(model_name: str, device: str):
        """Load tokenizer + model for a given model name."""
        print(f"[LLM] Loading tokenizer: {model_name}")
        print(f"      (First run downloads model weights — cached permanently after that)")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("[LLM] Tokenizer ready.")

        # float16 on GPU (MPS/CUDA), float32 on CPU
        dtype = torch.float32 if device == "cpu" else torch.float16
        print(f"[LLM] Loading model weights (dtype={dtype}, device={device})...")
        print(f"      This takes 3-10 min on first load (cached after that).")
        # device_map is CUDA-only — for MPS and CPU we load then move manually
        use_device_map = device.startswith("cuda")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if use_device_map else None,
        )
        if not use_device_map:
            model.to(device)
        print(f"[LLM] Model weights loaded.")
        return tokenizer, model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(self, claim: str, sources: List[Source], use_cache: bool = True) -> LLMResult:
        """
        Classify a list of sources against a claim.

        Args:
            claim:     The claim text to verify.
            sources:   List of Source objects (already NLI-classified — stance_hint is used).
            use_cache: If True, check file cache before running inference and save the result.

        Returns:
            LLMResult with per-source classifications and an overall verdict.

        Raises:
            ValueError: If JSON parsing fails on both the initial attempt and retry.
        """
        if not claim or not claim.strip():
            raise ValueError("claim cannot be empty.")
        if not sources:
            raise ValueError("sources cannot be empty.")

        # --- Cache lookup ---
        cache_key = _llm_cache_key(self.model_name, claim, sources)
        if use_cache:
            cached = self._cache.load(self._cache_namespace, cache_key)
            if cached is not None:
                logger.info("LLMModel: cache hit for claim='%s'", claim[:60])
                return _deserialize_llm_result(cached)

        user_message = self._build_user_message(claim, sources)

        # First attempt
        raw_output = self._generate(user_message)
        try:
            result = self._parse_json(raw_output, len(sources))
        except (ValueError, KeyError) as exc:
            logger.warning("LLMModel: first parse attempt failed (%s), retrying.", exc)

            # Retry with a stricter instruction appended
            retry_user_message = (
                user_message
                + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                "Output ONLY the JSON object — nothing before or after it."
            )
            raw_output_retry = self._generate(retry_user_message)
            try:
                result = self._parse_json(raw_output_retry, len(sources))
            except (ValueError, KeyError) as exc2:
                raise ValueError(
                    f"LLMModel: failed to parse JSON response after retry. Last error: {exc2}\n"
                    f"Raw output: {raw_output_retry[:500]}"
                ) from exc2

        # --- Cache save ---
        if use_cache:
            try:
                self._cache.save(self._cache_namespace, cache_key, _serialize_llm_result(result))
            except Exception as e:
                logger.warning("LLMModel: failed to save cache: %s", e)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, claim: str, sources: List[Source]) -> str:
        """Build the user-turn message: claim + numbered source list."""
        return _build_user_message(claim, sources)

    def _generate(self, user_message: str) -> str:
        """Apply chat template, generate tokens, decode output."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # apply_chat_template returns a raw tensor for Llama but BatchEncoding for some models (e.g. Qwen)
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        input_ids = input_ids.to(self.device)

        attention_mask = torch.ones_like(input_ids)
        print(f"[LLM] Generating... (prompt={input_ids.shape[-1]} tokens, max_new={LLM_MAX_NEW_TOKENS})")
        # Suppress harmless 'generation flags not valid' warning from model's default generation config
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        print("[LLM] Generation complete.")

        # Decode only the newly generated tokens (strip the prompt)
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @staticmethod
    def _extract_json_str(text: str) -> str:
        return _extract_json_str(text)

    def _parse_json(self, raw_output: str, num_sources: int) -> LLMResult:
        """Parse the raw model output string into an LLMResult."""
        return _parse_llm_json(raw_output, num_sources)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _llm_cache_key(
    model_name: str,
    claim: str,
    sources: List[Source],
    context_claims: Optional[List] = None,
) -> str:
    """Stable cache key: model + claim + source fingerprint + context fingerprint + prompt version."""
    source_fingerprint = stable_hash_object([
        {"id": s.source_id, "snippet": s.snippet or "", "stance_hint": s.stance_hint or ""}
        for s in sources
    ])
    context_fingerprint = stable_hash_object([
        {"claim": c.claim_text, "verdict": c.verdict}
        for c in (context_claims or [])
    ])
    return build_cache_key(
        source_name=model_name,
        query=claim,
        source_fingerprint=source_fingerprint,
        context_fingerprint=context_fingerprint,
        prompt_version=LLM_PROMPT_VERSION,
    )


def _serialize_llm_result(result: "LLMResult") -> dict:
    """Convert LLMResult to a JSON-serializable dict."""
    return {
        "sources": [
            {"index": sc.index, "classification": sc.classification, "rationale": sc.rationale}
            for sc in result.sources
        ],
        "overall_verdict": result.overall_verdict,
        "confidence": result.confidence,
        "short_explanation": result.short_explanation,
        "best_source_index": result.best_source_index,
        "node_links": [
            {"from": nl.from_index, "to": nl.to_index, "relation": nl.relation}
            for nl in result.node_links
        ],
    }


def _deserialize_llm_result(data: dict) -> "LLMResult":
    """Reconstruct LLMResult from a cached dict."""
    return LLMResult(
        sources=[
            SourceClassification(
                index=sc["index"],
                classification=sc["classification"],
                rationale=sc["rationale"],
            )
            for sc in data.get("sources", [])
        ],
        overall_verdict=data.get("overall_verdict", "insufficient"),
        confidence=float(data.get("confidence") or 0.0),
        short_explanation=str(data.get("short_explanation") or ""),
        best_source_index=int(data.get("best_source_index") or 1),
        node_links=[
            NodeLink(
                from_index=int(nl.get("from", 0)),
                to_index=int(nl.get("to", 0)),
                relation=str(nl.get("relation", "corroborates")),
            )
            for nl in data.get("node_links", [])
            if nl.get("from") and nl.get("to")
        ],
    )


# ---------------------------------------------------------------------------
# Shared JSON parsing helpers (used by both LLMModel and GroqLLMModel)
# ---------------------------------------------------------------------------


def _extract_json_str(text: str) -> str:
    """Strip any preamble/postamble outside the outermost JSON object."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output.")
    return text[start: end + 1]


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before a closing brace/bracket.

    `[{...}, {...},]` is valid in JS but not JSON, and small models emit it
    routinely — it was the single most common cause of hard classification
    failures. Commas inside string literals are left alone.
    """
    out: List[str] = []
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            continue

        if ch == ",":
            # Look ahead past whitespace: a closer here means this comma is stray.
            j = i + 1
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            if j < len(text) and text[j] in "}]":
                continue   # drop it
        out.append(ch)

    return "".join(out)


def _repair_truncated_json(text: str) -> Optional[str]:
    """Best-effort repair of a response the model cut off mid-object.

    When generation stops early, the trailing text is a partial object and
    `_extract_json_str`'s rfind("}") latches onto an *inner* closing brace,
    yielding malformed JSON. Rather than lose the whole classification, walk
    the text tracking string/escape state and bracket depth, rewind to the
    last completed element, and close the still-open containers.

    Returns repaired JSON text, or None if nothing salvageable was found.
    """
    start = text.find("{")
    if start == -1:
        return None

    stack: List[str] = []
    in_string = False
    escaped = False
    last_safe: Optional[int] = None   # index just past the last complete element

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            # A complete element just closed while still inside a container.
            if stack:
                last_safe = i + 1
        elif ch == "," and len(stack) <= 2:
            last_safe = i   # comma at shallow depth: everything before is whole

    if last_safe is None or not stack:
        return None

    salvaged = text[start:last_safe].rstrip().rstrip(",")

    # Re-derive what remains open after the truncation point.
    stack = []
    in_string = False
    escaped = False
    for ch in salvaged:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()

    closers = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    return salvaged + closers


def _parse_llm_json(raw_output: str, num_sources: int) -> LLMResult:
    """Parse raw model output string into an LLMResult (shared by local + Groq)."""
    json_str = _extract_json_str(raw_output)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        data = None

        # Most failures are a stray trailing comma before a closer — cheap to fix.
        cleaned = _strip_trailing_commas(json_str)
        if cleaned != json_str:
            try:
                data = json.loads(cleaned)
                logger.info("Recovered LLM JSON by stripping trailing comma(s).")
            except json.JSONDecodeError:
                data = None

        # Otherwise the response was cut off mid-object; salvage what completed
        # rather than discarding an expensive call outright.
        if data is None:
            repaired = _repair_truncated_json(_strip_trailing_commas(raw_output))
            if repaired is not None:
                try:
                    data = json.loads(repaired)
                    logger.warning(
                        "Recovered truncated LLM JSON: salvaged %d of %d chars",
                        len(repaired), len(json_str),
                    )
                except json.JSONDecodeError:
                    data = None

        if data is None:
            raise ValueError(f"JSON decode error: {exc}") from exc

    source_classifications: List[SourceClassification] = []
    for entry in data.get("sources", []):
        idx = int(entry.get("index", 0))
        classification = str(entry.get("classification", "insufficient")).lower()
        if classification not in _VALID_CLASSIFICATIONS:
            classification = "insufficient"
        rationale = str(entry.get("rationale", "")).strip()
        source_classifications.append(
            SourceClassification(index=idx, classification=classification, rationale=rationale)
        )

    verdict = str(data.get("overall_verdict", "insufficient")).lower()
    if verdict not in _VALID_VERDICTS:
        verdict = "insufficient"

    confidence = float(data.get("confidence") or 0.0)
    confidence = max(0.0, min(1.0, confidence))

    short_explanation = str(data.get("short_explanation") or "").strip()

    best_source_index = int(data.get("best_source_index") or 1)
    best_source_index = max(1, min(best_source_index, num_sources))

    _VALID_RELATIONS = {"corroborates", "contradicts", "provides_context"}
    node_links: List[NodeLink] = []
    for nl in data.get("node_links", []) or []:
        try:
            from_idx = int(nl.get("from", 0))
            to_idx   = int(nl.get("to", 0))
            relation = str(nl.get("relation", "corroborates")).lower().strip()
            if relation not in _VALID_RELATIONS:
                relation = "corroborates"
            # Skip self-links and out-of-range indices
            if from_idx == to_idx or from_idx < 1 or to_idx < 1:
                continue
            if from_idx > num_sources or to_idx > num_sources:
                continue
            node_links.append(NodeLink(from_index=from_idx, to_index=to_idx, relation=relation))
        except (TypeError, ValueError):
            continue

    return LLMResult(
        sources=source_classifications,
        overall_verdict=verdict,
        confidence=confidence,
        short_explanation=short_explanation,
        best_source_index=best_source_index,
        node_links=node_links,
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[LLMModel] = None


def get_llm_model(
    model_name: str = LLM_MODEL_NAME,
    device: str = LLM_DEVICE,
    fallback_model_name: str = LLM_FALLBACK_MODEL_NAME,
) -> LLMModel:
    """
    Return the shared LLMModel instance, loading it on first call.

    Within a single process (FastAPI server, Jupyter notebook) the model is
    loaded once and reused for every subsequent call — no repeated 15GB reads.

    Args:
        model_name:          Primary HuggingFace model ID.
        device:              torch device string ('cpu', 'cuda', etc.)
        fallback_model_name: Model to try if primary fails to load.

    Returns:
        The singleton LLMModel instance.
    """
    global _instance
    if _instance is None:
        _instance = LLMModel(
            model_name=model_name,
            device=device,
            fallback_model_name=fallback_model_name,
        )
    return _instance


# ---------------------------------------------------------------------------
# Groq API LLM (cloud — free tier, Llama 3.1 quality, instant inference)
# ---------------------------------------------------------------------------


class GroqLLMModel:
    """
    Cloud LLM wrapper using the Groq API (free tier).

    Drop-in replacement for LLMModel — same classify() interface, same
    LLMResult output.  No local GPU needed; inference takes ~1-2 seconds.

    Requires GROQ_API_KEY in environment / .env file.
    Sign up free at https://console.groq.com

    The system prompt is shared with the local LLMModel (same .txt file).
    """

    def __init__(
        self,
        model_name: str = GROQ_MODEL_NAME,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name

        # Load API key from argument → env → .env file
        load_dotenv()
        resolved_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
        if not resolved_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file or environment. "
                "Get a free key at https://console.groq.com"
            )
        self._client = Groq(
            api_key=resolved_key,
            max_retries=GROQ_SDK_MAX_RETRIES,
            timeout=GROQ_REQUEST_TIMEOUT_S,
        )

        # Reuse same system prompt as local model
        prompt_path = (
            Path(__file__).parent.parent
            / "prompts"
            / f"classify_sources_{LLM_PROMPT_VERSION}.txt"
        )
        self.system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        logger.info("GroqLLMModel: loaded system prompt from %s", prompt_path)
        print(f"[Groq] Model: {self.model_name}")

        self._cache = CacheService(_CACHE_BASE)
        self._cache_namespace = GROQ_CACHE_NAMESPACE

    def classify(
        self,
        claim: str,
        sources: List[Source],
        use_cache: bool = True,
        context_claims: Optional[List] = None,
    ) -> LLMResult:
        """
        Classify sources against a claim using the Groq API.

        Same signature and return type as LLMModel.classify().

        Args:
            use_cache:      If True, check file cache before hitting the API and save the result.
            context_claims: Optional list of ContextClaim objects from prior verifications.
        """
        if not claim or not claim.strip():
            raise ValueError("claim cannot be empty.")
        if not sources:
            raise ValueError("sources cannot be empty.")

        # --- Cache lookup ---
        cache_key = _llm_cache_key(self.model_name, claim, sources, context_claims)
        if use_cache:
            cached = self._cache.load(self._cache_namespace, cache_key)
            if cached is not None:
                logger.info("GroqLLMModel: cache hit for claim='%s'", claim[:60])
                return _deserialize_llm_result(cached)

        user_message = _build_user_message(claim, sources, context_claims)

        # First attempt
        raw_output = self._call_api(user_message)
        try:
            result = _parse_llm_json(raw_output, len(sources))
        except (ValueError, KeyError) as exc:
            logger.warning("GroqLLMModel: first parse attempt failed (%s), retrying.", exc)

            # Retry with stricter instruction
            retry_user_message = (
                user_message
                + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                "Output ONLY the JSON object — nothing before or after it."
            )
            raw_output_retry = self._call_api(retry_user_message)
            try:
                result = _parse_llm_json(raw_output_retry, len(sources))
            except (ValueError, KeyError) as exc2:
                raise ValueError(
                    f"GroqLLMModel: failed to parse JSON after retry. Last error: {exc2}\n"
                    f"Raw output: {raw_output_retry[:500]}"
                ) from exc2

        # --- Cache save ---
        if use_cache:
            try:
                self._cache.save(self._cache_namespace, cache_key, _serialize_llm_result(result))
            except Exception as e:
                logger.warning("GroqLLMModel: failed to save cache: %s", e)

        return result

    def _call_api(self, user_message: str) -> str:
        """Send messages to Groq API and return the assistant reply text.

        Retries on rate limits and transient errors. Groq's own reset hint is
        honoured when present, otherwise exponential backoff with jitter.
        A spent DAILY token budget raises GroqDailyQuotaExhausted immediately
        so callers can sleep until rollover rather than retrying pointlessly.
        """
        print(f"[Groq] Generating... (model={self.model_name}, max_tokens={GROQ_MAX_TOKENS})")
        last_exc: Optional[Exception] = None

        for attempt in range(GROQ_RATE_LIMIT_MAX_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=GROQ_MAX_TOKENS,
                    temperature=0.0,
                )
                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    logger.warning(
                        "Groq stopped at max_tokens (%d) — response will be truncated. "
                        "Consider lowering LLM_MAX_INPUT_SOURCES.", GROQ_MAX_TOKENS,
                    )
                print("[Groq] Generation complete.")
                return choice.message.content or ""

            except RateLimitError as exc:
                last_exc = exc
                hinted_wait, is_daily = _rate_limit_wait(exc)
                if is_daily:
                    logger.error(
                        "Groq DAILY token quota exhausted — resets in %.0fs (%.1fh)",
                        hinted_wait or 0.0, (hinted_wait or 0.0) / 3600.0,
                    )
                    raise GroqDailyQuotaExhausted(
                        f"Groq daily token quota exhausted: {exc}",
                        reset_seconds=hinted_wait or 3600.0,
                    ) from exc

                wait = hinted_wait if hinted_wait is not None else min(
                    GROQ_BACKOFF_BASE_S * (2 ** attempt), GROQ_BACKOFF_MAX_S
                )
                wait = min(wait, GROQ_BACKOFF_MAX_S) + random.uniform(0, GROQ_BACKOFF_JITTER_S)
                if attempt == GROQ_RATE_LIMIT_MAX_ATTEMPTS - 1:
                    break
                logger.warning(
                    "Groq 429 (attempt %d/%d) — sleeping %.1fs",
                    attempt + 1, GROQ_RATE_LIMIT_MAX_ATTEMPTS, wait,
                )
                time.sleep(wait)

            except (APIConnectionError, APITimeoutError, InternalServerError) as exc:
                last_exc = exc
                if attempt == GROQ_RATE_LIMIT_MAX_ATTEMPTS - 1:
                    break
                wait = min(GROQ_BACKOFF_BASE_S * (2 ** attempt), GROQ_BACKOFF_MAX_S)
                logger.warning(
                    "Groq transient error %s (attempt %d/%d) — sleeping %.1fs",
                    type(exc).__name__, attempt + 1, GROQ_RATE_LIMIT_MAX_ATTEMPTS, wait,
                )
                time.sleep(wait)

        assert last_exc is not None
        raise last_exc


_SNIPPET_MAX_CHARS = 300   # ~75 tokens per source — keeps 20 sources under 6000 TPM


def _build_user_message(
    claim: str,
    sources: List[Source],
    context_claims: Optional[List] = None,
) -> str:
    """Shared user message builder (used by both LLMModel and GroqLLMModel)."""
    lines = []

    # Prior context section — injected before the claim when history exists
    if context_claims:
        lines.append("PRIOR VERIFICATION CONTEXT (supplementary — do not override current evidence):")
        for i, ctx in enumerate(context_claims, start=1):
            conf_pct = round(ctx.confidence * 100)
            verdict_label = ctx.verdict.upper().replace("_", " ")
            lines.append(f"[{i}] Claim: \"{ctx.claim_text}\"")
            lines.append(f"    Verdict: {verdict_label} ({conf_pct}% confidence)")
            if ctx.top_snippets:
                snippets_str = " | ".join(
                    s[:120] for s in ctx.top_snippets[:3] if s
                )
                lines.append(f"    Key evidence: {snippets_str}")
        lines.append("")

    lines += [f"CLAIM: {claim}", "", "SOURCES:"]
    for i, source in enumerate(sources, start=1):
        raw_snippet = (source.snippet or "").strip() or "[no snippet]"
        snippet = raw_snippet[:_SNIPPET_MAX_CHARS] + ("…" if len(raw_snippet) > _SNIPPET_MAX_CHARS else "")
        nli_hint = source.stance_hint or "none"
        lines.append(
            f"[{i}] title: {source.title}\n"
            f"    type: {source.source_type}\n"
            f"    nli_hint: {nli_hint}\n"
            f"    snippet: {snippet}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Groq singleton
# ---------------------------------------------------------------------------

_groq_instance: Optional[GroqLLMModel] = None


def get_groq_llm_model(
    model_name: str = GROQ_MODEL_NAME,
    api_key: Optional[str] = None,
) -> GroqLLMModel:
    """
    Return the shared GroqLLMModel instance (no heavy loading — just API client init).

    Args:
        model_name: Groq model ID (default: llama-3.1-8b-instant).
        api_key:    Optional explicit API key; falls back to GROQ_API_KEY env var.
    """
    global _groq_instance
    if _groq_instance is None:
        _groq_instance = GroqLLMModel(model_name=model_name, api_key=api_key)
    return _groq_instance
