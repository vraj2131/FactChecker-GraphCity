from pathlib import Path

# -------------------------------------------------------------------
# FEVER raw data default paths
# -------------------------------------------------------------------
DEFAULT_FEVER_TRAIN_PATH = Path("data/raw/fever/train.jsonl")
DEFAULT_FEVER_DEV_PATH = Path("data/raw/fever/dev.jsonl")
DEFAULT_FEVER_TEST_PATH = Path("data/raw/fever/test.jsonl")

# -------------------------------------------------------------------
# FEVER wiki-pages path
# -------------------------------------------------------------------
DEFAULT_FEVER_WIKI_PAGES_DIR = Path("data/raw/fever/wiki-pages")

# -------------------------------------------------------------------
# Processed output directory
# -------------------------------------------------------------------
DEFAULT_PROCESSED_DIR = Path("data/processed")

# -------------------------------------------------------------------
# Processed FEVER output files
# -------------------------------------------------------------------
FEVER_CLAIMS_OUTPUT_FILENAME = "fever_claims.parquet"
FEVER_EVIDENCE_OUTPUT_FILENAME = "fever_evidence.parquet"

FEVER_CLAIMS_SAMPLE_OUTPUT_FILENAME = "fever_claims_sample.parquet"
FEVER_EVIDENCE_SAMPLE_OUTPUT_FILENAME = "fever_evidence_sample.parquet"


# -------------------------------------------------------------------
# Evidence snippet phase outputs
# -------------------------------------------------------------------
WIKI_SENTENCES_OUTPUT_FILENAME = "wiki_sentences.parquet"
FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME = "fever_evidence_snippets.parquet"
FEVER_EVIDENCE_SNIPPETS_SAMPLE_OUTPUT_FILENAME = "fever_evidence_snippets_sample.parquet"

# -------------------------------------------------------------------
# Debug/sample defaults
# -------------------------------------------------------------------
DEFAULT_SAMPLE_CLAIM_COUNT = 200
DEFAULT_RANDOM_SEED = 42
DEFAULT_WIKI_SAMPLE_PAGE_COUNT = 500

# -------------------------------------------------------------------
# Phase 3: Embeddings + FAISS
# -------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_NORMALIZE_EMBEDDINGS = True
DEFAULT_FAISS_METRIC = "cosine"
DEFAULT_FAISS_TOP_K = 5

# -------------------------------------------------------------------
# Embedding/cache directories
# -------------------------------------------------------------------
DEFAULT_EMBEDDINGS_CACHE_DIR = Path("data/cache/embeddings")
DEFAULT_FAISS_ARTIFACTS_DIR = Path("data/artifacts/faiss")

# -------------------------------------------------------------------
# Phase 3 output filenames (claims only)
# -------------------------------------------------------------------
FEVER_CLAIM_EMBEDDINGS_FILENAME = "fever_claim_embeddings.npy"
FEVER_CLAIM_IDS_FILENAME = "fever_claim_ids.npy"
FEVER_CLAIM_METADATA_FILENAME = "fever_claim_metadata.pkl"
FEVER_CLAIMS_FAISS_INDEX_FILENAME = "fever_claims.index"

# -------------------------------------------------------------------
# Evidence snippet index output filenames
# -------------------------------------------------------------------
FEVER_EVIDENCE_EMBEDDINGS_FILENAME = "fever_evidence_embeddings.npy"
FEVER_EVIDENCE_METADATA_FILENAME = "fever_evidence_metadata.pkl"
FEVER_EVIDENCE_FAISS_INDEX_FILENAME = "fever_evidence.index"

# -------------------------------------------------------------------
# Phase 3 debug / test defaults
# -------------------------------------------------------------------
DEFAULT_PHASE3_USE_SAMPLE = True
DEFAULT_PHASE3_QUERY_COUNT = 5

# -------------------------------------------------------------------
# Phase 4: Retriever / cache settings
# -------------------------------------------------------------------
DEFAULT_RETRIEVER_MAX_RESULTS = 10
DEFAULT_RETRIEVER_TIMEOUT_SECONDS = 30

# -------------------------------------------------------------------
# Retriever cache directories
# -------------------------------------------------------------------
DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_RETRIEVAL_CACHE_DIR = Path("data/cache/retrieval_results")
DEFAULT_API_RESPONSE_CACHE_DIR = Path("data/cache/api_responses")

# -------------------------------------------------------------------
# Retriever source names
# -------------------------------------------------------------------
SOURCE_NAME_WIKIPEDIA = "wikipedia"
SOURCE_NAME_FACTCHECK = "factcheck"
SOURCE_NAME_GUARDIAN = "guardian"
SOURCE_NAME_NEWSAPI = "newsapi"
SOURCE_NAME_GDELT = "gdelt"
SOURCE_NAME_LIVEWIKI = "livewiki"

SUPPORTED_RETRIEVER_SOURCES = [
    SOURCE_NAME_WIKIPEDIA,
    SOURCE_NAME_FACTCHECK,
    SOURCE_NAME_GUARDIAN,
    SOURCE_NAME_NEWSAPI,
    SOURCE_NAME_GDELT,
    SOURCE_NAME_LIVEWIKI,
]

# -------------------------------------------------------------------
# Optional per-source defaults
# -------------------------------------------------------------------
DEFAULT_WIKIPEDIA_MAX_RESULTS = 10
DEFAULT_FACTCHECK_MAX_RESULTS = 10
DEFAULT_GUARDIAN_MAX_RESULTS = 10
DEFAULT_NEWSAPI_MAX_RESULTS = 10
DEFAULT_GDELT_MAX_RESULTS = 10
DEFAULT_LIVEWIKI_MAX_RESULTS = 5

# -------------------------------------------------------------------
# Phase 6: Snippet Extraction
# -------------------------------------------------------------------

# Maximum number of sentences to extract per source
DEFAULT_SNIPPET_MAX_SENTENCES = 3

# Hard character limit for the final snippet text stored on a node
DEFAULT_SNIPPET_MAX_CHARS = 500

# Minimum word count for a sentence to be considered usable
DEFAULT_SNIPPET_MIN_WORDS = 5

# Fall back to source title when no snippet/body is available
SNIPPET_FALLBACK_TO_TITLE = True

# -------------------------------------------------------------------
# Phase 5: Retrieval Orchestration
# -------------------------------------------------------------------

# Deduplication
DEDUP_JACCARD_THRESHOLD = 0.85  # snippet near-duplicate similarity threshold

# Ranking weights (must sum to 1.0)
RANKING_WEIGHT_TRUST = 0.4
RANKING_WEIGHT_RELEVANCE = 0.4
RANKING_WEIGHT_TYPE_PRIORITY = 0.2

# Source type priority scores (normalized 0–1)
SOURCE_TYPE_PRIORITY: dict = {
    "factcheck": 1.0,
    "guardian": 0.8,
    "livewiki": 0.65,
    "newsapi": 0.6,
    "wikipedia": 0.5,
    "gdelt": 0.3,
}
SOURCE_TYPE_PRIORITY_DEFAULT = 0.1

# Cache namespace key for orchestrated retrieval results
RETRIEVAL_ORCHESTRATION_CACHE_NAMESPACE = "retrieval_results"

# Max results fetched per individual retriever before merge/dedup
DEFAULT_PER_RETRIEVER_MAX_RESULTS = 10

# Maximum results kept per source type after ranking (prevents one source flooding the pool)
MAX_RESULTS_PER_SOURCE_TYPE = 3

# -------------------------------------------------------------------
# Phase 7: NLI
# -------------------------------------------------------------------

# HuggingFace model ID for NLI inference
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Device for model inference
NLI_DEVICE = "mps"

# Confidence threshold — predictions below this are treated as not_enough_info
NLI_CONFIDENCE_THRESHOLD = 0.35

# Cache namespace for NLI scored results
NLI_CACHE_NAMESPACE = "nli_results"

# Batch size for NLI inference over multiple snippets
NLI_BATCH_SIZE = 8

# Stronger confirmation model (FEVER+ANLI-trained) used in cascade to verify supports/refutes
NLI_CONFIRM_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Minimum claim-relevance score for a snippet to be kept after expansion.
# Requires at least 2 meaningful claim terms to overlap (single-term matches score ~0.06).
SNIPPET_MIN_RELEVANCE_SCORE = 0.15

# -------------------------------------------------------------------
# Phase 8: LLM
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# LLM model tiers
#   DEV  — small/fast, used for coding + prompt iteration (switch to PROD for final runs)
#   PROD — large/accurate, used for final benchmarks and demo-quality output
# To switch: set LLM_MODEL_NAME = LLM_PROD_MODEL_NAME
# -------------------------------------------------------------------
LLM_DEV_MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"   # ~3GB, fast on MPS (~10s inference)
LLM_PROD_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # ~16GB, best quality

# Active model — change this one line to switch tiers
LLM_MODEL_NAME = LLM_DEV_MODEL_NAME

# Fallback model used automatically if the primary model fails to load
LLM_FALLBACK_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Device for LLM inference — "mps" for Apple Silicon GPU, "cuda" for NVIDIA, "cpu" fallback
LLM_DEVICE = "mps"

# Max new tokens — 512 fits 5-source JSON with rationales comfortably
LLM_MAX_NEW_TOKENS = 512

# Cache namespace for LLM outputs
LLM_CACHE_NAMESPACE = "llm_outputs"

# Prompt version — bump this to bust the LLM cache when the prompt changes
LLM_PROMPT_VERSION = "v1"

# Max sources shown to the LLM — 5 for dev speed, 8 for prod quality
LLM_MAX_INPUT_SOURCES = 5

# -------------------------------------------------------------------
# Phase 8b: Groq API (cloud LLM — free tier, Llama 3.1 quality)
# -------------------------------------------------------------------

# Groq model IDs — see https://console.groq.com/docs/models
GROQ_MODEL_NAME = "llama-3.1-8b-instant"       # fast, free, Llama 3.1 8B quality
GROQ_PROD_MODEL_NAME = "llama-3.3-70b-versatile"  # best quality on Groq free tier

# Max tokens for the JSON response (same as local)
GROQ_MAX_TOKENS = 512

# Cache namespace for Groq outputs
GROQ_CACHE_NAMESPACE = "groq_outputs"

# -------------------------------------------------------------------
# Phase 9: Confidence Service
# -------------------------------------------------------------------

# --- Main Claim Confidence Weights (must sum to 1.0) ---
CONFIDENCE_WEIGHT_DIRECTIONAL = 0.35     # support vs refute signal strength
CONFIDENCE_WEIGHT_LLM = 0.25            # LLM's own confidence estimate
CONFIDENCE_WEIGHT_EVIDENCE_QUALITY = 0.20  # avg trust × relevance of direct sources
CONFIDENCE_WEIGHT_CORROBORATION = 0.15   # independent source agreement bonus
CONFIDENCE_WEIGHT_COVERAGE = 0.05        # breadth of retriever types used

# --- Edge/Neighbor Confidence Weights ---
EDGE_WEIGHT_NLI = 0.35
EDGE_WEIGHT_TRUST = 0.25
EDGE_WEIGHT_RELEVANCE = 0.25
EDGE_WEIGHT_LLM_CLASS = 0.15

# --- LLM Classification → Strength ---
LLM_CLASSIFICATION_STRENGTH: dict = {
    "direct_support": 1.0,
    "direct_refute": 1.0,
    "correlated_context": 0.3,
    "insufficient": 0.1,
}

# --- Calibration Breakpoints (piecewise linear) ---
# Each tuple is (raw_threshold, calibrated_value).
# Linear interpolation between adjacent breakpoints.
CALIBRATION_BREAKPOINTS: list = [
    (0.0, 0.05),   # floor: never output 0.0 (always some uncertainty)
    (0.3, 0.15),
    (0.5, 0.40),
    (0.7, 0.60),
    (0.85, 0.80),
    (1.0, 0.95),   # ceiling: never output 1.0 (epistemic humility)
]

# --- Verdict Thresholds ---
CONFIDENCE_VERIFIED_THRESHOLD = 0.50     # above this → "verified"
CONFIDENCE_REJECTED_THRESHOLD = 0.50     # above this → "rejected"
CONFIDENCE_NEI_CEILING = 0.45            # NEI/mixed scores capped here

# --- Corroboration ---
MIN_INDEPENDENT_SOURCES_FOR_BONUS = 2    # need 2+ distinct source types for bonus
