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
NLI_DEVICE = "cpu"

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
