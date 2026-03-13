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

SUPPORTED_RETRIEVER_SOURCES = [
    SOURCE_NAME_WIKIPEDIA,
    SOURCE_NAME_FACTCHECK,
    SOURCE_NAME_GUARDIAN,
    SOURCE_NAME_NEWSAPI,
    SOURCE_NAME_GDELT,
]

# -------------------------------------------------------------------
# Optional per-source defaults
# -------------------------------------------------------------------
DEFAULT_WIKIPEDIA_MAX_RESULTS = 10
DEFAULT_FACTCHECK_MAX_RESULTS = 10
DEFAULT_GUARDIAN_MAX_RESULTS = 10
DEFAULT_NEWSAPI_MAX_RESULTS = 10
DEFAULT_GDELT_MAX_RESULTS = 10
