import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from backend.app.models.embedding_model import EmbeddingModel
from backend.app.utils.constants import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_FAISS_ARTIFACTS_DIR,
    DEFAULT_FAISS_METRIC,
    DEFAULT_NORMALIZE_EMBEDDINGS,
    DEFAULT_PHASE3_USE_SAMPLE,
    DEFAULT_PROCESSED_DIR,
    FEVER_CLAIMS_OUTPUT_FILENAME,
    FEVER_CLAIMS_SAMPLE_OUTPUT_FILENAME,
    FEVER_CLAIM_EMBEDDINGS_FILENAME,
    FEVER_CLAIM_METADATA_FILENAME,
    FEVER_CLAIMS_FAISS_INDEX_FILENAME,
    FEVER_EVIDENCE_EMBEDDINGS_FILENAME,
    FEVER_EVIDENCE_METADATA_FILENAME,
    FEVER_EVIDENCE_FAISS_INDEX_FILENAME,
    FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME,
    FEVER_EVIDENCE_SNIPPETS_SAMPLE_OUTPUT_FILENAME,
)
from backend.app.utils.file_io import load_numpy, load_pickle
from backend.app.vectorstore.faiss_store import FaissStore
from backend.app.vectorstore.index_builder import ClaimIndexBuilder


def _log(msg: str) -> None:
    print(msg, flush=True)


class _Timer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        _log(f"[DEBUG] START: {self.label}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        _log(f"[DEBUG] END:   {self.label}  ({dt:.3f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FEVER FAISS indexes for claims or evidence snippets."
    )

    parser.add_argument(
        "--index-type",
        type=str,
        default="claims",
        choices=["claims", "evidence"],
        help="Which index to build/search.",
    )

    parser.add_argument(
        "--claims-path",
        type=str,
        default="",
        help="Optional explicit path to processed FEVER claims parquet.",
    )

    parser.add_argument(
        "--evidence-path",
        type=str,
        default="",
        help="Optional explicit path to processed FEVER evidence snippets parquet.",
    )

    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use sample parquet instead of full parquet.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help="SentenceTransformer model name.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_EMBEDDING_DEVICE,
        help="Device to load embedding model on (e.g. cpu, cuda).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_FAISS_METRIC,
        help="FAISS similarity metric: cosine or l2.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k nearest neighbors to inspect in sanity check.",
    )

    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(DEFAULT_FAISS_ARTIFACTS_DIR),
        help="Directory to save FAISS index and metadata.",
    )

    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="data/cache/embeddings",
        help="Directory to save embedding cache.",
    )

    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip nearest-neighbor sanity check entirely.",
    )

    parser.add_argument(
        "--sanity-queries",
        type=int,
        default=3,
        help="Number of query rows to run during sanity check.",
    )

    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Do not rebuild; only run sanity check on existing artifacts.",
    )

    return parser.parse_args()


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.index_type == "claims":
        if args.claims_path:
            return Path(args.claims_path)

        if args.use_sample or DEFAULT_PHASE3_USE_SAMPLE:
            return DEFAULT_PROCESSED_DIR / FEVER_CLAIMS_SAMPLE_OUTPUT_FILENAME

        return DEFAULT_PROCESSED_DIR / FEVER_CLAIMS_OUTPUT_FILENAME

    # evidence
    if args.evidence_path:
        return Path(args.evidence_path)

    if args.use_sample or DEFAULT_PHASE3_USE_SAMPLE:
        return DEFAULT_PROCESSED_DIR / FEVER_EVIDENCE_SNIPPETS_SAMPLE_OUTPUT_FILENAME

    return DEFAULT_PROCESSED_DIR / FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME


def resolve_output_paths(index_type: str, embeddings_dir: Path, artifacts_dir: Path) -> tuple[Path, Path, Path]:
    if index_type == "claims":
        return (
            embeddings_dir / FEVER_CLAIM_EMBEDDINGS_FILENAME,
            artifacts_dir / FEVER_CLAIM_METADATA_FILENAME,
            artifacts_dir / FEVER_CLAIMS_FAISS_INDEX_FILENAME,
        )

    return (
        embeddings_dir / FEVER_EVIDENCE_EMBEDDINGS_FILENAME,
        artifacts_dir / FEVER_EVIDENCE_METADATA_FILENAME,
        artifacts_dir / FEVER_EVIDENCE_FAISS_INDEX_FILENAME,
    )


def print_build_summary(build_info: dict) -> None:
    _log("\n" + "=" * 70)
    _log("BUILD SUMMARY")
    _log("=" * 70)
    _log(f"Index type:       {build_info['index_type']}")
    _log(f"Records indexed:  {build_info['num_records_indexed']:,}")
    _log(f"Embedding dim:    {build_info['embedding_dim']}")
    _log(f"Embeddings saved: {build_info['embeddings_output_path']}")
    _log(f"Metadata saved:   {build_info['metadata_output_path']}")
    _log(f"FAISS index saved: {build_info['faiss_index_output_path']}")
    _log("=" * 70 + "\n")


def run_sanity_check(
    embeddings_path: Path,
    metadata_path: Path,
    faiss_index_path: Path,
    embedding_dim: int,
    metric: str,
    top_k: int,
    num_queries: int,
) -> None:
    _log("\n[INFO] Running nearest-neighbor sanity check...\n")

    with _Timer("load_numpy(embeddings)"):
        embeddings = load_numpy(embeddings_path)
    _log(f"[DEBUG] embeddings dtype={embeddings.dtype}, shape={embeddings.shape}")

    with _Timer("load_pickle(metadata)"):
        metadata: List[dict] = load_pickle(metadata_path)
    _log(f"[DEBUG] metadata len={len(metadata)}")

    with _Timer("FaissStore.load(index)"):
        store = FaissStore.load(
            path=faiss_index_path,
            embedding_dim=embedding_dim,
            metric=metric,
        )
    _log(f"[DEBUG] faiss ntotal={store.ntotal()}")

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    query_count = min(num_queries, len(metadata), embeddings.shape[0])
    if query_count <= 0:
        raise ValueError("No queries available for sanity check.")

    query_embeddings = np.ascontiguousarray(
        embeddings[:query_count].astype(np.float32)
    )

    with _Timer(f"FAISS search (queries={query_count}, top_k={top_k})"):
        distances, indices = store.search(query_embeddings, top_k=top_k)

    _log(f"[DEBUG] distances shape={distances.shape}, indices shape={indices.shape}")

    for q_idx in range(query_count):
        query_meta = metadata[q_idx]
        _log("-" * 70)
        _log(f"QUERY {q_idx + 1} | row_index={query_meta.get('row_index')}")
        if query_meta.get("record_type") == "claim":
            _log(f"claim_id={query_meta.get('claim_id')}")
            _log(f"text: {query_meta.get('claim_text_normalized')}")
        else:
            _log(f"claim_id={query_meta.get('claim_id')}")
            _log(f"page_title={query_meta.get('page_title_clean')}")
            _log(f"sentence_id={query_meta.get('sentence_id')}")
            _log(f"text: {query_meta.get('snippet_text')}")

        _log("Top neighbors:")
        for rank in range(min(top_k, indices.shape[1])):
            n_idx = int(indices[q_idx, rank])
            score = float(distances[q_idx, rank])

            if 0 <= n_idx < len(metadata):
                neighbor = metadata[n_idx]
                if neighbor.get("record_type") == "claim":
                    _log(
                        f"  {rank+1}. idx={n_idx} score={score:.4f} "
                        f"claim_id={neighbor.get('claim_id')} "
                        f"text={neighbor.get('claim_text_normalized')}"
                    )
                else:
                    _log(
                        f"  {rank+1}. idx={n_idx} score={score:.4f} "
                        f"claim_id={neighbor.get('claim_id')} "
                        f"page_title={neighbor.get('page_title_clean')} "
                        f"sentence_id={neighbor.get('sentence_id')} "
                        f"text={neighbor.get('snippet_text')}"
                    )
            else:
                _log(f"  {rank+1}. idx={n_idx} score={score:.4f} (out of range)")

    _log("\n[DONE] Sanity check complete.\n")


def main() -> None:
    args = parse_args()

    input_path = resolve_input_path(args)
    artifacts_dir = Path(args.artifacts_dir)
    embeddings_dir = Path(args.embeddings_dir)

    embeddings_output_path, metadata_output_path, faiss_index_output_path = resolve_output_paths(
        index_type=args.index_type,
        embeddings_dir=embeddings_dir,
        artifacts_dir=artifacts_dir,
    )

    _log(f"[INFO] Index type: {args.index_type}")
    _log(f"[INFO] Input parquet: {input_path}")
    _log(f"[INFO] Embedding model: {args.model_name}")
    _log(f"[INFO] Device: {args.device}")
    _log(f"[INFO] Metric: {args.metric}")
    _log(f"[INFO] Artifacts dir: {artifacts_dir}")
    _log(f"[INFO] Embeddings dir: {embeddings_dir}")
    _log(f"[INFO] skip_sanity_check={args.skip_sanity_check}, sanity_queries={args.sanity_queries}, top_k={args.top_k}")
    _log(f"[INFO] no_rebuild={args.no_rebuild}")

    build_info = None

    if not args.no_rebuild:
        if not input_path.exists():
            raise FileNotFoundError(f"Input parquet not found: {input_path}")

        with _Timer("load EmbeddingModel"):
            embedding_model = EmbeddingModel(
                model_name=args.model_name,
                device=args.device,
            )

        builder = ClaimIndexBuilder(
            embedding_model=embedding_model,
            faiss_metric=args.metric,
        )

        if args.index_type == "claims":
            with _Timer("build_index_from_claims"):
                build_info = builder.build_index_from_claims(
                    claims_parquet_path=input_path,
                    embeddings_output_path=embeddings_output_path,
                    metadata_output_path=metadata_output_path,
                    faiss_index_output_path=faiss_index_output_path,
                    batch_size=args.batch_size,
                    normalize_embeddings=DEFAULT_NORMALIZE_EMBEDDINGS,
                    show_progress_bar=True,
                )
        else:
            with _Timer("build_index_from_evidence_snippets"):
                build_info = builder.build_index_from_evidence_snippets(
                    evidence_snippets_parquet_path=input_path,
                    embeddings_output_path=embeddings_output_path,
                    metadata_output_path=metadata_output_path,
                    faiss_index_output_path=faiss_index_output_path,
                    batch_size=args.batch_size,
                    normalize_embeddings=DEFAULT_NORMALIZE_EMBEDDINGS,
                    show_progress_bar=True,
                )

        print_build_summary(build_info)

    else:
        _log("[INFO] Skipping rebuild; will sanity-check existing artifacts.")

        if not embeddings_output_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {embeddings_output_path}")
        if not metadata_output_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_output_path}")
        if not faiss_index_output_path.exists():
            raise FileNotFoundError(f"Missing FAISS index file: {faiss_index_output_path}")

        with _Timer("infer embedding_dim from saved embeddings"):
            emb = load_numpy(embeddings_output_path)
            if emb.ndim == 1:
                embedding_dim = int(emb.shape[0])
            else:
                embedding_dim = int(emb.shape[1])

        build_info = {
            "index_type": args.index_type,
            "embedding_dim": embedding_dim,
        }

    if args.skip_sanity_check:
        _log("[INFO] Skipping sanity check as requested.")
        _log(f"[DONE] {args.index_type} index phase completed (build only).")
        return

    run_sanity_check(
        embeddings_path=embeddings_output_path,
        metadata_path=metadata_output_path,
        faiss_index_path=faiss_index_output_path,
        embedding_dim=build_info["embedding_dim"],
        metric=args.metric,
        top_k=args.top_k,
        num_queries=args.sanity_queries,
    )

    _log(f"[DONE] {args.index_type} index build + sanity check completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _log(f"\n[ERROR] {e}")
        sys.exit(1)