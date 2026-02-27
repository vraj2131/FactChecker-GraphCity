import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

from backend.app.preprocessing.snippet_extractor import (
    build_fever_evidence_snippets_dataframe,
    build_wiki_sentences_dataframe,
)
from backend.app.utils.constants import (
    DEFAULT_FEVER_WIKI_PAGES_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_WIKI_SAMPLE_PAGE_COUNT,
    FEVER_EVIDENCE_OUTPUT_FILENAME,
    FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME,
    FEVER_EVIDENCE_SNIPPETS_SAMPLE_OUTPUT_FILENAME,
    WIKI_SENTENCES_OUTPUT_FILENAME,
)
from backend.app.utils.file_io import load_parquet, save_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare wiki sentence text and join it with FEVER evidence rows."
    )

    parser.add_argument(
        "--wiki-pages-dir",
        type=str,
        default=str(DEFAULT_FEVER_WIKI_PAGES_DIR),
        help="Directory containing FEVER wiki-pages JSONL files.",
    )
    parser.add_argument(
        "--fever-evidence-path",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR / FEVER_EVIDENCE_OUTPUT_FILENAME),
        help="Path to processed FEVER evidence parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory to save processed wiki/evidence snippet parquet files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of wiki JSONL files to process (useful for debugging).",
    )
    parser.add_argument(
        "--sample-page-count",
        type=int,
        default=DEFAULT_WIKI_SAMPLE_PAGE_COUNT,
        help="Number of pages/files worth of content to approximate for sample output naming/debug workflow.",
    )
    
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of sample rows to print in the summary. Use 0 to print none.",
    )
    
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only counts/summary, not sample rows.",
    )

    return parser.parse_args()


def print_summary(
    wiki_sentences_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    snippets_df: pd.DataFrame,
    preview_rows: int = 5,
    summary_only: bool = False,
) -> None:
    print("\n" + "=" * 72)
    print("WIKI EVIDENCE PREPARATION SUMMARY")
    print("=" * 72)

    print(f"Wiki sentence rows:           {len(wiki_sentences_df):,}")
    print(f"FEVER evidence rows input:    {len(evidence_df):,}")
    print(f"Evidence snippet rows output: {len(snippets_df):,}")

    if snippets_df.empty:
        print("=" * 72 + "\n")
        return

    matched_count = int(snippets_df["has_sentence_text"].sum())
    unmatched_count = int((~snippets_df["has_sentence_text"]).sum())

    print(f"\nMatched evidence rows:        {matched_count:,}")
    print(f"Unmatched evidence rows:      {unmatched_count:,}")

    if summary_only or preview_rows <= 0:
        print("=" * 72 + "\n")
        return

    if not wiki_sentences_df.empty:
        print("\nSample wiki sentence rows:")
        print(wiki_sentences_df.head(preview_rows).to_string(index=False))

    print("\nSample matched snippet rows:")
    sample_matched = snippets_df[snippets_df["has_sentence_text"]].head(preview_rows)
    if not sample_matched.empty:
        print(
            sample_matched[
                [
                    "claim_id",
                    "label",
                    "page_title",
                    "sentence_id",
                    "sentence_text",
                ]
            ].to_string(index=False)
        )

    print("\nSample unmatched snippet rows:")
    sample_unmatched = snippets_df[~snippets_df["has_sentence_text"]].head(preview_rows)
    if not sample_unmatched.empty:
        print(
            sample_unmatched[
                [
                    "claim_id",
                    "label",
                    "page_title",
                    "sentence_id",
                ]
            ].to_string(index=False)
        )

    print("=" * 72 + "\n")


def build_sample_snippets_df(
    snippets_df: pd.DataFrame,
    sample_claim_count: int = 200,
) -> pd.DataFrame:
    """
    Lightweight sample subset for debugging. We keep up to N unique claims.
    """
    if snippets_df.empty:
        return snippets_df.copy()

    sample_claim_ids = snippets_df["claim_id"].dropna().unique()[:sample_claim_count]
    return snippets_df[snippets_df["claim_id"].isin(sample_claim_ids)].copy()


def main() -> None:
    args = parse_args()

    wiki_pages_dir = Path(args.wiki_pages_dir)
    fever_evidence_path = Path(args.fever_evidence_path)
    output_dir = Path(args.output_dir)

    if not wiki_pages_dir.exists():
        raise FileNotFoundError(f"Wiki pages directory not found: {wiki_pages_dir}")

    if not fever_evidence_path.exists():
        raise FileNotFoundError(f"Processed FEVER evidence parquet not found: {fever_evidence_path}")

    # ------------------------------------------------------------------
    # Build wiki sentence dataframe
    # ------------------------------------------------------------------
    print(f"[INFO] Reading wiki pages from: {wiki_pages_dir}")
    if args.max_files is not None:
        print(f"[INFO] Limiting wiki processing to first {args.max_files} files")

    wiki_sentences_df = build_wiki_sentences_dataframe(
        wiki_pages_dir=wiki_pages_dir,
        max_files=args.max_files,
    )

    # ------------------------------------------------------------------
    # Load processed FEVER evidence
    # ------------------------------------------------------------------
    print(f"[INFO] Loading FEVER evidence parquet: {fever_evidence_path}")
    fever_evidence_df = load_parquet(fever_evidence_path)

    # ------------------------------------------------------------------
    # Join evidence rows with actual sentence text
    # ------------------------------------------------------------------
    print("[INFO] Joining FEVER evidence with wiki sentence text...")
    evidence_snippets_df = build_fever_evidence_snippets_dataframe(
        fever_evidence_df=fever_evidence_df,
        wiki_sentences_df=wiki_sentences_df,
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    wiki_sentences_out = output_dir / WIKI_SENTENCES_OUTPUT_FILENAME
    evidence_snippets_out = output_dir / FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME
    evidence_snippets_sample_out = output_dir / FEVER_EVIDENCE_SNIPPETS_SAMPLE_OUTPUT_FILENAME

    save_parquet(wiki_sentences_df, wiki_sentences_out)
    save_parquet(evidence_snippets_df, evidence_snippets_out)

    sample_snippets_df = build_sample_snippets_df(evidence_snippets_df)
    save_parquet(sample_snippets_df, evidence_snippets_sample_out)

    print(f"[INFO] Saved wiki sentences parquet:        {wiki_sentences_out}")
    print(f"[INFO] Saved evidence snippets parquet:    {evidence_snippets_out}")
    print(f"[INFO] Saved evidence snippets sample:     {evidence_snippets_sample_out}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print_summary(
        wiki_sentences_df=wiki_sentences_df,
        evidence_df=fever_evidence_df,
        snippets_df=evidence_snippets_df,
        preview_rows=args.preview_rows,
        summary_only=args.summary_only,
    )

    print("[DONE] Wiki evidence preparation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)