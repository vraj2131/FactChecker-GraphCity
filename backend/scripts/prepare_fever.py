import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.app.preprocessing.normalize_text import normalize_claim_text
from backend.app.utils.constants import (
    DEFAULT_FEVER_DEV_PATH,
    DEFAULT_FEVER_TEST_PATH,
    DEFAULT_FEVER_TRAIN_PATH,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_CLAIM_COUNT,
    FEVER_CLAIMS_OUTPUT_FILENAME,
    FEVER_CLAIMS_SAMPLE_OUTPUT_FILENAME,
    FEVER_EVIDENCE_OUTPUT_FILENAME,
    FEVER_EVIDENCE_SAMPLE_OUTPUT_FILENAME,
)
from backend.app.utils.file_io import ensure_dir, read_jsonl, save_parquet


def load_split(path: Path, split_name: str) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Missing file for split='{split_name}': {path}")
        return []

    print(f"[INFO] Loading {split_name}: {path}")
    rows = read_jsonl(path)
    print(f"[INFO] Loaded {len(rows):,} rows from {split_name}")
    return rows


def process_claim_rows(rows: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []

    for row in rows:
        processed.append(
            {
                "claim_id": row.get("id"),
                "split": split_name,
                "claim_text": row.get("claim"),
                "claim_text_normalized": normalize_claim_text(row.get("claim")),
                "label": row.get("label"),
                "verifiable": row.get("verifiable"),
                "num_evidence_sets": len(row.get("evidence", []) or []),
            }
        )

    return processed


def flatten_evidence_rows(rows: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    """
    Flatten FEVER evidence into row-wise format.

    FEVER evidence format:
    "evidence": [
        [[annotation_id, evidence_id, page_title, sentence_id]],
        [[annotation_id, evidence_id, page_title, sentence_id],
         [annotation_id, evidence_id, page_title, sentence_id]]
    ]
    """
    flattened: List[Dict[str, Any]] = []

    for row in rows:
        claim_id = row.get("id")
        claim_text = row.get("claim")
        label = row.get("label")
        verifiable = row.get("verifiable")
        evidence_sets = row.get("evidence", []) or []

        if not evidence_sets:
            continue

        for evidence_set_index, evidence_group in enumerate(evidence_sets):
            if not isinstance(evidence_group, list):
                continue

            for evidence_item_index, evidence_item in enumerate(evidence_group):
                if not isinstance(evidence_item, list) or len(evidence_item) != 4:
                    continue

                annotation_id, evidence_id, page_title, sentence_id = evidence_item

                flattened.append(
                    {
                        "claim_id": claim_id,
                        "split": split_name,
                        "claim_text": claim_text,
                        "claim_text_normalized": normalize_claim_text(claim_text),
                        "label": label,
                        "verifiable": verifiable,
                        "evidence_set_index": evidence_set_index,
                        "evidence_item_index": evidence_item_index,
                        "annotation_id": annotation_id,
                        "evidence_id": evidence_id,
                        "page_title": page_title,
                        "sentence_id": sentence_id,
                        "has_concrete_evidence": page_title is not None and sentence_id is not None,
                    }
                )

    return flattened


def build_sample_subsets(
    claims_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    sample_claim_count: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if claims_df.empty:
        return claims_df.copy(), evidence_df.iloc[0:0].copy()

    actual_n = min(sample_claim_count, len(claims_df))
    sample_claims_df = claims_df.sample(n=actual_n, random_state=random_seed).copy()

    sample_claim_ids = set(sample_claims_df["claim_id"].tolist())

    if evidence_df.empty:
        sample_evidence_df = evidence_df.iloc[0:0].copy()
    else:
        sample_evidence_df = evidence_df[evidence_df["claim_id"].isin(sample_claim_ids)].copy()

    return sample_claims_df, sample_evidence_df


def print_sanity_summary(claims_df: pd.DataFrame, evidence_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("FEVER PREPARATION SUMMARY")
    print("=" * 70)

    print(f"Total claims rows:    {len(claims_df):,}")
    print(f"Total evidence rows:  {len(evidence_df):,}")

    if not claims_df.empty:
        print("\nClaims by split:")
        print(claims_df["split"].value_counts(dropna=False).to_string())

        if "label" in claims_df.columns:
            print("\nClaims by label:")
            print(claims_df["label"].value_counts(dropna=False).to_string())

        if "verifiable" in claims_df.columns:
            print("\nClaims by verifiable:")
            print(claims_df["verifiable"].value_counts(dropna=False).to_string())

        broken_claims = claims_df["claim_text_normalized"].isna().sum()
        print(f"\nBroken normalized claim rows: {broken_claims:,}")

        print("\nSample 10 normalized claims:")
        sample_claim_texts = claims_df["claim_text_normalized"].dropna().head(10).tolist()
        for i, claim in enumerate(sample_claim_texts, start=1):
            print(f"{i:2d}. {claim}")

    if not evidence_df.empty:
        concrete_count = int(evidence_df["has_concrete_evidence"].sum())
        null_page_count = int(evidence_df["page_title"].isna().sum())
        null_sentence_count = int(evidence_df["sentence_id"].isna().sum())

        print(f"\nEvidence rows with concrete page+sentence: {concrete_count:,}")
        print(f"Evidence rows with null page_title:        {null_page_count:,}")
        print(f"Evidence rows with null sentence_id:       {null_sentence_count:,}")

        supported_example = evidence_df[evidence_df["label"] == "SUPPORTS"].head(5)
        refuted_example = evidence_df[evidence_df["label"] == "REFUTES"].head(5)

        if not supported_example.empty:
            print("\nSample evidence rows for SUPPORTS:")
            print(
                supported_example[
                    ["claim_id", "page_title", "sentence_id", "evidence_set_index", "evidence_item_index"]
                ].to_string(index=False)
            )

        if not refuted_example.empty:
            print("\nSample evidence rows for REFUTES:")
            print(
                refuted_example[
                    ["claim_id", "page_title", "sentence_id", "evidence_set_index", "evidence_item_index"]
                ].to_string(index=False)
            )

    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FEVER claims/evidence into processed parquet files.")

    parser.add_argument(
        "--train-path",
        type=str,
        default=str(DEFAULT_FEVER_TRAIN_PATH),
        help="Path to FEVER train JSONL file.",
    )
    parser.add_argument(
        "--dev-path",
        type=str,
        default=str(DEFAULT_FEVER_DEV_PATH),
        help="Path to FEVER dev JSONL file.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=str(DEFAULT_FEVER_TEST_PATH),
        help="Path to FEVER test JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory to store processed parquet outputs.",
    )
    parser.add_argument(
        "--sample-claim-count",
        type=int,
        default=DEFAULT_SAMPLE_CLAIM_COUNT,
        help="Number of claims to keep in the debug sample subset.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for debug subset sampling.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.random_seed)

    train_path = Path(args.train_path)
    dev_path = Path(args.dev_path)
    test_path = Path(args.test_path)
    output_dir = Path(args.output_dir)

    ensure_dir(output_dir)

    # ---------------------------------------------------------------
    # Load raw FEVER splits
    # ---------------------------------------------------------------
    train_rows = load_split(train_path, "train")
    dev_rows = load_split(dev_path, "dev")
    test_rows = load_split(test_path, "test")

    all_claim_rows: List[Dict[str, Any]] = []
    all_evidence_rows: List[Dict[str, Any]] = []

    for split_name, rows in [
        ("train", train_rows),
        ("dev", dev_rows),
        ("test", test_rows),
    ]:
        if not rows:
            continue

        all_claim_rows.extend(process_claim_rows(rows, split_name))
        all_evidence_rows.extend(flatten_evidence_rows(rows, split_name))

    claims_df = pd.DataFrame(all_claim_rows)
    evidence_df = pd.DataFrame(all_evidence_rows)

    # ---------------------------------------------------------------
    # Save processed outputs
    # ---------------------------------------------------------------
    claims_out = output_dir / FEVER_CLAIMS_OUTPUT_FILENAME
    evidence_out = output_dir / FEVER_EVIDENCE_OUTPUT_FILENAME

    save_parquet(claims_df, claims_out)
    save_parquet(evidence_df, evidence_out)

    print(f"[INFO] Saved claims parquet:   {claims_out}")
    print(f"[INFO] Saved evidence parquet: {evidence_out}")

    # ---------------------------------------------------------------
    # Save sample subsets for debugging
    # ---------------------------------------------------------------
    sample_claims_df, sample_evidence_df = build_sample_subsets(
        claims_df=claims_df,
        evidence_df=evidence_df,
        sample_claim_count=args.sample_claim_count,
        random_seed=args.random_seed,
    )

    sample_claims_out = output_dir / FEVER_CLAIMS_SAMPLE_OUTPUT_FILENAME
    sample_evidence_out = output_dir / FEVER_EVIDENCE_SAMPLE_OUTPUT_FILENAME

    save_parquet(sample_claims_df, sample_claims_out)
    save_parquet(sample_evidence_df, sample_evidence_out)

    print(f"[INFO] Saved sample claims parquet:   {sample_claims_out}")
    print(f"[INFO] Saved sample evidence parquet: {sample_evidence_out}")

    # ---------------------------------------------------------------
    # Print sanity summary
    # ---------------------------------------------------------------
    print_sanity_summary(claims_df, evidence_df)

    print("[DONE] FEVER preparation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)