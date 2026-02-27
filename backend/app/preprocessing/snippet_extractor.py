from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backend.app.preprocessing.source_cleaner import (
    clean_page_title,
    clean_sentence_text,
)
from backend.app.utils.file_io import iter_jsonl, list_jsonl_files


def parse_wiki_page_lines(raw_lines: str) -> List[Dict]:
    """
    Parse FEVER wiki page `lines` field into sentence rows.

    FEVER wiki-pages typically store lines like:
        0\tSentence one\n
        1\tSentence two\n
        2\tSentence three

    Returns rows like:
        {
            "sentence_id": 0,
            "sentence_text": "Sentence one"
        }
    """
    if raw_lines is None:
        return []

    text = str(raw_lines).strip()
    if not text:
        return []

    rows: List[Dict] = []

    for raw_line in text.split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        parts = raw_line.split("\t", 1)
        if len(parts) != 2:
            continue

        sentence_id_raw, sentence_text_raw = parts

        try:
            sentence_id = int(sentence_id_raw)
        except ValueError:
            continue

        sentence_text = clean_sentence_text(sentence_text_raw)
        if sentence_text is None:
            continue

        rows.append(
            {
                "sentence_id": sentence_id,
                "sentence_text": sentence_text,
            }
        )

    return rows


def extract_wiki_sentences_from_record(record: Dict) -> List[Dict]:
    """
    Convert one FEVER wiki-page record into sentence-level rows.

    Expected record structure usually includes:
    - "id" or page title key
    - "lines"

    Returns rows like:
        {
            "page_title": ...,
            "page_title_clean": ...,
            "sentence_id": ...,
            "sentence_text": ...
        }
    """
    raw_page_title = record.get("id")
    page_title_clean = clean_page_title(raw_page_title)

    if page_title_clean is None:
        return []

    lines = record.get("lines")
    sentence_rows = parse_wiki_page_lines(lines)

    output_rows: List[Dict] = []
    for row in sentence_rows:
        output_rows.append(
            {
                "page_title": raw_page_title,
                "page_title_clean": page_title_clean,
                "sentence_id": row["sentence_id"],
                "sentence_text": row["sentence_text"],
            }
        )

    return output_rows


def build_wiki_sentences_dataframe(
    wiki_pages_dir: Path,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read FEVER wiki-pages JSONL files and build a sentence-level dataframe.

    Output columns:
    - page_title
    - page_title_clean
    - sentence_id
    - sentence_text
    """
    files = list_jsonl_files(wiki_pages_dir)
    if max_files is not None:
        files = files[:max_files]

    all_rows: List[Dict] = []

    for file_path in files:
        for record in iter_jsonl(file_path):
            sentence_rows = extract_wiki_sentences_from_record(record)
            all_rows.extend(sentence_rows)

    df = pd.DataFrame(all_rows)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "page_title",
                "page_title_clean",
                "sentence_id",
                "sentence_text",
            ]
        )

    df = df.drop_duplicates(
        subset=["page_title_clean", "sentence_id", "sentence_text"]
    ).reset_index(drop=True)

    return df


def build_fever_evidence_snippets_dataframe(
    fever_evidence_df: pd.DataFrame,
    wiki_sentences_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join processed FEVER evidence rows with actual wiki sentence text.

    Required columns in fever_evidence_df:
    - claim_id
    - split
    - claim_text
    - claim_text_normalized
    - label
    - verifiable
    - evidence_set_index
    - evidence_item_index
    - annotation_id
    - evidence_id
    - page_title
    - sentence_id

    Required columns in wiki_sentences_df:
    - page_title_clean
    - sentence_id
    - sentence_text
    """
    required_evidence_cols = {
        "claim_id",
        "split",
        "claim_text",
        "claim_text_normalized",
        "label",
        "verifiable",
        "evidence_set_index",
        "evidence_item_index",
        "annotation_id",
        "evidence_id",
        "page_title",
        "sentence_id",
    }
    missing_evidence = required_evidence_cols - set(fever_evidence_df.columns)
    if missing_evidence:
        raise ValueError(
            f"fever_evidence_df missing required columns: {sorted(missing_evidence)}"
        )

    required_wiki_cols = {"page_title_clean", "sentence_id", "sentence_text"}
    missing_wiki = required_wiki_cols - set(wiki_sentences_df.columns)
    if missing_wiki:
        raise ValueError(
            f"wiki_sentences_df missing required columns: {sorted(missing_wiki)}"
        )

    evidence_df = fever_evidence_df.copy()
    evidence_df["page_title_clean"] = evidence_df["page_title"].apply(clean_page_title)

    joined_df = evidence_df.merge(
        wiki_sentences_df[["page_title_clean", "sentence_id", "sentence_text"]],
        how="left",
        on=["page_title_clean", "sentence_id"],
    )

    joined_df["has_sentence_text"] = joined_df["sentence_text"].notna()

    # A simple snippet field for later embedding / retrieval work
    joined_df["snippet_text"] = joined_df["sentence_text"]

    return joined_df