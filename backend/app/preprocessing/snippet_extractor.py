import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backend.app.preprocessing.normalize_text import normalize_snippet_text
from backend.app.preprocessing.source_cleaner import (
    clean_page_title,
    clean_sentence_text,
    truncate_to_char_limit,
)
from backend.app.utils.constants import (
    DEFAULT_SNIPPET_MAX_CHARS,
    DEFAULT_SNIPPET_MAX_SENTENCES,
    DEFAULT_SNIPPET_MIN_WORDS,
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


# ---------------------------------------------------------------------------
# Phase 6: Claim-based sentence extraction from Source snippets
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
    "and", "or", "but", "it", "its", "this", "that", "which", "who",
    "not", "no", "so", "if", "do", "did", "has", "have", "had",
})


def _tokenize(text: str) -> frozenset:
    """Lowercase meaningful word set — stopwords excluded."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return frozenset(
        w for w in text.split() if len(w) >= 2 and w not in _STOP_WORDS
    )


def score_sentence_relevance(claim: str, sentence: str) -> float:
    """
    Score how relevant a sentence is to a claim.

    Uses claim coverage ratio: fraction of meaningful claim terms found in
    the sentence. Multi-term matches are required for a useful score —
    sentences that match only a single claim term are heavily penalized.

    Returns a float in [0, 1]. Higher is more relevant.
    Returns 0.0 if either input is empty.
    """
    if not claim or not sentence:
        return 0.0

    claim_words = _tokenize(claim)
    sentence_words = _tokenize(sentence)

    if not claim_words or not sentence_words:
        return 0.0

    matched = claim_words & sentence_words
    n_matched = len(matched)

    if n_matched == 0:
        return 0.0

    # Fraction of the claim's key terms that appear in this sentence
    coverage = n_matched / len(claim_words)

    # Penalize single-term hits — "Paris" alone is weak evidence
    if n_matched < 2:
        coverage *= 0.3

    return min(coverage, 1.0)


def extract_best_snippet(
    claim: str,
    text: str,
    max_sentences: int = DEFAULT_SNIPPET_MAX_SENTENCES,
    max_chars: int = DEFAULT_SNIPPET_MAX_CHARS,
) -> Optional[str]:
    """
    Extract the most claim-relevant sentences from a body of text.

    Steps:
    1. Split text into sentences on sentence-ending punctuation.
    2. Filter out sentences shorter than DEFAULT_SNIPPET_MIN_WORDS words.
    3. Score each sentence against the claim using Jaccard overlap.
    4. Take the top max_sentences sentences, preserving their original order.
    5. Join, normalize (HTML entities, whitespace), and truncate to max_chars.
    6. Return None if no usable sentence is found.

    Args:
        claim:         The claim text to match against.
        text:          The raw source body or snippet text.
        max_sentences: Maximum number of sentences to include.
        max_chars:     Hard character limit on the final snippet.

    Returns:
        A clean snippet string, or None if nothing usable was found.
    """
    if not claim or not text:
        return None

    # Split on sentence-ending punctuation followed by whitespace or end-of-string
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    # Filter by minimum word count
    usable = [
        s for s in raw_sentences
        if s.strip() and len(s.strip().split()) >= DEFAULT_SNIPPET_MIN_WORDS
    ]

    if not usable:
        return None

    # Score and keep track of original index for order preservation
    scored = sorted(
        enumerate(usable),
        key=lambda idx_sent: score_sentence_relevance(claim, idx_sent[1]),
        reverse=True,
    )

    # Take top-k by score, then restore original document order
    top_k = sorted(scored[:max_sentences], key=lambda idx_sent: idx_sent[0])
    selected = [s for _, s in top_k]

    joined = " ".join(selected)
    normalized = normalize_snippet_text(joined)

    if not normalized:
        return None

    return truncate_to_char_limit(normalized, max_chars)