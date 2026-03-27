# from pathlib import Path
# from typing import Any, List, Optional, Set

# import pandas as pd

# from backend.app.retrieval.base_retriever import BaseRetriever
# from backend.app.schemas.source_schema import Source
# from backend.app.utils.constants import (
#     DEFAULT_PROCESSED_DIR,
#     DEFAULT_RETRIEVER_MAX_RESULTS,
#     FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME,
#     SOURCE_NAME_WIKIPEDIA,
# )


# class WikipediaRetriever(BaseRetriever):
#     """
#     Wikipedia retriever backed by processed FEVER evidence snippets.

#     Phase 4 goal:
#     - fast local smoke test
#     - return normalized Source objects
#     - avoid scanning full wiki_sentences.parquet row by row

#     This uses:
#         data/processed/fever_evidence_snippets.parquet
#     and only keeps rows with actual sentence text.
#     """

#     def __init__(
#         self,
#         evidence_snippets_path: Optional[Path] = None,
#     ) -> None:
#         super().__init__(source_name=SOURCE_NAME_WIKIPEDIA)
#         self.evidence_snippets_path = evidence_snippets_path or (
#             DEFAULT_PROCESSED_DIR / FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME
#         )
#         self._evidence_df: Optional[pd.DataFrame] = None

#     def fetch_raw(
#         self,
#         query: str,
#         max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
#         **kwargs: Any,
#     ) -> pd.DataFrame:
#         df = self._load_evidence_df()

#         query_tokens = self._tokenize(query)
#         if not query_tokens:
#             return df.iloc[0:0].copy()

#         query_lower = query.lower().strip()

#         sentence_token_overlap = df["sentence_text"].map(
#             lambda text: len(query_tokens & self._tokenize(text))
#         )
#         title_token_overlap = df["page_title_clean"].map(
#             lambda text: len(query_tokens & self._tokenize(text))
#         )
#         exact_query_in_text = df["sentence_text"].str.lower().str.contains(query_lower, regex=False).astype(int)
#         exact_query_in_title = df["page_title_clean"].str.lower().str.contains(query_lower, regex=False).astype(int)

#         scored_df = df.copy()
#         scored_df["sentence_token_overlap"] = sentence_token_overlap
#         scored_df["title_token_overlap"] = title_token_overlap
#         scored_df["exact_query_in_text"] = exact_query_in_text
#         scored_df["exact_query_in_title"] = exact_query_in_title
#         scored_df["retrieval_score"] = (
#             scored_df["sentence_token_overlap"]
#             + scored_df["title_token_overlap"]
#             + 2 * scored_df["exact_query_in_text"]
#             + 1 * scored_df["exact_query_in_title"]
#         )

#         scored_df = scored_df[scored_df["retrieval_score"] > 0].copy()
#         if scored_df.empty:
#             return scored_df

#         scored_df = scored_df.sort_values(
#             by=[
#                 "retrieval_score",
#                 "sentence_token_overlap",
#                 "title_token_overlap",
#                 "page_title_clean",
#                 "sentence_id",
#             ],
#             ascending=[False, False, False, True, True],
#         ).reset_index(drop=True)

#         return scored_df.head(max_results).copy()

#     def normalize(
#         self,
#         raw_data: Any,
#         query: str,
#         max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
#         **kwargs: Any,
#     ) -> List[Source]:
#         if raw_data is None or len(raw_data) == 0:
#             return []

#         sources: List[Source] = []

#         for _, row in raw_data.iterrows():
#             page_title_clean = str(row["page_title_clean"])
#             sentence_id = int(row["sentence_id"])
#             sentence_text = str(row["sentence_text"])

#             page_url = self._build_wikipedia_url(page_title_clean)
#             source_id = f"wikipedia::{page_title_clean}::{sentence_id}"

#             source = Source(
#                 source_id=source_id,
#                 source_type="wikipedia",
#                 title=f"Wikipedia: {page_title_clean} (sentence {sentence_id})",
#                 url=page_url,
#                 publisher="Wikipedia",
#                 snippet=sentence_text,
#                 published_at=None,
#                 trust_score=0.75,
#                 relevance_score=self._normalize_score(float(row.get("retrieval_score", 0.0))),
#                 stance_hint=None,
#             )
#             sources.append(source)

#         return self.postprocess(sources, max_results=max_results)

#     def _load_evidence_df(self) -> pd.DataFrame:
#         if self._evidence_df is not None:
#             return self._evidence_df

#         if not self.evidence_snippets_path.exists():
#             raise FileNotFoundError(
#                 f"evidence snippets parquet not found: {self.evidence_snippets_path}"
#             )

#         df = pd.read_parquet(self.evidence_snippets_path)

#         required_columns = {
#             "page_title",
#             "page_title_clean",
#             "sentence_id",
#             "sentence_text",
#             "has_sentence_text",
#         }
#         missing = required_columns - set(df.columns)
#         if missing:
#             raise ValueError(
#                 f"evidence snippets parquet missing required columns: {sorted(missing)}"
#             )

#         df = df[df["has_sentence_text"] == True].copy()
#         df = df.dropna(subset=["page_title_clean", "sentence_text"]).copy()
#         df["page_title_clean"] = df["page_title_clean"].astype(str).str.strip()
#         df["sentence_text"] = df["sentence_text"].astype(str).str.strip()
#         df = df[
#             (df["page_title_clean"] != "") &
#             (df["sentence_text"] != "")
#         ].reset_index(drop=True)

#         # keep unique sentence rows only
#         df = df.drop_duplicates(
#             subset=["page_title_clean", "sentence_id", "sentence_text"]
#         ).reset_index(drop=True)

#         self._evidence_df = df
#         return self._evidence_df

#     @staticmethod
#     def _tokenize(text: str) -> Set[str]:
#         cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text))
#         return {token for token in cleaned.split() if len(token) >= 2}

#     @staticmethod
#     def _build_wikipedia_url(page_title_clean: str) -> str:
#         page_slug = page_title_clean.replace(" ", "_")
#         return f"https://en.wikipedia.org/wiki/{page_slug}"

#     @staticmethod
#     def _normalize_score(score: float) -> float:
#         if score <= 0:
#             return 0.0
#         return min(score / 10.0, 1.0)

import re
from pathlib import Path
from typing import Any, List, Optional, Set

import pandas as pd

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RETRIEVER_MAX_RESULTS,
    FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME,
    SOURCE_NAME_WIKIPEDIA,
)


class WikipediaRetriever(BaseRetriever):
    """
    Wikipedia retriever backed by processed FEVER evidence snippets.

    Improvements in this patched version:
    - removes weak/common stopwords from token matching
    - boosts exact phrase matches
    - strongly boosts page-title matches
    - filters out very weak matches
    """

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
        "and", "or", "but", "if", "then", "than", "into", "over", "under",
        "after", "before", "during", "about", "between", "through", "up",
        "down", "out", "off", "this", "that", "these", "those"
    }

    def __init__(
        self,
        evidence_snippets_path: Optional[Path] = None,
    ) -> None:
        super().__init__(source_name=SOURCE_NAME_WIKIPEDIA)
        self.evidence_snippets_path = evidence_snippets_path or (
            DEFAULT_PROCESSED_DIR / FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME
        )
        self._evidence_df: Optional[pd.DataFrame] = None

    def fetch_raw(
        self,
        query: str,
        max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
        **kwargs: Any,
    ) -> pd.DataFrame:
        df = self._load_evidence_df()

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return df.iloc[0:0].copy()

        query_lower = query.lower().strip()
        query_compact = query_lower.replace(" ", "_")

        def sentence_overlap(text: str) -> int:
            return len(query_tokens & self._tokenize(text))

        def title_overlap(text: str) -> int:
            return len(query_tokens & self._tokenize(text))

        scored_df = df.copy()

        scored_df["sentence_token_overlap"] = scored_df["sentence_text"].map(sentence_overlap)
        scored_df["title_token_overlap"] = scored_df["page_title_clean"].map(title_overlap)

        scored_df["exact_query_in_text"] = scored_df["sentence_text"].str.lower().str.contains(
            query_lower, regex=False
        ).astype(int)

        scored_df["exact_query_in_title"] = scored_df["page_title_clean"].str.lower().str.contains(
            query_lower, regex=False
        ).astype(int)

        scored_df["query_compact_in_title"] = scored_df["page_title_clean"].str.lower().str.contains(
            query_compact, regex=False
        ).astype(int)

        scored_df["all_query_tokens_in_title"] = scored_df["page_title_clean"].map(
            lambda text: int(query_tokens.issubset(self._tokenize(text)))
        )

        scored_df["all_query_tokens_in_text"] = scored_df["sentence_text"].map(
            lambda text: int(query_tokens.issubset(self._tokenize(text)))
        )

        scored_df["retrieval_score"] = (
            2 * scored_df["sentence_token_overlap"]
            + 4 * scored_df["title_token_overlap"]
            + 6 * scored_df["exact_query_in_text"]
            + 8 * scored_df["exact_query_in_title"]
            + 10 * scored_df["query_compact_in_title"]
            + 8 * scored_df["all_query_tokens_in_title"]
            + 4 * scored_df["all_query_tokens_in_text"]
        )

        # Filter weak matches aggressively
        scored_df = scored_df[
            (scored_df["title_token_overlap"] >= 1) |
            (scored_df["sentence_token_overlap"] >= 2) |
            (scored_df["exact_query_in_text"] == 1) |
            (scored_df["exact_query_in_title"] == 1) |
            (scored_df["query_compact_in_title"] == 1) |
            (scored_df["all_query_tokens_in_title"] == 1)
        ].copy()

        if scored_df.empty:
            return scored_df

        scored_df = scored_df.sort_values(
            by=[
                "retrieval_score",
                "query_compact_in_title",
                "all_query_tokens_in_title",
                "title_token_overlap",
                "sentence_token_overlap",
                "page_title_clean",
                "sentence_id",
            ],
            ascending=[False, False, False, False, False, True, True],
        ).reset_index(drop=True)

        return scored_df.head(max_results).copy()

    def normalize(
        self,
        raw_data: Any,
        query: str,
        max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
        **kwargs: Any,
    ) -> List[Source]:
        if raw_data is None or len(raw_data) == 0:
            return []

        sources: List[Source] = []

        for _, row in raw_data.iterrows():
            page_title_clean = str(row["page_title_clean"])
            sentence_id = int(row["sentence_id"])
            sentence_text = self._clean_annotation_markup(str(row["sentence_text"]))

            page_url = self._build_wikipedia_url(page_title_clean)
            source_id = f"wikipedia::{page_title_clean}::{sentence_id}"

            source = Source(
                source_id=source_id,
                source_type="wikipedia",
                title=f"Wikipedia: {page_title_clean} (sentence {sentence_id})",
                url=page_url,
                publisher="Wikipedia",
                snippet=sentence_text,
                published_at=None,
                trust_score=0.75,
                relevance_score=self._normalize_score(float(row.get("retrieval_score", 0.0))),
                stance_hint=None,
            )
            sources.append(source)

        return self.postprocess(sources, max_results=max_results)

    def _load_evidence_df(self) -> pd.DataFrame:
        if self._evidence_df is not None:
            return self._evidence_df

        if not self.evidence_snippets_path.exists():
            raise FileNotFoundError(
                f"evidence snippets parquet not found: {self.evidence_snippets_path}"
            )

        df = pd.read_parquet(self.evidence_snippets_path)

        required_columns = {
            "page_title",
            "page_title_clean",
            "sentence_id",
            "sentence_text",
            "has_sentence_text",
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"evidence snippets parquet missing required columns: {sorted(missing)}"
            )

        df = df[df["has_sentence_text"] == True].copy()
        df = df.dropna(subset=["page_title_clean", "sentence_text"]).copy()
        df["page_title_clean"] = df["page_title_clean"].astype(str).str.strip()
        df["sentence_text"] = df["sentence_text"].astype(str).str.strip()
        df = df[
            (df["page_title_clean"] != "") &
            (df["sentence_text"] != "")
        ].reset_index(drop=True)

        df = df.drop_duplicates(
            subset=["page_title_clean", "sentence_id", "sentence_text"]
        ).reset_index(drop=True)

        self._evidence_df = df
        return self._evidence_df

    @classmethod
    def _tokenize(cls, text: str) -> Set[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text))
        tokens = {token for token in cleaned.split() if len(token) >= 2}
        return {token for token in tokens if token not in cls.STOPWORDS}

    @staticmethod
    def _build_wikipedia_url(page_title_clean: str) -> str:
        page_slug = page_title_clean.replace(" ", "_")
        return f"https://en.wikipedia.org/wiki/{page_slug}"

    @staticmethod
    def _clean_annotation_markup(text: str) -> str:
        """Strip FEVER wiki annotation markup appended after sentence text.

        FEVER wiki sentences have the form:
            "Actual sentence text. Entity1 Entity1_target Entity2 ..."
        We keep only the actual sentence up to (and including) the first
        sentence-ending punctuation mark, then decode FEVER bracket escapes.
        """
        match = re.search(r"[.!?]", text)
        if match:
            text = text[: match.end()].strip()
        else:
            text = text.strip()
        # Decode FEVER bracket escapes
        text = text.replace("-LRB-", "(").replace("-RRB-", ")")
        text = text.replace("-LSB-", "[").replace("-RSB-", "]")
        text = text.replace("-LCB-", "{").replace("-RCB-", "}")
        return text

    @staticmethod
    def _normalize_score(score: float) -> float:
        if score <= 0:
            return 0.0
        return min(score / 20.0, 1.0)
