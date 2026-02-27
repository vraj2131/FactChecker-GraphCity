from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from backend.app.models.embedding_model import EmbeddingModel
from backend.app.utils.file_io import (
    load_parquet,
    save_numpy,
    save_pickle,
)
from backend.app.vectorstore.faiss_store import FaissStore


class ClaimIndexBuilder:
    """
    Builds FAISS indexes for:
    - FEVER claims
    - FEVER evidence snippets

    Responsibilities:
    - load processed parquet files
    - extract valid text
    - generate embeddings
    - build FAISS index
    - save embeddings cache
    - save metadata map
    - save FAISS index
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        faiss_metric: str = "cosine",
    ) -> None:
        self.embedding_model = embedding_model
        self.faiss_metric = faiss_metric

    # ------------------------------------------------------------------
    # Claims
    # ------------------------------------------------------------------
    def load_claims_dataframe(self, claims_parquet_path: Path) -> pd.DataFrame:
        """
        Load processed FEVER claims parquet.
        """
        df = load_parquet(claims_parquet_path)

        required_columns = {
            "claim_id",
            "claim_text",
            "claim_text_normalized",
            "label",
            "verifiable",
            "split",
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Claims parquet is missing required columns: {sorted(missing)}")

        return df

    def prepare_claim_records(
        self,
        claims_df: pd.DataFrame,
        text_column: str = "claim_text_normalized",
    ) -> Tuple[List[str], List[Dict]]:
        """
        Convert claims dataframe into:
        - list of texts to embed
        - metadata list aligned by row index
        """
        if text_column not in claims_df.columns:
            raise ValueError(f"text_column '{text_column}' not found in claims dataframe.")

        usable_df = claims_df.dropna(subset=[text_column]).copy()
        usable_df[text_column] = usable_df[text_column].astype(str).str.strip()
        usable_df = usable_df[usable_df[text_column] != ""]

        if usable_df.empty:
            raise ValueError("No usable claims found after cleaning text column.")

        texts = usable_df[text_column].tolist()

        metadata: List[Dict] = []
        for row_idx, row in usable_df.reset_index(drop=True).iterrows():
            metadata.append(
                {
                    "row_index": row_idx,
                    "record_type": "claim",
                    "claim_id": row["claim_id"],
                    "claim_text": row["claim_text"],
                    "claim_text_normalized": row["claim_text_normalized"],
                    "label": row["label"],
                    "verifiable": row["verifiable"],
                    "split": row["split"],
                }
            )

        return texts, metadata

    def build_index_from_claims(
        self,
        claims_parquet_path: Path,
        embeddings_output_path: Path,
        metadata_output_path: Path,
        faiss_index_output_path: Path,
        text_column: str = "claim_text_normalized",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = True,
    ) -> Dict:
        """
        Full build pipeline for FEVER claims.
        """
        claims_df = self.load_claims_dataframe(claims_parquet_path)
        texts, metadata = self.prepare_claim_records(claims_df, text_column=text_column)

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

        embedding_dim = self.embedding_model.get_embedding_dimension()
        store = FaissStore(embedding_dim=embedding_dim, metric=self.faiss_metric)
        store.add(embeddings)

        save_numpy(embeddings, embeddings_output_path)
        save_pickle(metadata, metadata_output_path)
        store.save(faiss_index_output_path)

        return {
            "index_type": "claims",
            "num_records_indexed": len(texts),
            "embedding_dim": embedding_dim,
            "embeddings_output_path": str(embeddings_output_path),
            "metadata_output_path": str(metadata_output_path),
            "faiss_index_output_path": str(faiss_index_output_path),
        }

    # ------------------------------------------------------------------
    # Evidence snippets
    # ------------------------------------------------------------------
    def load_evidence_snippets_dataframe(self, evidence_snippets_parquet_path: Path) -> pd.DataFrame:
        """
        Load processed FEVER evidence snippets parquet.
        """
        df = load_parquet(evidence_snippets_parquet_path)

        required_columns = {
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
            "page_title_clean",
            "sentence_id",
            "sentence_text",
            "has_sentence_text",
            "snippet_text",
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Evidence snippets parquet is missing required columns: {sorted(missing)}"
            )

        return df

    def prepare_evidence_snippet_records(
        self,
        evidence_snippets_df: pd.DataFrame,
        text_column: str = "snippet_text",
        require_sentence_text: bool = True,
        drop_duplicates: bool = True,
    ) -> Tuple[List[str], List[Dict]]:
        """
        Convert evidence snippet dataframe into:
        - list of snippet texts to embed
        - metadata list aligned by row index
        """
        if text_column not in evidence_snippets_df.columns:
            raise ValueError(f"text_column '{text_column}' not found in evidence snippets dataframe.")

        usable_df = evidence_snippets_df.copy()

        if require_sentence_text:
            usable_df = usable_df[usable_df["has_sentence_text"] == True].copy()

        usable_df = usable_df.dropna(subset=[text_column]).copy()
        usable_df[text_column] = usable_df[text_column].astype(str).str.strip()
        usable_df = usable_df[usable_df[text_column] != ""]

        if drop_duplicates:
            usable_df = usable_df.drop_duplicates(
                subset=["page_title_clean", "sentence_id", text_column]
            ).copy()

        usable_df = usable_df.reset_index(drop=True)

        if usable_df.empty:
            raise ValueError("No usable evidence snippets found after filtering and cleaning.")

        texts = usable_df[text_column].tolist()

        metadata: List[Dict] = []
        for row_idx, row in usable_df.iterrows():
            metadata.append(
                {
                    "row_index": row_idx,
                    "record_type": "evidence_snippet",
                    "claim_id": row["claim_id"],
                    "claim_text": row["claim_text"],
                    "claim_text_normalized": row["claim_text_normalized"],
                    "label": row["label"],
                    "verifiable": row["verifiable"],
                    "split": row["split"],
                    "evidence_set_index": row["evidence_set_index"],
                    "evidence_item_index": row["evidence_item_index"],
                    "annotation_id": row["annotation_id"],
                    "evidence_id": row["evidence_id"],
                    "page_title": row["page_title"],
                    "page_title_clean": row["page_title_clean"],
                    "sentence_id": row["sentence_id"],
                    "sentence_text": row["sentence_text"],
                    "snippet_text": row[text_column],
                }
            )

        return texts, metadata

    def build_index_from_evidence_snippets(
        self,
        evidence_snippets_parquet_path: Path,
        embeddings_output_path: Path,
        metadata_output_path: Path,
        faiss_index_output_path: Path,
        text_column: str = "snippet_text",
        require_sentence_text: bool = True,
        drop_duplicates: bool = True,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = True,
    ) -> Dict:
        """
        Full build pipeline for FEVER evidence snippets.
        """
        evidence_snippets_df = self.load_evidence_snippets_dataframe(
            evidence_snippets_parquet_path
        )

        texts, metadata = self.prepare_evidence_snippet_records(
            evidence_snippets_df=evidence_snippets_df,
            text_column=text_column,
            require_sentence_text=require_sentence_text,
            drop_duplicates=drop_duplicates,
        )

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

        embedding_dim = self.embedding_model.get_embedding_dimension()
        store = FaissStore(embedding_dim=embedding_dim, metric=self.faiss_metric)
        store.add(embeddings)

        save_numpy(embeddings, embeddings_output_path)
        save_pickle(metadata, metadata_output_path)
        store.save(faiss_index_output_path)

        return {
            "index_type": "evidence_snippets",
            "num_records_indexed": len(texts),
            "embedding_dim": embedding_dim,
            "embeddings_output_path": str(embeddings_output_path),
            "metadata_output_path": str(metadata_output_path),
            "faiss_index_output_path": str(faiss_index_output_path),
        }