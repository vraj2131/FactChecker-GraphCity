from pathlib import Path
from typing import Tuple

import faiss
import numpy as np


class FaissStore:
    """
    Thin wrapper around a FAISS index for dense vector search.

    Responsibilities:
    - create an index
    - add embeddings
    - search nearest neighbors
    - save/load index
    """

    def __init__(self, embedding_dim: int, metric: str = "cosine") -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")

        supported_metrics = {"cosine", "l2"}
        if metric not in supported_metrics:
            raise ValueError(f"metric must be one of {supported_metrics}")

        self.embedding_dim = embedding_dim
        self.metric = metric
        self.index = self._create_index()
        faiss.omp_set_num_threads(1)

    def _create_index(self) -> faiss.Index:
        """
        Create the FAISS index.

        - cosine: uses inner product, assumes normalized embeddings
        - l2: uses Euclidean distance
        """
        if self.metric == "cosine":
            return faiss.IndexFlatIP(self.embedding_dim)

        return faiss.IndexFlatL2(self.embedding_dim)

    def add(self, embeddings: np.ndarray) -> None:
        """
        Add embeddings to the index.

        Expected shape:
            (num_vectors, embedding_dim)
        """
        embeddings = self._validate_embeddings(embeddings)
        self.index.add(embeddings)

    def search(self, query_embeddings: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search nearest neighbors for one or more query embeddings.

        Returns:
            distances: shape (num_queries, top_k)
            indices:   shape (num_queries, top_k)
        """
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")

        query_embeddings = self._validate_embeddings(query_embeddings)
        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices

    def save(self, path: Path) -> None:
        """
        Save the FAISS index to disk.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, embedding_dim: int, metric: str = "cosine") -> "FaissStore":
        """
        Load a FAISS index from disk into a FaissStore instance.
        """
        if not path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {path}")

        store = cls(embedding_dim=embedding_dim, metric=metric)
        store.index = faiss.read_index(str(path))
        return store

    def ntotal(self) -> int:
        """
        Return total number of vectors in the index.
        """
        return int(self.index.ntotal)

    def _validate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Ensure embeddings are 2D float32 and match expected dimension.
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"but got {embeddings.shape[1]}."
            )

        return np.ascontiguousarray(embeddings.astype(np.float32))