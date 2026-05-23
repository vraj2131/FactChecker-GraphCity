from typing import List, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Thin wrapper around a sentence-transformers embedding model.

    Responsibilities:
    - load the embedding model
    - encode one or many texts
    - return embeddings as float32 numpy arrays
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode one string or a sequence of strings into embeddings.

        Returns:
            np.ndarray of shape:
            - (embedding_dim,) for a single string input
            - (num_texts, embedding_dim) for multiple strings
        """
        is_single_input = isinstance(texts, str)
        input_texts: List[str] = [texts] if is_single_input else list(texts)

        if not input_texts:
            raise ValueError("texts cannot be empty.")

        cleaned_texts = [self._clean_text(text) for text in input_texts]

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

        embeddings = embeddings.astype(np.float32)

        if is_single_input:
            return embeddings[0]

        return embeddings

    def get_embedding_dimension(self) -> int:
        """
        Return the model embedding dimension.
        """
        return int(self.model.get_sentence_embedding_dimension())

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Minimal defensive cleanup before encoding.
        """
        if text is None:
            raise ValueError("Input text cannot be None.")

        cleaned = str(text).strip()
        if not cleaned:
            raise ValueError("Input text cannot be empty or whitespace only.")

        return cleaned

# from backend.app.models.embedding_model import EmbeddingModel

# model = EmbeddingModel(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     device="cpu"
# )

# embedding = model.encode("Amazon stock rose by 5% today.")
# print(embedding.shape)

# batch = model.encode([
#     "Amazon stock rose by 5% today.",
#     "Tesla shares fell after earnings."
# ])
# print(batch.shape)