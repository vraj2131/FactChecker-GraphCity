from typing import Dict, List

from backend.app.retrieval.base_retriever import BaseRetriever


class RetrieverRegistry:
    """
    Simple in-memory registry for retriever instances.

    Allows the rest of the system to:
    - register retrievers by name
    - fetch a retriever by name
    - list available retrievers
    """

    def __init__(self) -> None:
        self._retrievers: Dict[str, BaseRetriever] = {}

    def register(self, retriever: BaseRetriever) -> None:
        """
        Register a retriever instance using its source_name.
        """
        source_name = retriever.get_source_name().strip().lower()

        if not source_name:
            raise ValueError("Retriever source_name cannot be empty.")

        if source_name in self._retrievers:
            raise ValueError(f"Retriever '{source_name}' is already registered.")

        self._retrievers[source_name] = retriever

    def get(self, source_name: str) -> BaseRetriever:
        """
        Retrieve a registered retriever by name.
        """
        key = source_name.strip().lower()

        if key not in self._retrievers:
            available = ", ".join(sorted(self._retrievers.keys())) or "(none)"
            raise KeyError(
                f"Retriever '{key}' is not registered. Available retrievers: {available}"
            )

        return self._retrievers[key]

    def list_names(self) -> List[str]:
        """
        Return registered retriever names in sorted order.
        """
        return sorted(self._retrievers.keys())

    def is_registered(self, source_name: str) -> bool:
        """
        Check whether a retriever name is already registered.
        """
        key = source_name.strip().lower()
        return key in self._retrievers

    def clear(self) -> None:
        """
        Remove all registered retrievers.
        """
        self._retrievers.clear()
