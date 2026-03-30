"""
ChromaDB vector store wrapper.

ChromaDB runs entirely locally — it's just files on disk.  No server,
no cloud, no cost.

The collection is configured to use cosine distance so that similarity
scores are straightforward to interpret (0 = identical, 1 = unrelated).
"""

from __future__ import annotations

from typing import Any

import chromadb


_COLLECTION_NAME = "fpl_docs"


class VectorStore:
    """
    Thin wrapper around a ChromaDB collection.

    Pass a chromadb client at construction time so tests can inject an
    in-memory client (EphemeralClient) instead of writing to disk.
    """

    def __init__(self, client: chromadb.ClientAPI, collection_name: str = _COLLECTION_NAME) -> None:
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Store chunks with their embeddings and metadata."""
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        embedding: list[float],
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Return the n_results most similar chunks to the given embedding.

        Each result dict contains: text, metadata, distance.
        Distance is cosine distance (0 = identical, 1 = completely unrelated).
        """
        raw = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self.count()) or 1,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for text, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append({"text": text, "metadata": meta, "distance": dist})
        return results

    def count(self) -> int:
        return self._collection.count()


def make_persistent_store(path: str = "./chroma_db") -> VectorStore:
    """Create a store that persists to disk between sessions."""
    client = chromadb.PersistentClient(path=path)
    return VectorStore(client)


def make_ephemeral_store() -> VectorStore:
    """Create an in-memory store with a unique collection (used in tests)."""
    import uuid
    client = chromadb.EphemeralClient()
    return VectorStore(client, collection_name=f"fpl_docs_{uuid.uuid4().hex}")
