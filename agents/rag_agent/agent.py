"""
RagAgent — ingest FPL documents and retrieve relevant chunks by query.

Two public operations:
  ingest(document)           — chunk, embed, and store a document
  retrieve(query, ...)       — find the most relevant chunks for a question

Nothing here connects to the FPL API or any other agent yet.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

from .embedder import Embedder
from .models import Document, RetrievedChunk
from .store import VectorStore, make_persistent_store

_CHUNK_SIZE = 500    # target characters per chunk
_OVERLAP_WORDS = 10  # words carried over to next chunk for context continuity


class RagAgent:
    """
    Ingest FPL documents and retrieve relevant chunks.

    Usage::

        agent = RagAgent()                        # persists to ./chroma_db
        agent.ingest(Document(
            content="Salah has scored in 4 of his last 5 home games...",
            source="FPL Focus",
            doc_type="opinion",
            author="FPL Focus",
            trust_weight=8,
        ))
        results = agent.retrieve("who should I captain?")
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        db_path: str = "./chroma_db",
    ) -> None:
        self._store = store or make_persistent_store(db_path)
        self._embedder = embedder or Embedder()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, document: Document) -> int:
        """
        Chunk, embed, and store a document.

        Returns the number of chunks stored.
        """
        chunks = _split_into_chunks(document.content)

        if not chunks:
            return 0

        embeddings = self._embedder.embed(chunks)

        metadata = {
            "source": document.source,
            "doc_type": document.doc_type,
            "author": document.author or "",
            "trust_weight": document.trust_weight,
        }

        ids = [f"{document.source}_{uuid.uuid4().hex[:8]}" for _ in chunks]

        self._store.add(
            ids=ids,
            texts=chunks,
            embeddings=embeddings,
            metadatas=[metadata] * len(chunks),
        )

        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        preferred_authors: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Return the most relevant chunks for a query.

        Args:
            query:             The question or topic to search for.
            n_results:         How many chunks to return.
            preferred_authors: Authors whose chunks get a trust boost.
                               Pass the user's favourite experts here.
        """
        if self._store.count() == 0:
            return []

        query_embedding = self._embedder.embed([query])[0]
        # Fetch more than needed so re-ranking has candidates to work with
        raw_results = self._store.query(query_embedding, n_results=n_results * 3)

        chunks = []
        for result in raw_results:
            meta = result["metadata"]
            distance = result["distance"]

            # Cosine distance 0–1 → similarity 0–1
            relevance = max(0.0, 1.0 - distance)

            # Boost trust weight if this author is preferred
            trust = meta["trust_weight"]
            if preferred_authors:
                author = meta.get("author", "") or meta.get("source", "")
                if any(p.lower() in author.lower() for p in preferred_authors):
                    trust = min(10, trust + 3)

            # Boosted score: relevance dominates (70%), trust shapes the rest
            boosted = relevance * (0.7 + 0.3 * trust / 10)

            chunks.append(
                RetrievedChunk(
                    text=result["text"],
                    source=meta["source"],
                    doc_type=meta["doc_type"],
                    author=meta.get("author") or None,
                    trust_weight=meta["trust_weight"],
                    relevance_score=round(relevance, 4),
                    boosted_score=round(boosted, 4),
                )
            )

        # Re-rank by boosted score and return the top n
        chunks.sort(key=lambda c: c.boosted_score, reverse=True)
        return chunks[:n_results]

    def document_count(self) -> int:
        """Total number of chunks currently stored."""
        return self._store.count()


# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------

def _split_into_chunks(text: str, chunk_size: int = _CHUNK_SIZE) -> list[str]:
    """
    Split text into overlapping chunks of roughly chunk_size characters.

    Splits on sentence boundaries where possible so chunks don't cut
    words or ideas in half.  The last few words of each chunk are
    repeated at the start of the next to preserve context across boundaries.
    """
    # Split into sentences on . ! ? followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            # Carry over the last few words as overlap
            words = current.split()
            overlap = " ".join(words[-_OVERLAP_WORDS:])
            current = overlap + " " + sentence
        else:
            current = (current + " " + sentence).strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks
