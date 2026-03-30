"""
Data shapes for the RAG agent.

Document  — what goes in  (a piece of text + metadata about its source)
RetrievedChunk — what comes out (a relevant snippet + its scores)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    A document to be ingested into the RAG system.

    trust_weight controls how much this source is favoured during retrieval.
    1 = low trust (unknown blog), 10 = high trust (official rules / your
    hand-picked expert).  Defaults to 5 (neutral).
    """
    content: str
    source: str                              # e.g. "FPL Focus", "Official FPL Rules"
    doc_type: str                            # "fact" or "opinion"
    author: Optional[str] = None            # individual writer if known
    trust_weight: int = Field(default=5, ge=1, le=10)


class RetrievedChunk(BaseModel):
    """A single chunk returned by a retrieval query."""
    text: str
    source: str
    doc_type: str
    author: Optional[str]
    trust_weight: int
    relevance_score: float    # 0–1  pure semantic similarity
    boosted_score: float      # 0–1  relevance weighted by trust
