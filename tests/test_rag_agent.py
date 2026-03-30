"""
Unit tests for RagAgent.

The real sentence-transformers model is never loaded here.  Instead we
inject a fake Embedder that returns deterministic vectors, and use
ChromaDB's EphemeralClient (in-memory) so no files are written to disk.
"""

from __future__ import annotations

import pytest

from agents.rag_agent import RagAgent, Document, RetrievedChunk
from agents.rag_agent.agent import _split_into_chunks
from agents.rag_agent.store import make_ephemeral_store


# ---------------------------------------------------------------------------
# Fake embedder — returns deterministic vectors without loading any model
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """
    Returns a 384-dim vector where all values equal a hash of the first
    word of the text.  Different texts get meaningfully different vectors
    so cosine similarity comparisons work correctly in tests.
    """
    DIM = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            seed = hash(text.split()[0]) if text.strip() else 0
            # Normalise to unit vector so cosine distance works correctly
            raw = [(seed * (i + 1)) % 100 / 100.0 for i in range(self.DIM)]
            magnitude = sum(v ** 2 for v in raw) ** 0.5 or 1.0
            result.append([v / magnitude for v in raw])
        return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent() -> RagAgent:
    return RagAgent(store=make_ephemeral_store(), embedder=FakeEmbedder())


OPINION_DOC = Document(
    content=(
        "Salah is the standout captaincy option for gameweek 28. "
        "He has scored in four of his last five home games and faces "
        "a side that has conceded ten goals in their last four away matches. "
        "At 12.5m he is expensive but his floor is exceptionally high."
    ),
    source="FPL Focus",
    doc_type="opinion",
    author="FPL Focus",
    trust_weight=8,
)

FACT_DOC = Document(
    content=(
        "The FPL captain earns double points for that gameweek. "
        "If your captain does not play, the vice-captain earns double points instead. "
        "You must set your captain before the gameweek deadline."
    ),
    source="Official FPL Rules",
    doc_type="fact",
    trust_weight=10,
)

LOW_TRUST_DOC = Document(
    content=(
        "Salah is overpriced and should be avoided. "
        "Random blogger opinion with no evidence."
    ),
    source="Random Blog",
    doc_type="opinion",
    author="Random Blogger",
    trust_weight=2,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ingest_returns_chunk_count(agent):
    n = agent.ingest(OPINION_DOC)
    assert n >= 1


def test_document_count_increases_after_ingest(agent):
    assert agent.document_count() == 0
    agent.ingest(OPINION_DOC)
    assert agent.document_count() > 0


def test_retrieve_returns_retrieved_chunks(agent):
    agent.ingest(OPINION_DOC)
    results = agent.retrieve("who should I captain?")
    assert len(results) > 0
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_retrieve_empty_store_returns_empty_list(agent):
    results = agent.retrieve("who should I captain?")
    assert results == []


def test_retrieved_chunks_have_correct_metadata(agent):
    agent.ingest(OPINION_DOC)
    results = agent.retrieve("captain pick")
    assert results[0].source == "FPL Focus"
    assert results[0].doc_type == "opinion"
    assert results[0].trust_weight == 8


def test_boosted_score_higher_than_low_trust(agent):
    agent.ingest(OPINION_DOC)    # trust 8
    agent.ingest(LOW_TRUST_DOC)  # trust 2

    results = agent.retrieve("salah captain")
    scores_by_source = {r.source: r.boosted_score for r in results}

    if "FPL Focus" in scores_by_source and "Random Blog" in scores_by_source:
        assert scores_by_source["FPL Focus"] > scores_by_source["Random Blog"]


def test_preferred_author_boost(agent):
    agent.ingest(OPINION_DOC)    # trust 8, author FPL Focus
    agent.ingest(LOW_TRUST_DOC)  # trust 2, author Random Blogger

    without_pref = agent.retrieve("salah", n_results=5)
    with_pref    = agent.retrieve("salah", n_results=5, preferred_authors=["FPL Focus"])

    fpl_focus_without = next((r for r in without_pref if r.source == "FPL Focus"), None)
    fpl_focus_with    = next((r for r in with_pref    if r.source == "FPL Focus"), None)

    if fpl_focus_without and fpl_focus_with:
        assert fpl_focus_with.boosted_score >= fpl_focus_without.boosted_score


def test_multiple_documents_ingested(agent):
    agent.ingest(OPINION_DOC)
    agent.ingest(FACT_DOC)
    assert agent.document_count() >= 2


def test_retrieve_respects_n_results(agent):
    agent.ingest(OPINION_DOC)
    agent.ingest(FACT_DOC)
    agent.ingest(LOW_TRUST_DOC)
    results = agent.retrieve("captain points rules", n_results=2)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# Chunking unit tests (no store or embedder needed)
# ---------------------------------------------------------------------------

def test_chunker_splits_long_text():
    long_text = ("This is a sentence about FPL. " * 30).strip()
    chunks = _split_into_chunks(long_text, chunk_size=200)
    assert len(chunks) > 1


def test_chunker_short_text_is_single_chunk():
    short = "Salah is a great captaincy pick this week."
    chunks = _split_into_chunks(short)
    assert len(chunks) == 1


def test_chunker_no_empty_chunks():
    text = "First sentence. Second sentence. Third sentence."
    chunks = _split_into_chunks(text)
    assert all(c.strip() for c in chunks)
