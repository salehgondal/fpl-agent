"""
Local embedding model wrapper.

Uses sentence-transformers (runs entirely on your machine, no API calls,
no cost).  The model file (~80 MB) is downloaded once on first use and
cached locally by the library.

all-MiniLM-L6-v2 is fast on CPU and produces 384-dimensional vectors —
a good balance of speed and quality for an FPL RAG use case.
"""

from __future__ import annotations

_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """
    Wraps a sentence-transformers model.

    Lazy-loads the model on first call to embed() so that importing this
    module doesn't trigger a download or slow startup.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None   # loaded on first use

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of text strings into a list of embedding vectors.

        Each vector is a list of 384 floats.  Texts are embedded in one
        batched call for efficiency.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

        vectors = self._model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()
