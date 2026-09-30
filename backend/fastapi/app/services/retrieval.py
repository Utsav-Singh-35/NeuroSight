"""Evidence retrieval over the cited knowledge base.

Deliberately small
------------------
The corpus is 28 chunks. At that scale a vector database is pure overhead: the
whole index is a ``(28, 384)`` float32 array, about 43 KB, and a similarity
search is one matrix–vector product. So the "vector store" here is a ``.npz``
file and the search is four lines of numpy. That is also easier to defend in a
viva than a framework whose internals cannot be inspected.

Two backends, so the choice is measured rather than asserted
------------------------------------------------------------
``tfidf``
    Lexical baseline via scikit-learn, already a dependency. No model download,
    no network, runs anywhere. Strong when a query reuses the corpus's wording.

``embedding``
    Semantic search with ``sentence-transformers/all-MiniLM-L6-v2`` (~80 MB,
    384-dim). Handles paraphrase, which is what real user questions look like.

Both implement the same interface, so :mod:`scripts.eval_retrieval` can score
them on the same hand-labelled query set. If the embedding model cannot be
obtained (no network, no cache), the embedding backend reports itself
unavailable and callers fall back to TF-IDF instead of failing.

Cosine similarity
-----------------
Vectors are L2-normalised at index time, which reduces cosine similarity to a
dot product::

    scores = matrix @ query_vector
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.knowledge import get_knowledge_base

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILENAME = "index.npz"
META_FILENAME = "index_meta.json"


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise so dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class RetrievalHit:
    """One retrieved chunk with its score."""

    chunk_id: str
    module: str
    class_label: str
    section: str
    heading: str
    text: str
    citations: list[str]
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "module": self.module,
            "class": self.class_label,
            "section": self.section,
            "heading": self.heading,
            "text": self.text,
            "citations": self.citations,
            "score": round(float(self.score), 6),
            "rank": self.rank,
        }


class BaseRetriever:
    """Shared plumbing: holds chunks, ranks a score vector."""

    backend = "base"

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.available = False

    def _rank(
        self,
        scores: np.ndarray,
        k: int,
        module: str | None,
        class_label: str | None,
        min_score: float,
    ) -> list[RetrievalHit]:
        """Turn a score vector into the top-k hits, applying filters."""
        order = np.argsort(-scores)
        hits: list[RetrievalHit] = []

        for index in order:
            if len(hits) >= k:
                break
            chunk = self.chunks[int(index)]
            if module and chunk["module"] != module:
                continue
            if class_label and chunk["class"] != class_label:
                continue
            score = float(scores[int(index)])
            if score < min_score:
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=chunk["chunk_id"],
                    module=chunk["module"],
                    class_label=chunk["class"],
                    section=chunk["section"],
                    heading=chunk["heading"],
                    text=chunk["text"],
                    citations=chunk["citations"],
                    score=score,
                    rank=len(hits) + 1,
                )
            )

        return hits

    def search(
        self,
        query: str,
        k: int = 5,
        module: str | None = None,
        class_label: str | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievalHit]:
        raise NotImplementedError


class TfidfRetriever(BaseRetriever):
    """Lexical baseline. No downloads, no network, sklearn only."""

    backend = "tfidf"

    def __init__(self, chunks: list[dict]) -> None:
        super().__init__(chunks)
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        texts = [f"{c['heading']}. {c['text']}" for c in chunks]
        matrix = self._vectorizer.fit_transform(texts)
        self._matrix = _l2_normalise(np.asarray(matrix.todense(), dtype=np.float32))
        self.available = True

    def search(
        self,
        query: str,
        k: int = 5,
        module: str | None = None,
        class_label: str | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievalHit]:
        vector = np.asarray(
            self._vectorizer.transform([query]).todense(), dtype=np.float32
        )
        vector = _l2_normalise(vector)[0]
        return self._rank(self._matrix @ vector, k, module, class_label, min_score)


class EmbeddingRetriever(BaseRetriever):
    """Semantic search over precomputed MiniLM embeddings."""

    backend = "embedding"

    def __init__(
        self,
        chunks: list[dict],
        vectors: np.ndarray | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        super().__init__(chunks)
        self.model_name = model_name
        self._model = None
        self._matrix = vectors

        if vectors is not None:
            if len(vectors) != len(chunks):
                raise ValueError(
                    f"Index has {len(vectors)} vectors but knowledge base has "
                    f"{len(chunks)} chunks. Rebuild the index."
                )
            self._matrix = _l2_normalise(vectors.astype(np.float32))
            self.available = True

    def _get_model(self):
        """Lazily load the encoder. Only needed to embed a query."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed texts into an L2-normalised matrix."""
        model = self._get_model()
        vectors = model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, batch_size=16
        )
        return _l2_normalise(np.asarray(vectors, dtype=np.float32))

    def build(self) -> np.ndarray:
        """Embed every chunk and populate the index."""
        texts = [f"{c['heading']}. {c['text']}" for c in self.chunks]
        self._matrix = self.encode(texts)
        self.available = True
        return self._matrix

    def search(
        self,
        query: str,
        k: int = 5,
        module: str | None = None,
        class_label: str | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievalHit]:
        if self._matrix is None:
            raise RuntimeError("Index not built. Call build() or load a saved index.")
        vector = self.encode([query])[0]
        return self._rank(self._matrix @ vector, k, module, class_label, min_score)


def save_index(path: Path, vectors: np.ndarray, chunks: list[dict], model_name: str) -> None:
    """Persist embeddings plus the chunk metadata they correspond to.

    Metadata is stored next to the vectors so a stale index can be detected:
    the chunk count and IDs must still match the knowledge base.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vectors=vectors.astype(np.float32))

    meta = {
        "model": model_name,
        "dimension": int(vectors.shape[1]),
        "chunk_count": int(vectors.shape[0]),
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }
    with open(path.parent / META_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def load_index(path: Path) -> tuple[np.ndarray | None, dict]:
    """Load persisted embeddings and metadata, tolerating absence."""
    if not path.exists():
        return None, {}

    with np.load(path) as data:
        vectors = data["vectors"]

    meta_path = path.parent / META_FILENAME
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

    return vectors, meta


def get_retriever(
    backend: str = "auto",
    knowledge_root: str | Path | None = None,
    module: str | None = None,
) -> BaseRetriever:
    """Build a retriever over the knowledge base.

    Args:
        backend: ``"embedding"``, ``"tfidf"``, or ``"auto"``. Auto prefers a
            saved embedding index and falls back to TF-IDF when none exists or
            the index is stale, so retrieval always works.
        knowledge_root: Knowledge directory; defaults to the repo's.
        module: Restrict the index to one module's chunks.

    Returns:
        A ready retriever. Check ``.backend`` to see which was chosen.
    """
    kb = get_knowledge_base(str(knowledge_root) if knowledge_root else None)
    chunks = kb.chunks(module)

    if not chunks:
        raise RuntimeError("Knowledge base has no chunks to index.")

    root = Path(knowledge_root) if knowledge_root else Path(kb.root)
    index_path = root / INDEX_FILENAME

    if backend in ("embedding", "auto"):
        vectors, meta = load_index(index_path)

        if vectors is not None:
            expected = [c["chunk_id"] for c in chunks]
            if meta.get("chunk_ids") == expected:
                return EmbeddingRetriever(
                    chunks, vectors, meta.get("model", DEFAULT_MODEL)
                )
            logger.warning(
                "Embedding index is stale (%d indexed vs %d current chunks). "
                "Rebuild with scripts/build_index.py.",
                meta.get("chunk_count", len(vectors)),
                len(chunks),
            )

        if backend == "embedding":
            # Explicitly requested: build in-process rather than silently
            # degrading. This needs the model, so it may download.
            retriever = EmbeddingRetriever(chunks, model_name=DEFAULT_MODEL)
            retriever.build()
            return retriever

        logger.info("No usable embedding index; falling back to TF-IDF retrieval.")

    return TfidfRetriever(chunks)
