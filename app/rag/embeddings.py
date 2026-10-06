"""Embedding interface.

Application code depends on EmbeddingProvider only. OllamaEmbeddingProvider uses
nomic-embed-text (never the chat model). HashingEmbeddingProvider is a
deterministic, dependency-free provider for offline tests and evaluation.
"""

import hashlib
import math
import re
from abc import ABC, abstractmethod

import httpx

from app.config import Settings, settings as default_settings


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(ABC):
    name: str = "embedding"
    # Cosine-similarity thresholds that suit this provider's score distribution:
    # below min_score a chunk is not considered relevant; at or above
    # confident_score a match is strong enough to trust despite unknown query terms.
    min_score: float = 0.5
    confident_score: float = 0.7
    # Width of the "strongly relevant" band used by source precedence (see ranking.py).
    relevance_margin: float = 0.05

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbeddingProvider(EmbeddingProvider):
    min_score = 0.55
    confident_score = 0.72
    relevance_margin = 0.05

    def __init__(self, cfg: Settings = default_settings, client: httpx.Client | None = None, batch_size: int = 32):
        self.model = cfg.embedding_model
        self.name = f"ollama:{self.model}"
        self.batch_size = batch_size
        self._client = client or httpx.Client(base_url=cfg.ollama_host, timeout=cfg.embedding_timeout_seconds)
        # nomic-embed-text is trained with task prefixes.
        nomic = "nomic" in self.model
        self._doc_prefix = "search_document: " if nomic else ""
        self._query_prefix = "search_query: " if nomic else ""

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), self.batch_size):
            batch = inputs[start:start + self.batch_size]
            try:
                response = self._client.post("/api/embed", json={"model": self.model, "input": batch})
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Ollama embedding request failed: {exc}") from exc
            embeddings = response.json().get("embeddings") or []
            if len(embeddings) != len(batch):
                raise EmbeddingError("Ollama returned an unexpected number of embeddings")
            vectors.extend(embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._query_prefix + text])[0]


_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its me my no not of on or our "
    "so that the their them then there these this to us was we what when where which who why will with you your "
    "get getting just should would could".split()
)


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "e"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic bag-of-words embeddings (feature hashing). Offline use only."""

    min_score = 0.12
    confident_score = 0.45
    relevance_margin = 0.15

    def __init__(self, dimensions: int = 1024):
        self.dimensions = dimensions
        self.name = f"hashing:{dimensions}"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = tokenize(text)
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.md5(feature.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vec[index] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
