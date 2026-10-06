"""ChromaDB wrapper for the CloudFlow knowledge collection.

Only knowledge sources (articles, release notes, resolved tickets, policies) are
stored here. Account data such as usage and invoices stays in SQLite.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings as default_settings


@dataclass
class ChunkHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    similarity: float


class ChromaVectorStore:
    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str | None = None,
        client: Any | None = None,
    ):
        self.collection_name = collection_name or default_settings.chroma_collection
        self.client = client or chromadb.PersistentClient(
            path=str(path or default_settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _collection(self, embedding_model: str | None = None):
        metadata = {"hnsw:space": "cosine"}
        if embedding_model:
            metadata["embedding_model"] = embedding_model
        # embedding_function=None: embeddings always come from our EmbeddingProvider.
        return self.client.get_or_create_collection(
            name=self.collection_name, metadata=metadata, embedding_function=None
        )

    def embedding_model(self) -> str | None:
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            return None
        return (collection.metadata or {}).get("embedding_model")

    def prepare(self, embedding_model: str) -> bool:
        """Ensure the collection exists for this embedding model.

        Returns True if an existing collection was dropped because it was built
        with a different embedding model (vectors would not be comparable).
        """
        existing = self.embedding_model()
        reset = existing is not None and existing != embedding_model
        if reset:
            self.client.delete_collection(self.collection_name)
        self._collection(embedding_model)
        return reset

    def existing_hashes(self) -> dict[str, str]:
        collection = self._collection()
        data = collection.get(include=["metadatas"])
        return {cid: (meta or {}).get("content_hash", "") for cid, meta in zip(data["ids"], data["metadatas"])}

    def upsert(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        if ids:
            self._collection().upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection().delete(ids=ids)

    def count(self) -> int:
        return self._collection().count()

    def all_documents(self) -> list[str]:
        return self._collection().get(include=["documents"])["documents"] or []

    def source_ids(self) -> set[str]:
        data = self._collection().get(include=["metadatas"])
        return {meta["source_id"] for meta in data["metadatas"] if meta}

    def get_source_chunks(self, source_ids: list[str]) -> list[tuple[str, str, dict, list[float]]]:
        """All chunks of the given sources as (chunk_id, text, metadata, embedding)."""
        if not source_ids:
            return []
        data = self._collection().get(where={"source_id": {"$in": source_ids}},
                                      include=["documents", "metadatas", "embeddings"])
        return list(zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"]))

    def query(self, embedding: list[float], n_results: int, where: dict | None = None) -> list[ChunkHit]:
        collection = self._collection()
        total = collection.count()
        if total == 0:
            return []
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            ChunkHit(chunk_id=cid, text=doc, metadata=meta, similarity=1.0 - dist)
            for cid, doc, meta, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
