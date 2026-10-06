"""Ingest knowledge sources into ChromaDB.

Usage: python -m app.rag.ingest [--reset]

Idempotent: chunks are keyed by stable IDs (e.g. KB-API-AUTH-chunk-001) and a
content hash. Unchanged chunks are not re-embedded, changed chunks are
re-embedded, and chunks whose source disappeared are deleted.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from app.rag.chunker import chunk_documents
from app.rag.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from app.rag.loader import load_documents
from app.rag.vector_store import ChromaVectorStore
from app.sources import DATA_DIR


class IngestStats(BaseModel):
    embedding_model: str
    collection: str
    documents: int
    documents_by_type: dict[str, int]
    chunks: int
    embedded: int
    unchanged: int
    deleted: int
    collection_count: int
    collection_reset: bool


def ingest(
    embedder: EmbeddingProvider,
    store: ChromaVectorStore,
    data_dir: Path = DATA_DIR,
    reset: bool = False,
) -> IngestStats:
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)

    if reset:
        try:
            store.client.delete_collection(store.collection_name)
        except Exception:
            pass
    collection_reset = store.prepare(embedder.name) or reset

    existing = store.existing_hashes()
    wanted = {c.chunk_id: c for c in chunks}
    to_embed = [c for c in chunks if existing.get(c.chunk_id) != c.metadata["content_hash"]]
    stale = [cid for cid in existing if cid not in wanted]

    if to_embed:
        vectors = embedder.embed_documents([c.embed_text for c in to_embed])
        store.upsert(
            ids=[c.chunk_id for c in to_embed],
            embeddings=vectors,
            documents=[c.text for c in to_embed],
            metadatas=[c.metadata for c in to_embed],
        )
    store.delete(stale)

    return IngestStats(
        embedding_model=embedder.name,
        collection=store.collection_name,
        documents=len(docs),
        documents_by_type=dict(Counter(d.doc_type for d in docs)),
        chunks=len(chunks),
        embedded=len(to_embed),
        unchanged=len(chunks) - len(to_embed),
        deleted=len(stale),
        collection_count=store.count(),
        collection_reset=collection_reset,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest CloudFlow knowledge into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="drop the collection and rebuild it")
    args = parser.parse_args()

    from app.rag.embeddings import EmbeddingError

    try:
        stats = ingest(OllamaEmbeddingProvider(), ChromaVectorStore(), reset=args.reset)
    except EmbeddingError as exc:
        print(f"Ingestion failed: {exc}\nIs Ollama running with the embedding model pulled?", file=sys.stderr)
        return 1

    by_type = stats.documents_by_type
    print("RAG INGESTION")
    print("-------------")
    print(f"Embedding model:   {stats.embedding_model}")
    print(f"Collection:        {stats.collection}")
    print(f"Articles:          {by_type.get('article', 0)}")
    print(f"Release notes:     {by_type.get('release_note', 0)}")
    print(f"Tickets:           {by_type.get('resolved_ticket', 0)}")
    print(f"Policies:          {by_type.get('policy', 0)}")
    print(f"Documents:         {stats.documents}")
    print(f"Chunks:            {stats.chunks}")
    print(f"Embedded:          {stats.embedded}")
    print(f"Unchanged:         {stats.unchanged}")
    print(f"Deleted (stale):   {stats.deleted}")
    print(f"Collection count:  {stats.collection_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
