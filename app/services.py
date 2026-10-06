"""Wiring of providers and stores. Tests build Services with offline providers."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import Settings, settings as default_settings
from app.llm import LLMProvider, OllamaLLMProvider
from app.rag.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from app.rag.retriever import Retriever
from app.rag.vector_store import ChromaVectorStore


@dataclass
class Services:
    cfg: Settings
    db_path: str | Path
    embedder: EmbeddingProvider
    store: ChromaVectorStore
    retriever: Retriever
    llm: LLMProvider | None


def build_services(
    cfg: Settings = default_settings,
    *,
    db_path: str | Path | None = None,
    embedder: EmbeddingProvider | None = None,
    store: ChromaVectorStore | None = None,
    llm: LLMProvider | None = None,
    use_llm: bool | None = None,
) -> Services:
    embedder = embedder or OllamaEmbeddingProvider(cfg)
    store = store or ChromaVectorStore(cfg.chroma_path, cfg.chroma_collection)
    if llm is None and (cfg.llm_enabled if use_llm is None else use_llm):
        llm = OllamaLLMProvider(cfg)
    return Services(
        cfg=cfg,
        db_path=db_path or cfg.database_path,
        embedder=embedder,
        store=store,
        retriever=Retriever(embedder, store),
        llm=llm,
    )


@lru_cache(maxsize=1)
def get_services() -> Services:
    return build_services()
