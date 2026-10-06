"""Version-aware retrieval over the knowledge collection.

retrieve() returns structured, ranked, citable sources, or an explicit
"no reliable source" result. It never invents sources: every result comes from
a chunk stored in ChromaDB.
"""

import re

from app.config import settings as default_settings
from app.rag.chunker import version_key
from app.rag.embeddings import EmbeddingProvider, tokenize
from app.rag.models import NO_RELIABLE_SOURCE_MESSAGE, ExcludedSource, RetrievalResult, RetrievedSource
from app.rag.ranking import rank_sources
from app.rag.vector_store import ChromaVectorStore, ChunkHit
from app.sources import SUPPORTED_VERSIONS

MAX_CHUNKS_PER_SOURCE = 2

# Everyday words that carry no topic, so their absence from the knowledge base
# says nothing about whether it can answer the question.
COMMON_WORDS = frozenset("""
able about after again all also always any anyone anything around ask asked back because been before being
best better both call came come could day days did different does doing done each else even ever every
explain feel few find fine first follow get give given go going good got great happen happened happening
hello help hey hi know last let like look looking lot made make many may maybe mean might more most much
must need needed new next now off okay once one only other others ought out over own part please possible
quick really right said same say see seem seems show since some someone something soon still such sure take
tell thank thanks thing things think though thought through time today told too try trying turn two under
until upon use used using very want wanted way week well went were while whole why work working works yes
yet yesterday
""".split())


class RetrievalError(RuntimeError):
    pass


class Retriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: ChromaVectorStore,
        min_score: float | None = None,
        confident_score: float | None = None,
        margin: float | None = None,
        candidate_multiplier: int = 6,
    ):
        self.embedder = embedder
        self.store = store
        self.min_score = min_score if min_score is not None else (
            default_settings.retrieval_min_score or embedder.min_score)
        self.confident_score = confident_score if confident_score is not None else (
            default_settings.retrieval_confident_score or embedder.confident_score)
        self.margin = margin if margin is not None else (
            default_settings.retrieval_relevance_margin or embedder.relevance_margin)
        self.candidate_multiplier = candidate_multiplier
        self._vocabulary: set[str] | None = None

    # ------------------------------------------------------------ helpers

    def _check_collection(self) -> None:
        built_with = self.store.embedding_model()
        if built_with and built_with != self.embedder.name:
            raise RetrievalError(
                f"collection was built with '{built_with}' but the retriever uses '{self.embedder.name}'; "
                "run: python -m app.rag.ingest"
            )

    def vocabulary(self) -> set[str]:
        """Stemmed vocabulary of the whole knowledge base (for coverage checks)."""
        if self._vocabulary is None:
            vocab: set[str] = set()
            for text in self.store.all_documents():
                vocab.update(tokenize(text))
            self._vocabulary = vocab
        return self._vocabulary

    def unknown_terms(self, query: str) -> list[str]:
        """Query terms that appear nowhere in the knowledge base.

        Prefix matching (5 chars) tolerates simple word-form differences.
        """
        vocab = self.vocabulary()
        prefixes = {t[:5] for t in vocab if len(t) >= 5}
        unknown = []
        for term in tokenize(re.sub(r"\[[A-Z_]+\]", " ", query)):
            if len(term) < 3 or term.isdigit() or term in vocab or term in COMMON_WORDS:
                continue
            if len(term) >= 5 and term[:5] in prefixes:
                continue
            unknown.append(term)
        return unknown

    @staticmethod
    def _where(product_version: str | None, doc_types: list[str] | None) -> dict | None:
        clauses = []
        if product_version:
            clauses.append({version_key(product_version): True})
        if doc_types:
            clauses.append({"doc_type": {"$in": doc_types}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @staticmethod
    def _group(hits: list[ChunkHit]) -> list[RetrievedSource]:
        grouped: dict[str, list[ChunkHit]] = {}
        for hit in hits:
            grouped.setdefault(hit.metadata["source_id"], []).append(hit)

        sources = []
        for source_id, source_hits in grouped.items():
            best = sorted(source_hits, key=lambda h: -h.similarity)[:MAX_CHUNKS_PER_SOURCE]
            best.sort(key=lambda h: h.metadata.get("chunk_index", 0))
            meta = best[0].metadata
            sources.append(RetrievedSource(
                source_id=source_id,
                title=meta["title"],
                doc_type=meta["doc_type"],
                product_versions=[v for v in str(meta["product_versions"]).split(",") if v],
                authority_level=int(meta["authority_level"]),
                content="\n\n".join(h.text for h in best),
                score=round(max(h.similarity for h in source_hits), 4),
                last_updated=meta["last_updated"],
                effective_from=meta["effective_from"],
                deprecated_on=meta.get("deprecated_on") or None,
                supersedes=meta.get("supersedes") or None,
                superseded_by=meta.get("superseded_by") or None,
                is_current=not meta.get("is_deprecated", False),
                chunk_ids=[h.chunk_id for h in best],
                metadata={k: v for k, v in meta.items() if k not in ("content_hash",)},
            ))
        return sources

    # ---------------------------------------------------------------- API

    def fetch_sources(self, source_ids: list[str], query: str) -> list[RetrievedSource]:
        """Load specific sources by ID (e.g. the active policies from policy_registry).

        Each gets the cosine similarity of its best chunk to the query as its score.
        Only sources that exist in the collection are returned.
        """
        rows = self.store.get_source_chunks(source_ids)
        if not rows:
            return []
        q = self.embedder.embed_query(query)
        q_norm = sum(v * v for v in q) ** 0.5 or 1.0
        hits = []
        for chunk_id, text, meta, emb in rows:
            e_norm = sum(float(v) * float(v) for v in emb) ** 0.5 or 1.0
            similarity = sum(a * float(b) for a, b in zip(q, emb)) / (q_norm * e_norm)
            hits.append(ChunkHit(chunk_id=chunk_id, text=text, metadata=meta, similarity=similarity))
        sources = self._group(hits)
        for source in sources:
            source.metadata["selected_by"] = "policy_registry"
        order = {sid: i for i, sid in enumerate(source_ids)}
        return sorted(sources, key=lambda s: order.get(s.source_id, len(order)))

    def retrieve(
        self,
        query: str,
        product_version: str | None = None,
        top_k: int = 5,
        doc_types: list[str] | None = None,
    ) -> RetrievalResult:
        if product_version is not None and product_version not in SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported product version: {product_version}")
        if not query.strip():
            return RetrievalResult(query=query, product_version=product_version, sources=[],
                                   no_reliable_source=True, message=NO_RELIABLE_SOURCE_MESSAGE)
        self._check_collection()

        embedding = self.embedder.embed_query(query)
        hits = self.store.query(
            embedding,
            n_results=max(top_k * self.candidate_multiplier, 20),
            where=self._where(product_version, doc_types),
        )
        ranked, excluded = rank_sources(self._group(hits), product_version, self.min_score, self.margin)

        message = None
        if ranked:
            unknown = self.unknown_terms(query)
            if unknown and ranked[0].score < self.confident_score:
                # The query is about something the knowledge base never mentions,
                # and the best match isn't strong enough to trust anyway.
                excluded.extend(
                    ExcludedSource(source_id=s.source_id, score=s.score, reason="query_terms_not_covered")
                    for s in ranked
                )
                ranked = []
                message = f"{NO_RELIABLE_SOURCE_MESSAGE} Terms not covered: {', '.join(unknown[:5])}."

        if not ranked:
            return RetrievalResult(
                query=query, product_version=product_version, sources=[], no_reliable_source=True,
                message=message or NO_RELIABLE_SOURCE_MESSAGE, excluded=excluded,
            )
        return RetrievalResult(
            query=query, product_version=product_version, sources=ranked[:top_k], excluded=excluded,
        )
