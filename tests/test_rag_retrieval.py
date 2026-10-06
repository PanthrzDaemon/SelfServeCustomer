"""Ingestion and retrieval against a real (temporary) ChromaDB with offline embeddings."""

import pytest

from app.rag.chunker import chunk_documents
from app.rag.embeddings import HashingEmbeddingProvider
from app.rag.ingest import ingest
from app.rag.loader import load_documents
from app.rag.retriever import RetrievalError, Retriever
from app.rag.vector_store import ChromaVectorStore
from app.sources import all_sources, load_sources

ALL_SOURCE_IDS = {d["source_id"] for d in all_sources(load_sources())}


@pytest.fixture
def fresh_store(tmp_path):
    return ChromaVectorStore(tmp_path / "chroma", "fresh")


# ------------------------------------------------------------------ ingestion

def test_ingest_stores_every_chunk(fresh_store):
    stats = ingest(HashingEmbeddingProvider(), fresh_store)
    expected_chunks = len(chunk_documents(load_documents()))
    assert stats.documents == len(ALL_SOURCE_IDS)
    assert stats.documents_by_type == {"article": 45, "release_note": 2, "resolved_ticket": 30, "policy": 6}
    assert stats.chunks == stats.embedded == stats.collection_count == expected_chunks
    assert fresh_store.source_ids() == ALL_SOURCE_IDS


def test_ingest_is_idempotent(fresh_store):
    ingest(HashingEmbeddingProvider(), fresh_store)
    again = ingest(HashingEmbeddingProvider(), fresh_store)
    assert again.embedded == 0 and again.deleted == 0
    assert again.unchanged == again.chunks == again.collection_count


def test_ingest_removes_stale_chunks(fresh_store):
    embedder = HashingEmbeddingProvider()
    ingest(embedder, fresh_store)
    fresh_store.upsert(["GHOST-chunk-001"], embedder.embed_documents(["ghost"]), ["ghost"],
                       [{"source_id": "GHOST", "content_hash": "x"}])
    stats = ingest(embedder, fresh_store)
    assert stats.deleted == 1
    assert "GHOST" not in fresh_store.source_ids()


def test_changing_embedding_model_rebuilds_collection(fresh_store):
    ingest(HashingEmbeddingProvider(512), fresh_store)
    stats = ingest(HashingEmbeddingProvider(1024), fresh_store)
    assert stats.collection_reset is True
    assert stats.embedded == stats.chunks
    assert fresh_store.embedding_model() == "hashing:1024"


def test_retriever_rejects_collection_built_with_other_model(offline_store):
    with pytest.raises(RetrievalError):
        Retriever(HashingEmbeddingProvider(512), offline_store).retrieve("api keys")


def test_no_operational_account_data_in_vector_store(offline_store):
    text = "\n".join(offline_store.all_documents())
    for marker in ("ACC-1001", "INV-2026-", "owner@"):
        assert marker not in text


# ------------------------------------------------------------------ retrieval

def test_results_are_structured_and_citable(offline_retriever):
    result = offline_retriever.retrieve("How do I create my first workflow?", "4.3")
    assert not result.no_reliable_source
    top = result.sources[0]
    assert top.source_id in ALL_SOURCE_IDS
    assert top.citation == f"[{top.source_id}] {top.title}"
    assert top.content and top.chunk_ids and top.rank == 1
    assert isinstance(top.product_versions, list) and isinstance(top.authority_level, int)
    assert 0 < top.score <= 1


def test_every_result_comes_from_the_store(offline_retriever, offline_store):
    stored = offline_store.source_ids()
    for query in ("billing invoice", "salesforce oauth", "webhook signature", "refund"):
        for source in offline_retriever.retrieve(query, "4.3").sources:
            assert source.source_id in stored


def test_salesforce_43_prefers_current_documentation(offline_retriever):
    result = offline_retriever.retrieve("How do I authenticate with Salesforce?", "4.3")
    assert result.sources[0].source_id == "KB-INT-SALESFORCE-V43"
    returned = result.source_ids
    assert "KB-INT-SALESFORCE-V42" not in returned and "TKT-1007" not in returned
    assert all("4.3" in s.product_versions for s in result.sources)


def test_http_429_retrieves_rate_limit_docs(offline_retriever):
    result = offline_retriever.retrieve("Why am I getting HTTP 429?", "4.3")
    assert {"KB-API-HTTP-429", "KB-API-RATE-LIMITS-V43"} & set(result.source_ids[:3])


def test_export_history_prefers_43_docs_for_43(offline_retriever):
    result = offline_retriever.retrieve("How do I export workflow history?", "4.3")
    assert "KB-GS-EXPORT-HISTORY-V43" in result.source_ids
    assert "KB-GS-EXPORT-HISTORY-V42" not in result.source_ids


def test_export_history_uses_42_docs_for_42(offline_retriever):
    result = offline_retriever.retrieve("How do I export workflow history?", "4.2")
    assert "KB-GS-EXPORT-HISTORY-V42" in result.source_ids
    assert "KB-GS-EXPORT-HISTORY-V43" not in result.source_ids


def test_conflicting_old_ticket_loses_to_current_docs(offline_retriever):
    # TKT-1003 (4.2) recommends extra API keys; current 4.3 docs say limits are per workspace.
    query = "429 errors: create a second API key and split traffic between keys to double throughput"
    result = offline_retriever.retrieve(query, "4.3")
    assert "TKT-1003" not in result.source_ids
    assert {"KB-API-RATE-LIMITS-V43", "KB-API-HTTP-429", "KB-API-AUTH"} & set(result.source_ids)


def test_without_version_superseded_docs_are_dropped(offline_retriever):
    result = offline_retriever.retrieve("Salesforce integration username password security token OAuth", None)
    if "KB-INT-SALESFORCE-V43" in result.source_ids:
        assert "KB-INT-SALESFORCE-V42" not in result.source_ids


def test_doc_type_filter(offline_retriever):
    result = offline_retriever.retrieve("refund", "4.3", doc_types=["policy"])
    assert result.sources and all(s.doc_type == "policy" for s in result.sources)


def test_no_reliable_source_for_unknown_topic(offline_retriever):
    result = offline_retriever.retrieve("Does CloudFlow integrate with SAP Ariba?", "4.3")
    assert result.no_reliable_source and result.sources == []
    assert "No reliable source" in result.message


def test_empty_query_is_no_reliable_source(offline_retriever):
    assert offline_retriever.retrieve("   ").no_reliable_source


def test_unsupported_version_rejected(offline_retriever):
    with pytest.raises(ValueError):
        offline_retriever.retrieve("api", "5.0")


def test_fetch_sources_by_id(offline_retriever):
    sources = offline_retriever.fetch_sources(["POLICY-REFUND-01", "DOES-NOT-EXIST"], "refund")
    assert [s.source_id for s in sources] == ["POLICY-REFUND-01"]
    assert sources[0].metadata["selected_by"] == "policy_registry"
