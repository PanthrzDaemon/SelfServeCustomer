import re

import pytest

from app.rag.chunker import DEFAULT_MAX_CHARS, chunk_document, chunk_documents, chunk_metadata, split_text, version_key
from app.rag.loader import load_documents
from app.rag.models import SourceDocument

REQUIRED_CHUNK_METADATA = [
    "source_id", "doc_type", "title", "product_versions", "authority_level", "last_updated",
    "effective_from", "deprecated_on", "is_deprecated", "supersedes", "superseded_by", "provenance",
    "synthetic", "chunk_index", "chunk_count", "content_hash", "v4_2", "v4_3",
]


@pytest.fixture(scope="module")
def docs():
    return load_documents()


@pytest.fixture(scope="module")
def chunks(docs):
    return chunk_documents(docs)


def make_doc(content: str, **overrides) -> SourceDocument:
    fields = dict(source_id="KB-TEST", doc_type="article", title="Test", content=content, product_versions=["4.3"],
                  authority_level=1, last_updated="2026-04-01", effective_from="2026-04-01")
    fields.update(overrides)
    return SourceDocument(**fields)


def test_loader_links_superseded_documents(docs):
    by_id = {d.source_id: d for d in docs}
    assert by_id["KB-INT-SALESFORCE-V42"].superseded_by == "KB-INT-SALESFORCE-V43"
    assert by_id["KB-INT-SALESFORCE-V43"].supersedes == "KB-INT-SALESFORCE-V42"
    assert by_id["KB-GS-CREATE-WORKFLOW"].superseded_by is None


def test_every_document_is_chunked(docs, chunks):
    assert {c.source_id for c in chunks} == {d.source_id for d in docs}


def test_chunk_ids_are_stable_and_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r".+-chunk-\d{3}", cid) for cid in ids)
    assert chunk_documents(load_documents())[0].chunk_id == chunks[0].chunk_id


def test_chunks_are_bounded(chunks):
    assert all(len(c.text) <= DEFAULT_MAX_CHARS for c in chunks)


def test_chunks_carry_all_metadata(chunks):
    for chunk in chunks:
        for key in REQUIRED_CHUNK_METADATA:
            assert key in chunk.metadata, (chunk.chunk_id, key)
        # Chroma only accepts flat scalar values.
        assert all(isinstance(v, (str, int, float, bool)) for v in chunk.metadata.values())


def test_embed_text_keeps_document_context(chunks):
    for chunk in chunks:
        assert chunk.embed_text.startswith(chunk.metadata["title"])
        assert "CloudFlow versions:" in chunk.embed_text


def test_chunking_preserves_all_content(docs):
    for doc in docs:
        joined = " ".join(c.text for c in chunk_document(doc))
        for word in doc.content.split():
            assert word in joined, doc.source_id


def test_long_document_is_split_into_ordered_chunks():
    content = "\n\n".join(f"Section {i}. " + "Detail sentence. " * 30 for i in range(6))
    chunks = chunk_document(make_doc(content), max_chars=400)
    assert len(chunks) > 3
    assert [c.chunk_index for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.metadata["chunk_count"] == len(chunks) for c in chunks)


def test_oversized_paragraph_is_split():
    pieces = split_text("Word " * 500, max_chars=300)
    assert len(pieces) > 1 and all(len(p) <= 300 for p in pieces)


def test_metadata_normalization():
    doc = make_doc("x", product_versions=["4.2", "4.3"], deprecated_on=None, supersedes=None)
    meta = chunk_metadata(doc)
    assert meta["product_versions"] == "4.2,4.3"
    assert meta[version_key("4.2")] is True and meta[version_key("4.3")] is True
    assert meta["deprecated_on"] == "" and meta["supersedes"] == "" and meta["is_deprecated"] is False


def test_deprecated_flag():
    assert chunk_metadata(make_doc("x", deprecated_on="2026-04-01"))["is_deprecated"] is True
    assert chunk_metadata(make_doc("x", deprecated_on="2999-01-01"))["is_deprecated"] is False


def test_content_hash_changes_with_content_or_metadata():
    base = chunk_document(make_doc("Hello"))[0].metadata["content_hash"]
    assert chunk_document(make_doc("Hello"))[0].metadata["content_hash"] == base
    assert chunk_document(make_doc("Hello!"))[0].metadata["content_hash"] != base
    assert chunk_document(make_doc("Hello", authority_level=2))[0].metadata["content_hash"] != base
