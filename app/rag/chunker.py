"""Split source documents into embedding-sized chunks that keep their metadata."""

import hashlib
import re
from datetime import date

from app.rag.models import Chunk, SourceDocument
from app.sources import SUPPORTED_VERSIONS

DEFAULT_MAX_CHARS = 900


def version_key(version: str) -> str:
    """Chroma metadata key for a version flag, e.g. '4.3' -> 'v4_3'."""
    return "v" + version.replace(".", "_")


def is_deprecated(doc: SourceDocument, today: date | None = None) -> bool:
    if not doc.deprecated_on:
        return False
    return date.fromisoformat(doc.deprecated_on) <= (today or date.today())


def chunk_metadata(doc: SourceDocument) -> dict[str, str | int | bool]:
    """Flat, Chroma-compatible metadata (no None, no nested values)."""
    meta: dict[str, str | int | bool] = {
        "source_id": doc.source_id,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "category": doc.category or "",
        "product_versions": ",".join(doc.product_versions),
        "authority_level": doc.authority_level,
        "last_updated": doc.last_updated,
        "effective_from": doc.effective_from,
        "deprecated_on": doc.deprecated_on or "",
        "is_deprecated": is_deprecated(doc),
        "supersedes": doc.supersedes or "",
        "superseded_by": doc.superseded_by or "",
        "provenance": doc.provenance,
        "synthetic": doc.synthetic,
    }
    for version in SUPPORTED_VERSIONS:
        meta[version_key(version)] = version in doc.product_versions
    return meta


def _split_long(block: str, max_chars: int) -> list[str]:
    """Split one oversized block by lines, then sentences, then hard-wrap."""
    pieces: list[str] = []
    for unit in block.split("\n") if "\n" in block else re.split(r"(?<=[.!?])\s+", block):
        if len(unit) <= max_chars:
            pieces.append(unit)
        elif "\n" in block:
            pieces.extend(_split_long(unit, max_chars))
        else:
            pieces.extend(unit[i:i + max_chars] for i in range(0, len(unit), max_chars))
    return _pack(pieces, max_chars, "\n" if "\n" in block else " ")


def _pack(units: list[str], max_chars: int, sep: str) -> list[str]:
    packed: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}{sep}{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                packed.append(current)
            current = unit
    if current:
        packed.append(current)
    return packed


def split_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split on blank lines (paragraphs/sections), packing small ones together."""
    blocks: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        blocks.extend([block] if len(block) <= max_chars else _split_long(block, max_chars))
    return _pack(blocks, max_chars, "\n\n")


def chunk_document(doc: SourceDocument, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    texts = split_text(doc.content, max_chars)
    base = chunk_metadata(doc)
    header = f"{doc.title}\nCloudFlow versions: {', '.join(doc.product_versions)}\n\n"
    chunks = []
    for index, text in enumerate(texts, start=1):
        embed_text = header + text
        meta = {
            **base,
            "chunk_index": index,
            "chunk_count": len(texts),
            "content_hash": hashlib.sha256(
                (embed_text + repr(sorted(base.items()))).encode("utf-8")
            ).hexdigest()[:16],
        }
        chunks.append(Chunk(
            chunk_id=f"{doc.source_id}-chunk-{index:03d}",
            source_id=doc.source_id,
            chunk_index=index,
            text=text,
            embed_text=embed_text,
            metadata=meta,
        ))
    return chunks


def chunk_documents(docs: list[SourceDocument], max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, max_chars)]
