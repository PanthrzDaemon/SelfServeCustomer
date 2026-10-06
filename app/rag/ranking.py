"""Deterministic source precedence applied after vector retrieval.

Similarity alone decides nothing beyond relevance. The steps are:

1. Relevance gate: sources below `min_score` are dropped.
2. Product version: when a version is given, sources that don't apply to it are dropped.
3. Supersession: a source is dropped when the source that supersedes it is also
   a candidate and applies to the requested version.
4. Ordering:
     a. product-version match (with no version given, sources that apply to the
        latest version come before sources that only apply to older ones)
     b. relevance band: sources within `margin` of the best similarity first
     c. current before deprecated
     d. authority level (1 = current docs/policy ... 4 = historical ticket)
     e. similarity, highest first
   So a current 4.3 article beats a more similar 4.2-era ticket, and among
   equally applicable sources an authoritative document that is barely
   relevant can't push out a clearly better match.
"""

from app.rag.models import ExcludedSource, RetrievedSource
from app.sources import SUPPORTED_VERSIONS

LATEST_VERSION = max(SUPPORTED_VERSIONS, key=lambda v: tuple(int(p) for p in v.split(".")))


def precedence_key(source: RetrievedSource, product_version: str | None, best_score: float, margin: float) -> tuple:
    band = 0 if source.score >= best_score - margin else 1
    version_rank = 0 if source.applies_to(product_version or LATEST_VERSION) else 1
    currency_rank = 0 if source.is_current else 1
    return (version_rank, band, currency_rank, source.authority_level, -source.score, source.source_id)


def rank_sources(
    candidates: list[RetrievedSource],
    product_version: str | None,
    min_score: float,
    margin: float,
) -> tuple[list[RetrievedSource], list[ExcludedSource]]:
    """Return (ranked sources, excluded sources with reasons)."""
    excluded: list[ExcludedSource] = []
    kept: list[RetrievedSource] = []

    for source in candidates:
        if source.score < min_score:
            excluded.append(ExcludedSource(source_id=source.source_id, score=source.score, reason="below_relevance_threshold"))
        elif not source.applies_to(product_version):
            excluded.append(ExcludedSource(source_id=source.source_id, score=source.score, reason="version_mismatch"))
        else:
            kept.append(source)

    by_id = {s.source_id: s for s in kept}
    current: list[RetrievedSource] = []
    for source in kept:
        successor = by_id.get(source.superseded_by or "")
        if successor is not None and successor.applies_to(product_version):
            excluded.append(ExcludedSource(
                source_id=source.source_id, score=source.score, reason=f"superseded_by:{successor.source_id}"
            ))
        else:
            current.append(source)

    if not current:
        return [], excluded

    best = max(s.score for s in current)
    ranked = sorted(current, key=lambda s: precedence_key(s, product_version, best, margin))
    for position, source in enumerate(ranked, start=1):
        source.rank = position
    return ranked, excluded
