"""Source precedence is deterministic and independent of the embedding model."""

from app.rag.models import RetrievedSource
from app.rag.ranking import rank_sources

MIN, MARGIN = 0.5, 0.05


def src(source_id, score, versions=("4.2", "4.3"), authority=1, current=True, superseded_by=None, doc_type="article"):
    return RetrievedSource(
        source_id=source_id, title=source_id, doc_type=doc_type, product_versions=list(versions),
        authority_level=authority, content="...", score=score, last_updated="2026-04-01",
        effective_from="2026-04-01", is_current=current, superseded_by=superseded_by,
    )


def ids(ranked):
    return [s.source_id for s in ranked]


def test_version_filter_excludes_other_versions():
    ranked, excluded = rank_sources([src("OLD", 0.9, ["4.2"]), src("NEW", 0.8, ["4.3"])], "4.3", MIN, MARGIN)
    assert ids(ranked) == ["NEW"]
    assert excluded[0].source_id == "OLD" and excluded[0].reason == "version_mismatch"


def test_current_43_doc_beats_more_similar_42_ticket():
    candidates = [src("TKT-OLD", 0.92, ["4.2"], authority=4), src("KB-NEW", 0.85, ["4.3"], authority=1)]
    ranked, _ = rank_sources(candidates, "4.3", MIN, MARGIN)
    assert ids(ranked) == ["KB-NEW"]


def test_without_version_latest_applicable_sources_come_first():
    candidates = [src("TKT-42", 0.95, ["4.2"], authority=4), src("KB-43", 0.70, ["4.3"])]
    ranked, _ = rank_sources(candidates, None, MIN, MARGIN)
    assert ids(ranked) == ["KB-43", "TKT-42"]


def test_authority_beats_slightly_higher_similarity_within_band():
    candidates = [src("TKT", 0.80, authority=4), src("RN", 0.79, authority=2), src("KB", 0.77, authority=1)]
    ranked, _ = rank_sources(candidates, "4.3", MIN, MARGIN)
    assert ids(ranked) == ["KB", "RN", "TKT"]


def test_clearly_better_match_is_not_pushed_out_by_weak_authoritative_source():
    candidates = [src("TKT", 0.85, authority=4), src("KB-WEAK", 0.60, authority=1)]
    ranked, _ = rank_sources(candidates, "4.3", MIN, MARGIN)
    assert ids(ranked) == ["TKT", "KB-WEAK"]


def test_current_beats_deprecated():
    candidates = [src("DEPRECATED", 0.80, current=False), src("CURRENT", 0.78)]
    ranked, _ = rank_sources(candidates, None, MIN, MARGIN)
    assert ids(ranked) == ["CURRENT", "DEPRECATED"]


def test_deprecated_source_kept_when_it_is_the_only_applicable_one():
    ranked, _ = rank_sources([src("ONLY-42", 0.8, ["4.2"], current=False)], "4.2", MIN, MARGIN)
    assert ids(ranked) == ["ONLY-42"]


def test_superseded_source_dropped_when_successor_present():
    candidates = [src("V42", 0.90, ["4.2"], superseded_by="V43"), src("V43", 0.80, ["4.3"])]
    ranked, excluded = rank_sources(candidates, None, MIN, MARGIN)
    assert ids(ranked) == ["V43"]
    assert excluded[0].reason == "superseded_by:V43"


def test_superseded_source_kept_for_its_own_version():
    candidates = [src("V42", 0.90, ["4.2"], superseded_by="V43"), src("V43", 0.80, ["4.3"])]
    ranked, _ = rank_sources(candidates, "4.2", MIN, MARGIN)
    assert ids(ranked) == ["V42"]


def test_relevance_threshold():
    ranked, excluded = rank_sources([src("LOW", 0.3), src("OK", 0.7)], None, MIN, MARGIN)
    assert ids(ranked) == ["OK"]
    assert excluded[0].reason == "below_relevance_threshold"


def test_no_candidates_returns_empty():
    ranked, _ = rank_sources([src("LOW", 0.1)], None, MIN, MARGIN)
    assert ranked == []


def test_ranks_are_assigned_in_order():
    ranked, _ = rank_sources([src("A", 0.8, authority=4), src("B", 0.79)], None, MIN, MARGIN)
    assert [s.rank for s in ranked] == [1, 2]
