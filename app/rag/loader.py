"""Load knowledge sources (articles, tickets, policies) for ingestion."""

from pathlib import Path

from app.rag.models import SourceDocument
from app.sources import DATA_DIR, all_sources, load_sources, validate_source_metadata


def load_documents(data_dir: Path = DATA_DIR) -> list[SourceDocument]:
    """Load and validate every knowledge source, linking superseded documents."""
    raw = all_sources(load_sources(data_dir))
    errors = validate_source_metadata(raw)
    if errors:
        raise ValueError("invalid source metadata: " + "; ".join(errors[:5]))

    superseded_by = {d["supersedes"]: d["source_id"] for d in raw if d.get("supersedes")}
    fields = set(SourceDocument.model_fields)
    return [
        SourceDocument(**{k: v for k, v in d.items() if k in fields}, superseded_by=superseded_by.get(d["source_id"]))
        for d in raw
    ]
