"""Data models for the RAG layer."""

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """A knowledge source (article, release note, ticket or policy) with its metadata."""

    source_id: str
    doc_type: str
    title: str
    content: str
    product_versions: list[str]
    authority_level: int
    last_updated: str
    effective_from: str
    deprecated_on: str | None = None
    supersedes: str | None = None
    # Reverse of `supersedes`, computed at load time.
    superseded_by: str | None = None
    category: str | None = None
    provenance: str = "synthetic"
    synthetic: bool = True


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    # What gets embedded: a short header (title, versions) plus the chunk text,
    # so a chunk never loses the context of the document it came from.
    embed_text: str
    metadata: dict[str, str | int | float | bool]


class RetrievedSource(BaseModel):
    """One ranked source returned by the retriever, ready to cite."""

    source_id: str
    title: str
    doc_type: str
    product_versions: list[str]
    authority_level: int
    content: str
    score: float = Field(description="Best cosine similarity of the source's chunks to the query")
    last_updated: str
    effective_from: str
    deprecated_on: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    is_current: bool = True
    chunk_ids: list[str] = Field(default_factory=list)
    rank: int = 0
    metadata: dict = Field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"[{self.source_id}] {self.title}"

    def applies_to(self, product_version: str | None) -> bool:
        return product_version is None or product_version in self.product_versions


class ExcludedSource(BaseModel):
    source_id: str
    score: float
    reason: str


class RetrievalResult(BaseModel):
    query: str
    product_version: str | None
    sources: list[RetrievedSource]
    no_reliable_source: bool = False
    message: str | None = None
    excluded: list[ExcludedSource] = Field(default_factory=list)

    @property
    def source_ids(self) -> list[str]:
        return [s.source_id for s in self.sources]


NO_RELIABLE_SOURCE_MESSAGE = "No reliable source found in the CloudFlow knowledge base for this question."
