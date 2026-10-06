"""Shared fixtures. Offline fixtures need no Ollama: they use deterministic
hashing embeddings, a temporary ChromaDB and a scripted fake LLM."""

import pytest

from app.agents.graph import SupportAgent
from app.config import settings
from app.db.seed import seed
from app.llm import LLMError, LLMProvider, ollama_available
from app.rag.embeddings import HashingEmbeddingProvider
from app.rag.ingest import ingest
from app.rag.retriever import Retriever
from app.rag.vector_store import ChromaVectorStore
from app.services import build_services


class ScriptedLLM(LLMProvider):
    """Fake LLM returning scripted answers in order (the last one repeats).

    A callable entry receives (system, prompt) and returns the answer, so a test
    can build answers from the prompt it was given.
    """

    name = "scripted"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response(system, prompt) if callable(response) else response


class FailingLLM(LLMProvider):
    name = "failing"

    def generate(self, system: str, prompt: str) -> str:
        raise LLMError("offline")


@pytest.fixture(scope="session")
def offline_store(tmp_path_factory):
    store = ChromaVectorStore(tmp_path_factory.mktemp("chroma"), "test_knowledge")
    ingest(HashingEmbeddingProvider(), store)
    return store


@pytest.fixture(scope="session")
def offline_retriever(offline_store):
    return Retriever(HashingEmbeddingProvider(), offline_store)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "cloudflow.db"
    seed(path)
    return path


def make_services(db_path, store, llm=None):
    return build_services(db_path=db_path, embedder=HashingEmbeddingProvider(), store=store, llm=llm, use_llm=False)


@pytest.fixture
def offline_services(db_path, offline_store):
    return make_services(db_path, offline_store)


@pytest.fixture
def agent(offline_services):
    return SupportAgent(offline_services)


requires_ollama = pytest.mark.skipif(not ollama_available(settings), reason="Ollama is not running")
