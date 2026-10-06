from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables and `.env`.

    LLM and embedding providers are referenced only through configuration here;
    provider clients live behind their own interfaces (app/llm.py, app/rag/embeddings.py).
    """

    app_name: str = "CloudFlow AI Support"
    environment: str = "development"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3"
    embedding_model: str = "nomic-embed-text"
    # Qwen3 thinking models reason before answering; the reasoning is discarded.
    # With ollama_raw_chatml the prompt is sent in Qwen's ChatML format with an
    # empty <think></think> block pre-filled, which keeps reasoning short (about
    # 3x faster on qwen3:4b). Set it to false to use /api/chat with `think` instead.
    ollama_raw_chatml: bool = True
    ollama_think: bool = True
    ollama_num_ctx: int = 8192
    llm_enabled: bool = True
    llm_timeout_seconds: float = 300.0
    llm_max_tokens: int = 1500
    embedding_timeout_seconds: float = 60.0

    chroma_path: str = "./chroma"
    chroma_collection: str = "cloudflow_knowledge"
    database_path: str = "./cloudflow.db"

    # Retrieval thresholds (cosine similarity). Unset = the embedding provider's defaults.
    retrieval_min_score: float | None = None
    retrieval_confident_score: float | None = None
    retrieval_relevance_margin: float | None = None

    # Optional LLM groundedness judge in the critic (deterministic checks always run).
    critic_llm_judge: bool = False

    # If set, POST /ingest requires the X-Admin-Token header.
    admin_token: str | None = None

    # Used by the Streamlit frontend.
    api_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
