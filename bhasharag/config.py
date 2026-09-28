"""Central configuration, read from environment variables / a `.env` file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


@dataclass
class Settings:
    # Storage
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _env("DATA_DIR", "data"))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 50))

    # Embeddings (multilingual: Hindi + English share one vector space)
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    )
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 32))

    # Chunking (measured in *embedding-model tokens*, not characters)
    chunk_tokens: int = field(default_factory=lambda: _env_int("CHUNK_TOKENS", 300))
    chunk_overlap_tokens: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_TOKENS", 50))

    # Retrieval
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    candidates_per_retriever: int = field(default_factory=lambda: _env_int("CANDIDATES", 25))
    use_reranker: bool = field(default_factory=lambda: _env_bool("USE_RERANKER", True))
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    )

    # OCR fallback for scanned PDFs (needs Tesseract installed with `hin` + `eng` data)
    enable_ocr: bool = field(default_factory=lambda: _env_bool("ENABLE_OCR", False))
    # Path to tesseract.exe; empty = use PATH, then the default Windows install folder.
    tesseract_cmd: str = field(default_factory=lambda: _env("TESSERACT_CMD", ""))

    # Generation
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "none").lower())
    rewrite_followups: bool = field(default_factory=lambda: _env_bool("REWRITE_FOLLOWUPS", True))
    max_history_turns: int = field(default_factory=lambda: _env_int("MAX_HISTORY_TURNS", 6))

    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:3b"))

    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    )
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "llama-3.3-70b-versatile"))

    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-5"))
    anthropic_effort: str = field(default_factory=lambda: _env("ANTHROPIC_EFFORT", "medium"))

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "knowledge_base.sqlite3"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "vectors.faiss"

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
