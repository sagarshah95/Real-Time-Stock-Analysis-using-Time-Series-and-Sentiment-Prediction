"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider selection
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    llm_model: str = Field(default="", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    agent_max_iterations: int = Field(default=8, alias="AGENT_MAX_ITERATIONS")

    # RAG
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    chroma_persist_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "chroma"),
        alias="CHROMA_PERSIST_DIR",
    )
    rag_chunk_size: int = Field(default=800, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=120, alias="RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")

    # Hugging Face Hub (embeddings / model downloads)
    hf_token: str = Field(
        default="",
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"),
    )

    # Paths (see data/ and config/paths.py)
    inference_data_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "transcripts"),
        alias="INFERENCE_DATA_DIR",
    )
    sp500_csv: str = Field(
        default=str(PROJECT_ROOT / "data" / "datasets" / "SP500.csv"),
        alias="SP500_CSV",
    )

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_url: str = Field(default="http://localhost:8000", alias="API_URL")

    # MCP
    mcp_host: str = Field(default="0.0.0.0", alias="MCP_HOST")
    mcp_port: int = Field(default=8001, alias="MCP_PORT")

    # Feature flags
    enable_llm_agent: bool = Field(default=True, alias="ENABLE_LLM_AGENT")

    # X (Twitter) API — X_API_KEY / CONSUMER_KEY are equivalent
    x_bearer_token: str = Field(default="", alias="X_BEARER_TOKEN")
    x_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("X_API_KEY", "CONSUMER_KEY"),
    )
    x_api_secret: str = Field(
        default="",
        validation_alias=AliasChoices("X_API_SECRET", "CONSUMER_KEY_SECRET"),
    )
    x_access_token: str = Field(default="", alias="X_ACCESS_TOKEN")
    x_access_token_secret: str = Field(default="", alias="X_ACCESS_TOKEN_SECRET")
    x_auth_mode: str = Field(default="oauth1", alias="X_AUTH_MODE")
    x_stream_poll_seconds: int = Field(default=60, alias="X_STREAM_POLL_SECONDS")
    x_tweet_lookback_days: int = Field(default=7, alias="X_TWEET_LOOKBACK_DAYS")

    # Kafka (X tweet stream)
    kafka_bootstrap_servers: str = Field(default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS")
    kafka_x_tweets_topic: str = Field(default="x-stock-tweets", alias="KAFKA_X_TWEETS_TOPIC")
    kafka_enabled: bool = Field(default=True, alias="KAFKA_ENABLED")

    @property
    def x_configured(self) -> bool:
        oauth1 = all(
            v.strip()
            for v in (
                self.x_api_key,
                self.x_api_secret,
                self.x_access_token,
                self.x_access_token_secret,
            )
        )
        return oauth1 or bool(self.resolved_x_bearer_token)

    def x_credential_status(self) -> dict[str, bool]:
        """Which .env credential groups are populated (values not validated)."""
        return {
            "oauth1_complete": all(
                v.strip()
                for v in (
                    self.x_api_key,
                    self.x_api_secret,
                    self.x_access_token,
                    self.x_access_token_secret,
                )
            ),
            "api_key_set": bool(self.x_api_key.strip()),
            "api_secret_set": bool(self.x_api_secret.strip()),
            "access_token_set": bool(self.x_access_token.strip()),
            "access_token_secret_set": bool(self.x_access_token_secret.strip()),
            "bearer_set": bool(self.resolved_x_bearer_token),
            "auth_mode": self.x_auth_mode.strip().lower() or "oauth1",
        }

    @property
    def resolved_x_bearer_token(self) -> str:
        from urllib.parse import unquote
        raw = self.x_bearer_token.strip()
        return unquote(raw) if raw else ""

    @property
    def x_trends_data_dir(self) -> Path:
        path = PROJECT_ROOT / "data" / "x_trends"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def active_provider(self) -> str:
        return self.llm_provider.strip().lower()

    @property
    def resolved_api_key(self) -> str:
        provider = self.active_provider
        key_map = {
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
        }
        return (key_map.get(provider) or self.openai_api_key).strip()

    @property
    def resolved_base_url(self) -> str:
        provider = self.active_provider
        if provider == "openai":
            return self.openai_base_url
        return _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["openai"])["base_url"]

    @property
    def resolved_model(self) -> str:
        if self.llm_model.strip():
            return self.llm_model.strip()
        provider = self.active_provider
        return _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["openai"])["model"]

    @property
    def llm_configured(self) -> bool:
        return bool(self.resolved_api_key)

    @property
    def hf_token_configured(self) -> bool:
        return bool(self.hf_token.strip())

    @property
    def inference_data_path(self) -> Path:
        from config.paths import transcripts_dir
        return transcripts_dir(self.inference_data_dir)

    @property
    def sp500_path(self) -> Path:
        from config.paths import sp500_csv_path
        return sp500_csv_path(self.sp500_csv)

    @property
    def chroma_path(self) -> Path:
        path = Path(self.chroma_persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


def apply_hf_hub_token(token: str = "", *, force: bool = False) -> None:
    """Expose HF token to huggingface_hub / Chroma embedding downloads."""
    value = token.strip()
    if not value:
        return
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if force or not os.environ.get(key):
            os.environ[key] = value


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    apply_hf_hub_token(settings.hf_token)
    return settings


def reload_settings() -> Settings:
    """Reload .env — call after updating credentials without restarting Python."""
    get_settings.cache_clear()
    settings = get_settings()
    apply_hf_hub_token(settings.hf_token, force=True)
    return settings
