from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    app_name: str = "PDF RAG Demo"
    mistral_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MISTRAL_API_KEY", "Mistral_API_KEY"),
    )
    mistral_chat_model: str = "mistral-small-latest"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_chunk_chars: int = 900
    chunk_overlap: int = 150
    semantic_top_k: int = 8
    keyword_top_k: int = 8
    final_top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
DATA_DIR.mkdir(parents=True, exist_ok=True)
