"""
pipeline/core/settings.py

Reads all configuration from environment variables / .env file.
Never hardcode secrets — always import `settings` from here.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM / API Keys (required at enrich + PR steps only) ---
    groq_api_key: str | None = None
    cross_repo_token: str | None = None

    # --- GitHub PR Target ---
    github_target_repo: str = "serkan-uslu/ollama-explorer"
    github_target_branch: str = "data/models-update"

    # --- Crawl Config ---
    ollama_library_url: str = "https://ollama.com/library"
    request_delay: float = 0.5

    # --- Enrichment Config ---
    enrich_version: int = 1
    llm_model: str = "llama3.3:70b"
    # "groq" veya "ollama"
    llm_provider: str = "ollama"
    # Ollama local API (OpenAI uyumlu)
    ollama_base_url: str = "http://localhost:11434/v1"
    # Paralel enrichment worker sayısı (Ollama için 2-4 önerilir)
    enrich_workers: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # ignore unknown env vars (e.g. PROJECT_NAME from old .env)


settings = Settings()
