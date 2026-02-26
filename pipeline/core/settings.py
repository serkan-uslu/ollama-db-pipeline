"""
pipeline/core/settings.py

Reads all configuration from environment variables / .env file.
Never hardcode secrets — always import `settings` from here.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM / API Keys ---
    groq_api_key: str
    cross_repo_token: str

    # --- GitHub PR Target ---
    github_target_repo: str = "serkan-uslu/ollama-explorer"
    github_target_branch: str = "data/models-update"

    # --- Crawl Config ---
    ollama_library_url: str = "https://ollama.com/library"
    request_delay: float = 0.5

    # --- Enrichment Config ---
    enrich_version: int = 1
    llm_model: str = "llama-3.3-70b-versatile"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
