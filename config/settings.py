from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_knowledge_retrieval"

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    embedding_model: str = "all-MiniLM-L6-v2"

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 5

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    log_level: str = "INFO"
    allowed_origins: str = "http://localhost:3000,http://localhost:8080,http://127.0.0.1:8080"

    @property
    def origins_list(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
