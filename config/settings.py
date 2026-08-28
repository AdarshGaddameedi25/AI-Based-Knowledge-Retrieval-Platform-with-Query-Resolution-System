from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    chroma_persist_dir: str = "./data/vector_store"
    chroma_collection_name: str = "knowledge_base"

    chunk_size: int = 500
    chunk_overlap: int = 50

    top_k_results: int = 5

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    log_level: str = "INFO"


settings = Settings()
