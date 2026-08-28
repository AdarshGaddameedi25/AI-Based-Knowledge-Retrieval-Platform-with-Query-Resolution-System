import os
from dataclasses import dataclass, field
from typing import Optional
import openai

from config.settings import settings


@dataclass
class EmbeddingRecord:
    chunk_id: str
    document_id: str
    chunk_index: int
    source_file: str
    embedding: list[float]
    model: str
    page_number: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class EmbeddingService:
    def __init__(self):
        if not settings.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file before generating embeddings."
            )
        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model

    def embed_text(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_chunks(self, chunks, source_file: str) -> list[EmbeddingRecord]:
        records = []
        for chunk in chunks:
            vector = self.embed_text(chunk.text)
            records.append(
                EmbeddingRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    source_file=source_file,
                    embedding=vector,
                    model=self.model,
                    page_number=chunk.page_number,
                    metadata=chunk.metadata,
                )
            )
        return records
