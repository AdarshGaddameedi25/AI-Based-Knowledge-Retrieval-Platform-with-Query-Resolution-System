from dataclasses import dataclass, field
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


@dataclass
class EmbeddingRecord:
    chunk_id: str
    document_id: str
    chunk_index: int
    source_file: str
    embedding: List[float]
    model: str
    page_number: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            EmbeddingService._model = SentenceTransformer(settings.embedding_model)
            logger.info("Embedding model loaded")

    def generate_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        cleaned = [t if t and t.strip() else " " for t in texts]
        vectors = self._model.encode(cleaned, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_chunks(self, chunks, source_file: str) -> List[EmbeddingRecord]:
        texts = [c.text for c in chunks]
        vectors = self.generate_embeddings(texts)
        records = []
        for chunk, vector in zip(chunks, vectors):
            records.append(
                EmbeddingRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    source_file=source_file,
                    embedding=vector,
                    model=settings.embedding_model,
                    page_number=chunk.page_number,
                    metadata=chunk.metadata,
                )
            )
        return records
