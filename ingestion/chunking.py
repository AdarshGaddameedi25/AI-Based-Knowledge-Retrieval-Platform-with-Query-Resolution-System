import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import tiktoken
from config.settings import settings


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    page_number: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class TextChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def _tokenize(self, text: str) -> List[int]:
        return self._encoder.encode(text)

    def _decode(self, tokens: List[int]) -> str:
        return self._encoder.decode(tokens)

    def chunk(self, text: str, document_id: str, metadata: dict = None) -> List[Chunk]:
        tokens = self._tokenize(text)
        chunks = []
        start = 0
        index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._decode(chunk_tokens)

            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    chunk_index=index,
                    text=chunk_text,
                    token_count=len(chunk_tokens),
                    metadata=metadata or {},
                )
            )

            index += 1
            start += self.chunk_size - self.chunk_overlap
            if start >= len(tokens):
                break

        return chunks

    def chunk_pages(self, pages: List[Tuple[int, str]], document_id: str, metadata: dict = None) -> List[Chunk]:
        all_chunks = []
        global_index = 0

        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            tokens = self._tokenize(page_text)
            start = 0

            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self._decode(chunk_tokens)

                chunk_meta = dict(metadata or {})
                chunk_meta["page_number"] = page_num

                all_chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        chunk_index=global_index,
                        text=chunk_text,
                        token_count=len(chunk_tokens),
                        page_number=page_num,
                        metadata=chunk_meta,
                    )
                )

                global_index += 1
                start += self.chunk_size - self.chunk_overlap
                if start >= len(tokens):
                    break

        return all_chunks
