import os
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings


class VectorStore:
    def __init__(self):
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, embedding_records: list) -> None:
        if not embedding_records:
            return

        ids = [r.chunk_id for r in embedding_records]
        embeddings = [r.embedding for r in embedding_records]
        documents = []
        metadatas = []

        for r in embedding_records:
            meta = {
                "document_id": r.document_id,
                "chunk_index": r.chunk_index,
                "source_file": r.source_file,
                "model": r.model,
            }
            if r.page_number is not None:
                meta["page_number"] = r.page_number
            meta.update({k: str(v) for k, v in r.metadata.items()})
            metadatas.append(meta)
            documents.append("")

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        top_k = top_k or settings.top_k_results
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["metadatas", "distances", "documents"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = self._collection.query(**kwargs)

        hits = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i, chunk_id in enumerate(ids):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "similarity_score": 1 - distances[i],
                    "rank": i + 1,
                    "metadata": metadatas[i],
                }
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    def delete_collection(self) -> None:
        self._client.delete_collection(settings.chroma_collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
