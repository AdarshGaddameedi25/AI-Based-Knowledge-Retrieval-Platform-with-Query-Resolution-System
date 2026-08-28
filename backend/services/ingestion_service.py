from ingestion.document_loader import DocumentLoader
from ingestion.text_cleaner import TextCleaner
from ingestion.chunking import TextChunker
from ingestion.embeddings import EmbeddingService
from retrieval.vector_store import VectorStore


class IngestionService:
    def __init__(self):
        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()
        self.chunker = TextChunker()
        self.store = VectorStore()

    def ingest_file(self, file_path: str) -> dict:
        embedder = EmbeddingService()
        doc = self.loader.load(file_path)
        clean_text = self.cleaner.clean(doc.text)
        chunks = self.chunker.chunk(clean_text, doc.document_id, metadata={"source_file": doc.file_name})
        embeddings = embedder.embed_chunks(chunks, source_file=doc.file_name)
        for emb, chunk in zip(embeddings, chunks):
            emb.metadata["text"] = chunk.text
        self.store.add_chunks(embeddings)
        return {
            "document_id": doc.document_id,
            "file_name": doc.file_name,
            "chunks_created": len(chunks),
        }

    def ingest_directory(self, directory: str) -> list[dict]:
        docs = self.loader.load_directory(directory)
        results = []
        embedder = EmbeddingService()
        for doc in docs:
            clean_text = self.cleaner.clean(doc.text)
            chunks = self.chunker.chunk(clean_text, doc.document_id, metadata={"source_file": doc.file_name})
            embeddings = embedder.embed_chunks(chunks, source_file=doc.file_name)
            for emb, chunk in zip(embeddings, chunks):
                emb.metadata["text"] = chunk.text
            self.store.add_chunks(embeddings)
            results.append({"document_id": doc.document_id, "file_name": doc.file_name, "chunks": len(chunks)})
        return results
