import logging
from datetime import datetime
from sqlalchemy.orm import Session

from ingestion.document_loader import DocumentLoader
from ingestion.text_cleaner import TextCleaner
from ingestion.chunking import TextChunker
from ingestion.embeddings import EmbeddingService
from backend.db.models import Document, DocumentChunk, Embedding

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self):
        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()
        self.chunker = TextChunker()
        self.embedder = EmbeddingService()

    def ingest_file(self, file_path: str, db: Session, original_filename: str = None, domain: str = None) -> dict:
        doc_record = self.loader.load(file_path)
        if original_filename:
            doc_record.file_name = original_filename

        existing = db.query(Document).filter_by(file_hash=doc_record.file_hash).first()
        if existing:
            logger.info(f"Document already indexed: {existing.file_name} (id={existing.document_id})")
            return {
                "document_id": existing.document_id,
                "file_name": existing.file_name,
                "chunks_created": db.query(DocumentChunk).filter_by(document_id=existing.document_id).count(),
                "status": "already_indexed",
            }

        db_doc = Document(
            document_id=doc_record.document_id,
            file_name=doc_record.file_name,
            file_type=doc_record.file_type,
            file_size=doc_record.file_size,
            file_hash=doc_record.file_hash,
            upload_date=datetime.utcnow(),
            status="processing",
            domain=domain,
            doc_metadata=doc_record.metadata,
        )
        db.add(db_doc)
        db.flush()

        try:
            if doc_record.pages:
                meta = {"source_file": doc_record.file_name, "domain": domain or ""}
                chunks = self.chunker.chunk_pages(doc_record.pages, doc_record.document_id, metadata=meta)
            else:
                clean_text = self.cleaner.clean(doc_record.text)
                meta = {"source_file": doc_record.file_name, "domain": domain or ""}
                chunks = self.chunker.chunk(clean_text, doc_record.document_id, metadata=meta)

            logger.info(f"Created {len(chunks)} chunks for {doc_record.file_name}")

            embedding_records = self.embedder.embed_chunks(chunks, source_file=doc_record.file_name)

            logger.info(f"Generated {len(embedding_records)} embeddings for {doc_record.file_name}")

            for chunk, emb_rec in zip(chunks, embedding_records):
                db_chunk = DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=doc_record.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    token_count=chunk.token_count,
                    chunk_metadata=chunk.metadata,
                )
                db.add(db_chunk)

                db_emb = Embedding(
                    chunk_id=chunk.chunk_id,
                    embedding=emb_rec.embedding,
                    embedding_model=emb_rec.model,
                    embedding_dimension=len(emb_rec.embedding),
                )
                db.add(db_emb)

            db_doc.status = "indexed"
            db.commit()

            logger.info(f"Successfully indexed: {doc_record.file_name}")
            return {
                "document_id": doc_record.document_id,
                "file_name": doc_record.file_name,
                "chunks_created": len(chunks),
                "status": "indexed",
            }

        except Exception as e:
            db.rollback()
            db_doc.status = "failed"
            db.add(db_doc)
            db.commit()
            logger.error(f"Ingestion failed for {doc_record.file_name}: {e}")
            raise

    def list_documents(self, db: Session) -> list:
        docs = db.query(Document).order_by(Document.upload_date.desc()).all()
        results = []
        for d in docs:
            chunk_count = db.query(DocumentChunk).filter_by(document_id=d.document_id).count()
            results.append({
                "document_id": d.document_id,
                "file_name": d.file_name,
                "file_type": d.file_type,
                "file_size": d.file_size,
                "upload_date": d.upload_date.isoformat(),
                "status": d.status,
                "domain": d.domain,
                "chunks": chunk_count,
            })
        return results
