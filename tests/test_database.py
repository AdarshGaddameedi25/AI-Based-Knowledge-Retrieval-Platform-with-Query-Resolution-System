import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config.settings import settings


def get_test_engine():
    return create_engine(settings.database_url, pool_pre_ping=True)


def test_postgres_connection():
    engine = get_test_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
    assert result == 1


def test_pgvector_extension():
    engine = get_test_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT installed_version FROM pg_available_extensions WHERE name = 'vector'")
        ).fetchone()
    assert result is not None, "pgvector extension not available on this PostgreSQL instance"


def test_documents_table_exists():
    engine = get_test_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'documents'")
        ).scalar()
    assert result == 1, "documents table does not exist — run: alembic upgrade head"


def test_embeddings_table_exists():
    engine = get_test_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'embeddings'")
        ).scalar()
    assert result == 1, "embeddings table does not exist — run: alembic upgrade head"


def test_vector_column_type():
    engine = get_test_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT udt_name FROM information_schema.columns
                WHERE table_name = 'embeddings' AND column_name = 'embedding'
            """)
        ).fetchone()
    assert result is not None
    assert "vector" in result[0].lower()


def test_insert_and_search_vector():
    from pgvector.sqlalchemy import Vector
    engine = get_test_engine()
    Session = sessionmaker(bind=engine)

    import uuid
    from backend.db.models import Document, DocumentChunk, Embedding
    from datetime import datetime

    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    embedding_id = str(uuid.uuid4())
    test_vector = [0.1] * 384

    with Session() as db:
        doc = Document(
            document_id=doc_id,
            file_name="test_db_vector.txt",
            file_type="TXT",
            file_size=100,
            file_hash="test_hash_" + doc_id,
            upload_date=datetime.utcnow(),
            status="indexed",
        )
        db.add(doc)

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            chunk_index=0,
            text="pgvector is a PostgreSQL extension for vector similarity search.",
            token_count=12,
        )
        db.add(chunk)

        emb = Embedding(
            embedding_id=embedding_id,
            chunk_id=chunk_id,
            embedding=test_vector,
            embedding_model="test-model",
            embedding_dimension=384,
        )
        db.add(emb)
        db.commit()

        embedding_str = "[" + ",".join(str(v) for v in test_vector) + "]"
        results = db.execute(
            text("""
                SELECT e.chunk_id, 1 - (e.embedding <=> CAST(:emb AS vector)) AS score
                FROM embeddings e
                WHERE e.chunk_id = :chunk_id
            """),
            {"emb": embedding_str, "chunk_id": chunk_id},
        ).fetchall()

        assert len(results) == 1
        assert abs(results[0].score - 1.0) < 0.001

        db.query(Embedding).filter_by(embedding_id=embedding_id).delete()
        db.query(DocumentChunk).filter_by(chunk_id=chunk_id).delete()
        db.query(Document).filter_by(document_id=doc_id).delete()
        db.commit()
