"""Initial schema with pgvector

Revision ID: 0001_initial
Revises: 
Create Date: 2026-09-05

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("file_type", sa.String(), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("file_hash", sa.String(), nullable=True),
        sa.Column("upload_date", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("domain", sa.String(), nullable=True),
        sa.Column("doc_metadata", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("document_id"),
    )
    op.create_index("ix_documents_file_hash", "documents", ["file_hash"])
    op.create_index("ix_documents_status", "documents", ["status"])

    op.create_table(
        "document_chunks",
        sa.Column("chunk_id", sa.String(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("chunk_metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["documents.document_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index("ix_chunks_document_id", "document_chunks", ["document_id"])

    op.create_table(
        "embeddings",
        sa.Column("embedding_id", sa.String(), nullable=False),
        sa.Column("chunk_id", sa.String(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False, server_default=str(EMBEDDING_DIM)),
        sa.ForeignKeyConstraint(["chunk_id"], ["document_chunks.chunk_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("embedding_id"),
        sa.UniqueConstraint("chunk_id"),
    )

    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_embeddings_vector
        ON embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)

    op.create_table(
        "queries",
        sa.Column("query_id", sa.String(), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("query_id"),
    )

    op.create_table(
        "retrieval_results",
        sa.Column("result_id", sa.String(), nullable=False),
        sa.Column("query_id", sa.String(), nullable=False),
        sa.Column("chunk_id", sa.String(), nullable=True),
        sa.Column("document_id", sa.String(), nullable=True),
        sa.Column("similarity_score", sa.Float(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["query_id"], ["queries.query_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["document_chunks.chunk_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.document_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("result_id"),
    )

    op.create_table(
        "responses",
        sa.Column("response_id", sa.String(), nullable=False),
        sa.Column("query_id", sa.String(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["query_id"], ["queries.query_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("response_id"),
        sa.UniqueConstraint("query_id"),
    )


def downgrade() -> None:
    op.drop_table("responses")
    op.drop_table("retrieval_results")
    op.drop_table("queries")
    op.drop_index("ix_embeddings_vector", table_name="embeddings")
    op.drop_table("embeddings")
    op.drop_index("ix_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_index("ix_documents_file_hash", table_name="documents")
    op.drop_index("ix_documents_status", table_name="documents")
    op.drop_table("documents")
