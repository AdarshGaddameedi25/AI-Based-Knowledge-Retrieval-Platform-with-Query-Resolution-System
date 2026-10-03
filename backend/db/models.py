import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime,
    ForeignKey, JSON, BigInteger, Boolean
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from backend.db.base import Base

EMBEDDING_DIM = 384


def _uuid():
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"

    document_id = Column(String, primary_key=True, default=_uuid)
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(BigInteger, nullable=False)
    file_hash = Column(String, nullable=True, index=True)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String, default="pending", nullable=False)
    domain = Column(String, nullable=True)
    doc_metadata = Column(JSON, default=dict)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=False)
    chunk_metadata = Column(JSON, default=dict)

    document = relationship("Document", back_populates="chunks")
    embedding = relationship("Embedding", back_populates="chunk", uselist=False, cascade="all, delete-orphan")


class Embedding(Base):
    __tablename__ = "embeddings"

    embedding_id = Column(String, primary_key=True, default=_uuid)
    chunk_id = Column(String, ForeignKey("document_chunks.chunk_id", ondelete="CASCADE"), nullable=False, unique=True)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    embedding_model = Column(String, nullable=False)
    embedding_dimension = Column(Integer, nullable=False, default=EMBEDDING_DIM)

    chunk = relationship("DocumentChunk", back_populates="embedding")


class Query(Base):
    __tablename__ = "queries"

    query_id = Column(String, primary_key=True, default=_uuid)
    query_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    session_id = Column(String, nullable=True)

    retrieval_results = relationship("RetrievalResult", back_populates="query", cascade="all, delete-orphan")
    response = relationship("Response", back_populates="query", uselist=False, cascade="all, delete-orphan")


class RetrievalResult(Base):
    __tablename__ = "retrieval_results"

    result_id = Column(String, primary_key=True, default=_uuid)
    query_id = Column(String, ForeignKey("queries.query_id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(String, ForeignKey("document_chunks.chunk_id", ondelete="SET NULL"), nullable=True)
    document_id = Column(String, ForeignKey("documents.document_id", ondelete="SET NULL"), nullable=True)
    similarity_score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)

    query = relationship("Query", back_populates="retrieval_results")


class Response(Base):
    __tablename__ = "responses"

    response_id = Column(String, primary_key=True, default=_uuid)
    query_id = Column(String, ForeignKey("queries.query_id", ondelete="CASCADE"), nullable=False, unique=True)
    answer = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    query = relationship("Query", back_populates="response")


class QueryAnalytics(Base):
    """
    M4.1 — Query analytics record.
    Stores full lifecycle metadata for every query processed by the orchestrator.
    Separate from the core knowledge-base tables.
    """
    __tablename__ = "query_analytics"

    analytics_id = Column(String, primary_key=True, default=_uuid)
    query_id = Column(String, nullable=True, index=True)          # FK-like link; nullable if not persisted
    session_id = Column(String, nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Query text
    original_query = Column(Text, nullable=False)
    normalized_query = Column(Text, nullable=True)

    # Classification
    query_type = Column(String, nullable=True)            # factual | procedural | comparative | ambiguous | direct
    detected_intent = Column(String, nullable=True)
    detected_domain = Column(String, nullable=True, index=True)

    # Routing
    routing_path = Column(String, nullable=True)          # retrieval | clarification | direct

    # Clarification
    clarification_required = Column(Boolean, default=False, nullable=False)
    clarification_count = Column(Integer, default=0, nullable=False)

    # Retrieval
    retrieval_count = Column(Integer, default=0, nullable=False)
    retrieved_chunks = Column(Integer, default=0, nullable=False)
    best_similarity_score = Column(Float, nullable=True)
    avg_similarity_score = Column(Float, nullable=True)
    source_count = Column(Integer, default=0, nullable=False)

    # Response
    response_status = Column(String, nullable=True)        # answered | knowledge_gap | error | clarification
    response_generated = Column(Boolean, default=False, nullable=False)
    knowledge_gap = Column(Boolean, default=False, nullable=False, index=True)
    knowledge_gap_reason = Column(Text, nullable=True)

    # Performance
    response_latency_ms = Column(Float, nullable=True)

    # Key terms (stored as JSON list)
    key_terms = Column(JSON, default=list)


class KnowledgeGap(Base):
    """
    M4.1 — Knowledge gap record.
    Created when a query cannot be answered due to insufficient evidence.
    Supports deduplication/grouping of repeated gaps.
    """
    __tablename__ = "knowledge_gaps"

    gap_id = Column(String, primary_key=True, default=_uuid)
    normalized_query = Column(Text, nullable=False, index=True)
    original_query = Column(Text, nullable=False)
    domain = Column(String, nullable=True, index=True)
    reason = Column(Text, nullable=True)                   # e.g. "no_chunks_retrieved" | "below_threshold" | "repeated_unanswered"
    first_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    frequency = Column(Integer, default=1, nullable=False)
    best_similarity_score = Column(Float, nullable=True)
    retrieval_count = Column(Integer, default=0, nullable=False)
    session_id = Column(String, nullable=True)
    status = Column(String, default="detected", nullable=False)  # detected | reviewed | resolved
