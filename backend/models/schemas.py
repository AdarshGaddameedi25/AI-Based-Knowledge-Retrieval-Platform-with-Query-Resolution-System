from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


class DocumentModel(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str
    file_type: str
    file_size: int
    upload_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "pending"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkModel(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryModel(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    detected_intent: Optional[str] = None
    key_terms: List[str] = Field(default_factory=list)


class RetrievalResultModel(BaseModel):
    chunk_id: str
    document_id: str
    source_file: str
    text: str
    similarity_score: float
    rank: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResponseModel(BaseModel):
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ConversationMessageModel(BaseModel):
    role: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
