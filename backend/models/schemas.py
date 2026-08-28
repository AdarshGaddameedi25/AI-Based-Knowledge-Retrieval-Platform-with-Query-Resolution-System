from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid


class DocumentModel(BaseModel):
    document_id: str = str(uuid.uuid4())
    file_name: str
    file_type: str
    file_size: int
    upload_date: str = datetime.utcnow().isoformat()
    status: str = "pending"
    metadata: dict = {}


class ChunkModel(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    page_number: Optional[int] = None
    metadata: dict = {}


class QueryModel(BaseModel):
    query_id: str = str(uuid.uuid4())
    query_text: str
    timestamp: str = datetime.utcnow().isoformat()
    detected_intent: Optional[str] = None
    key_terms: list[str] = []


class RetrievalResultModel(BaseModel):
    chunk_id: str
    document_id: str
    source_file: str
    text: str
    similarity_score: float
    rank: int
    metadata: dict = {}


class ResponseModel(BaseModel):
    response_id: str = str(uuid.uuid4())
    query: str
    answer: str
    citations: list[dict] = []
    timestamp: str = datetime.utcnow().isoformat()


class ConversationMessageModel(BaseModel):
    role: str
    content: str
    timestamp: str = datetime.utcnow().isoformat()
