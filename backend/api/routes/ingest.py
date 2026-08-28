import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ingestion.document_loader import DocumentLoader
from ingestion.text_cleaner import TextCleaner
from ingestion.chunking import TextChunker
from ingestion.embeddings import EmbeddingService
from retrieval.vector_store import VectorStore

router = APIRouter()

loader = DocumentLoader()
cleaner = TextCleaner()
chunker = TextChunker()
store = VectorStore()


@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".docx", ".txt", ".csv"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    try:
        embedder = EmbeddingService()
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = loader.load(tmp_path)
        doc.file_name = file.filename
        clean_text = cleaner.clean(doc.text)
        chunks = chunker.chunk(clean_text, doc.document_id, metadata={"source_file": file.filename})
        embeddings = embedder.embed_chunks(chunks, source_file=file.filename)

        for emb, chunk in zip(embeddings, chunks):
            emb.metadata["text"] = chunk.text

        store.add_chunks(embeddings)
    finally:
        os.unlink(tmp_path)

    return JSONResponse({
        "document_id": doc.document_id,
        "file_name": file.filename,
        "chunks_created": len(chunks),
        "status": "indexed",
    })


@router.get("/documents/count")
def document_count():
    return {"total_chunks_indexed": store.count()}
