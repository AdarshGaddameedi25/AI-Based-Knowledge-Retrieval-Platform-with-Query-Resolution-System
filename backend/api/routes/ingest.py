import os
import logging
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.db.base import get_db
from backend.services.ingestion_service import IngestionService

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}
MAX_FILE_SIZE = 50 * 1024 * 1024

_service = IngestionService()


import shutil

@router.post("/documents/upload")
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    original_name = os.path.basename(file.filename or "upload")
    ext = os.path.splitext(original_name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    file_size = os.path.getsize(tmp_path)
    if file_size == 0:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail="File is empty.")

    if file_size > MAX_FILE_SIZE:
        os.unlink(tmp_path)
        raise HTTPException(status_code=413, detail=f"File exceeds maximum size of {MAX_FILE_SIZE // (1024*1024)} MB.")

    try:
        result = _service.ingest_file(tmp_path, db=db, original_filename=original_name)
    except Exception as e:
        logger.error(f"Ingestion error for {original_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return JSONResponse(result)


@router.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    try:
        return _service.list_documents(db)
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
