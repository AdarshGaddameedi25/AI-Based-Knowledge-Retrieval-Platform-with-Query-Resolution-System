import os
import uuid
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from ingestion.pdf_loader import PDFLoader
from ingestion.docx_loader import DOCXLoader
from ingestion.txt_loader import TXTLoader
from ingestion.csv_loader import CSVLoader


@dataclass
class DocumentRecord:
    document_id: str
    file_name: str
    file_type: str
    file_size: int
    upload_date: str
    status: str
    text: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    SUPPORTED_TYPES = {".pdf", ".docx", ".txt", ".csv"}

    def __init__(self):
        self._loaders = {
            ".pdf": PDFLoader(),
            ".docx": DOCXLoader(),
            ".txt": TXTLoader(),
            ".csv": CSVLoader(),
        }

    def load(self, file_path: str) -> DocumentRecord:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {self.SUPPORTED_TYPES}")

        loader = self._loaders[ext]
        text = loader.extract(file_path)

        return DocumentRecord(
            document_id=str(uuid.uuid4()),
            file_name=os.path.basename(file_path),
            file_type=ext.lstrip(".").upper(),
            file_size=os.path.getsize(file_path),
            upload_date=datetime.utcnow().isoformat(),
            status="loaded",
            text=text,
            metadata={"source_path": file_path},
        )

    def load_directory(self, directory: str) -> list[DocumentRecord]:
        records = []
        for root, _, files in os.walk(directory):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_TYPES:
                    try:
                        record = self.load(os.path.join(root, fname))
                        records.append(record)
                    except Exception as e:
                        print(f"Failed to load {fname}: {e}")
        return records
