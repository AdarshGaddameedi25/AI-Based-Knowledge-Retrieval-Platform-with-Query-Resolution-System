import pypdf
from typing import List, Tuple


class PDFLoader:
    def extract(self, file_path: str) -> str:
        pages = self.extract_pages(file_path)
        return "\n".join(text for _, text in pages)

    def extract_pages(self, file_path: str) -> List[Tuple[int, str]]:
        pages = []
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages.append((page_num, page_text))
        return pages
