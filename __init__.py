"""
loader.py
---------
DocumentLoader class — loads .md, .txt, and .pdf files from a directory.
Strips Markdown syntax before returning clean plain text.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    name: str
    text: str
    char_count: int


class DocumentLoader:
    SUPPORTED = {".md", ".txt", ".pdf"}

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load(self) -> List[Document]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        files = [f for f in sorted(self.data_dir.iterdir())
                 if f.suffix.lower() in self.SUPPORTED]

        if not files:
            raise ValueError(f"No supported files in '{self.data_dir}'")

        docs = []
        for f in files:
            text = self._read(f)
            if text.strip():
                docs.append(Document(name=f.name, text=text, char_count=len(text)))
                print(f"  ✔  {f.name}  ({len(text):,} chars)")

        print(f"\n  {len(docs)} document(s) loaded\n")
        return docs

    def _read(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in (".md", ".txt"):
            return self._clean_md(path.read_text(encoding="utf-8"))
        if suffix == ".pdf":
            return self._read_pdf(path)
        return ""

    @staticmethod
    def _clean_md(text: str) -> str:
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]*`", "", text)
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_\n]+)_{1,3}", r"\1", text)
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
