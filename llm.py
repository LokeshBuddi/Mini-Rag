"""
chunker.py
----------
SentenceChunker — splits text into overlapping windows,
preferring sentence boundaries for cleaner semantic units.
"""

import re
from dataclasses import dataclass
from typing import List
from core.loader import Document


@dataclass
class Chunk:
    id: int
    source: str
    text: str

    def preview(self, n: int = 80) -> str:
        return self.text[:n] + ("…" if len(self.text) > n else "")


class SentenceChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, docs: List[Document]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        cid = 0
        for doc in docs:
            for text in self._split(doc.text):
                all_chunks.append(Chunk(id=cid, source=doc.name, text=text))
                cid += 1
        print(f"  {len(all_chunks)} chunks created\n")
        return all_chunks

    def _split(self, text: str) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                end = self._boundary(text, start, end)
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            start += self.chunk_size - self.overlap
        return chunks

    @staticmethod
    def _boundary(text: str, start: int, end: int) -> int:
        search_from = end - max(1, (end - start) // 5)
        for i in range(end, search_from, -1):
            if i < len(text) and text[i - 1] in ".?!" and (i >= len(text) or text[i] == " "):
                return i
        for i in range(end, search_from, -1):
            if i < len(text) and text[i] == " ":
                return i
        return end
