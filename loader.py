"""
engine.py
---------
RAGEngine — single entry point that wires loader → chunker
→ vector store → LLM into one cohesive object.
"""

from core.loader       import DocumentLoader
from core.chunker      import SentenceChunker
from core.vector_store import VectorStore
from core.llm          import LLMClient
from typing import List


class RAGEngine:
    def __init__(
        self,
        data_dir:   str = "data",
        store_dir:  str = "vector_store",
        top_k:      int = 3,
        api_key:    str | None = None,
        chunk_size: int = 500,
        overlap:    int = 100,
    ):
        self.top_k  = top_k
        self.store  = VectorStore(store_dir)
        self.llm    = LLMClient(api_key=api_key)

        # Build or load the index
        if self.store.exists():
            print("── Loading existing vector store ──")
            self.store.load()
        else:
            print("── Building vector store ──")
            loader  = DocumentLoader(data_dir)
            chunker = SentenceChunker(chunk_size=chunk_size, overlap=overlap)
            docs    = loader.load()
            chunks  = chunker.chunk(docs)
            self.store.build(chunks)

    def query(self, question: str) -> dict:
        if not question.strip():
            return {"question": question, "chunks": [], "answer": "Please enter a question."}

        chunks = self.store.search(question, top_k=self.top_k)
        answer = self.llm.answer(question, chunks)

        return {
            "question": question,
            "chunks":   chunks,
            "answer":   answer,
        }

    def rebuild(self, data_dir: str = "data", chunk_size: int = 500, overlap: int = 100) -> None:
        """Force a full re-ingest."""
        loader  = DocumentLoader(data_dir)
        chunker = SentenceChunker(chunk_size=chunk_size, overlap=overlap)
        docs    = loader.load()
        chunks  = chunker.chunk(docs)
        self.store.build(chunks)
