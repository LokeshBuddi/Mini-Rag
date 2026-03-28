"""
llm.py
------
LLMClient — thin wrapper around the OpenRouter chat completions API.
Tries a prioritised list of free models with automatic fallback.
"""

import os
import requests
from typing import List

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-3-4b-it:free",
    "openrouter/free",
]

SYSTEM = (
    "You are a knowledgeable assistant for Indecimal, a home construction marketplace. "
    "Answer ONLY using the context provided. "
    "If the answer is absent from the context, say: "
    "'This information is not available in the provided documents.'"
)

TEMPLATE = """\
Context:
{context}

Question: {question}

Answer strictly from the context above:"""


class LLMClient:
    def __init__(self, api_key: str | None = None, max_tokens: int = 512, temperature: float = 0.2):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def answer(self, question: str, chunks: List[dict]) -> str:
        if not self.api_key:
            return (
                "⚠️ No API key found.\n"
                "Set OPENROUTER_API_KEY in your .env file or enter it in the sidebar.\n"
                "Get a free key at https://openrouter.ai"
            )

        context = "\n\n".join(
            f"[{c['rank']}] (source: {c['source']})\n{c['text']}"
            for c in chunks
        )
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": TEMPLATE.format(
                context=context, question=question
            )},
        ]

        for model in FREE_MODELS:
            try:
                r = requests.post(
                    ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://rag-indecimal.app",
                        "X-Title": "Indecimal RAG v2",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                    timeout=60,
                )
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except requests.exceptions.Timeout:
                return "⚠️ Request timed out. Please try again."
            except requests.exceptions.HTTPError:
                continue
            except Exception as e:
                return f"⚠️ Unexpected error: {e}"

        return "⚠️ All models unavailable. Check https://openrouter.ai/models for active free models."
