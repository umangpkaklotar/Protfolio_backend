"""Embeddings module for Umang P. Kaklotar's Portfolio Chatbot.

Handles text embedding generation using Google Gemini Embedding API with
graceful fallbacks for local testing and configuration.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Any
import requests
from dotenv import load_dotenv

# Ensure backend/.env is loaded with override=True
env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    load_dotenv(override=True)

try:
    from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
    BaseEF = EmbeddingFunction[Documents]
except ImportError:
    BaseEF = object
    Documents = List[str]
    Embeddings = List[List[float]]

logger = logging.getLogger(__name__)

# Default Gemini embedding model (gemini-embedding-2 is the active model)
DEFAULT_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")


def get_gemini_api_key() -> str:
    """Retrieve the Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY", "").strip()


def _fallback_deterministic_embedding(text: str, dim: int = 768) -> List[float]:
    """Generate a pseudo-deterministic normalized vector for offline testing

    when GEMINI_API_KEY is not configured yet. This allows ChromaDB ingestion
    and testing to complete without throwing hard errors.
    """
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
    vector = []
    current = seed
    for _ in range(dim):
        current = (current * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        val = (current / 0xFFFFFFFFFFFFFFFF) * 2.0 - 1.0
        vector.append(val)
    norm = sum(x * x for x in vector) ** 0.5
    if norm > 0:
        vector = [x / norm for x in vector]
    return vector


class PortfolioEmbedder(BaseEF):
    """Embedding generator supporting Google Gemini and ChromaDB."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = (api_key or get_gemini_api_key()).strip()
        self.model = model or os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.model_path = self.model

    def name(self) -> str:
        """Name of this embedding function for ChromaDB registration."""
        return "portfolio_gemini_embedder"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document strings with automatic model fallback."""
        if not texts:
            return []

        api_key = self.api_key or get_gemini_api_key()
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY is not set. Using deterministic fallback embeddings for %d documents.",
                len(texts),
            )
            return [_fallback_deterministic_embedding(t) for t in texts]

        # Candidate models to try in order
        configured = (self.model_path or DEFAULT_EMBEDDING_MODEL).replace("models/", "").strip()
        candidates = [configured] if configured else []
        for fb in ["gemini-embedding-2", "gemini-embedding-001", "text-embedding-004"]:
            if fb not in candidates:
                candidates.append(fb)

        for model_to_try in candidates:
            # 1. Try google.genai SDK
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                embeddings: List[List[float]] = []
                batch_size = 16
                for i in range(0, len(texts), batch_size):
                    sub_batch = texts[i : i + batch_size]
                    res = client.models.embed_content(
                        model=model_to_try,
                        contents=sub_batch,
                    )
                    if hasattr(res, "embeddings") and res.embeddings:
                        for emb in res.embeddings:
                            embeddings.append(list(emb.values))
                    elif hasattr(res, "embedding") and res.embedding:
                        embeddings.append(list(res.embedding.values))
                if len(embeddings) == len(texts):
                    return embeddings
            except Exception as e:
                logger.debug("google.genai embedding failed for '%s': %s", model_to_try, e)

            # 2. Try direct REST API fallback
            try:
                full_model = f"models/{model_to_try}" if not model_to_try.startswith("models/") else model_to_try
                url = f"https://generativelanguage.googleapis.com/v1beta/{full_model}:batchEmbedContents?key={api_key}"
                requests_payload = {
                    "requests": [
                        {
                            "model": full_model,
                            "content": {"parts": [{"text": t}]},
                        }
                        for t in texts
                    ]
                }
                resp = requests.post(url, json=requests_payload, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    embs = [item["values"] for item in data.get("embeddings", [])]
                    if len(embs) == len(texts):
                        return embs
            except Exception as e:
                logger.debug("Gemini REST embedding failed for '%s': %s", model_to_try, e)

        # Final offline fallback if network or API failed
        logger.warning("Falling back to deterministic offline embeddings.")
        return [_fallback_deterministic_embedding(t) for t in texts]

    def embed_query(self, input: Any = None, text: Any = None, *args, **kwargs) -> Any:
        """Embed query input (string or list of strings) for ChromaDB."""
        target = input if input is not None else text
        if target is None:
            return []
        if isinstance(target, str):
            target = [target]
        elif not isinstance(target, (list, tuple)):
            target = [str(target)]
        return self.embed_documents(list(target))

    def __call__(self, input: Any = None, *args, **kwargs) -> Any:
        """ChromaDB embedding function protocol call."""
        if input is None:
            input = []
        if isinstance(input, str):
            input = [input]
        elif not isinstance(input, (list, tuple)):
            input = [str(input)]
        return self.embed_documents(list(input))
