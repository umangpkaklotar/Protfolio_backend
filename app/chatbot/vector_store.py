"""Vector store module for Umang P. Kaklotar's Portfolio Chatbot.

Manages persistent ChromaDB vector store located at backend/chroma_db/.
Handles collection initialization, upserting documents without duplicates,
and similarity search retrieval.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Ensure backend/.env is loaded
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
env_file = BACKEND_DIR / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    load_dotenv(override=True)

from .embeddings import PortfolioEmbedder

logger = logging.getLogger(__name__)

# Guarantee persistent ChromaDB directory is always inside backend/chroma_db
raw_chroma_dir = os.getenv("CHROMA_DB_DIR", "chroma_db")
if os.path.isabs(raw_chroma_dir):
    DEFAULT_DB_DIR = Path(raw_chroma_dir)
else:
    DEFAULT_DB_DIR = BACKEND_DIR / raw_chroma_dir

CHROMA_DIR = str(DEFAULT_DB_DIR.resolve())
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "portfolio_knowledge")


class PortfolioVectorStore:
    """Wrapper around ChromaDB PersistentClient for portfolio knowledge."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = str(Path(persist_dir or CHROMA_DIR).resolve())
        self.collection_name = collection_name or COLLECTION_NAME
        self.embedder = PortfolioEmbedder()
        self._client = None
        self._collection = None

    @property
    def client(self):
        """Lazy-loaded persistent ChromaDB client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            # Ensure directory exists
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        return self._client

    @property
    def collection(self):
        """Lazy-loaded or created collection with custom embedding function."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedder,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def reset_collection(self):
        """Delete existing collection and recreate fresh to avoid duplicates."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info("Deleted existing collection '%s'.", self.collection_name)
        except Exception:
            pass
        self._collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def upsert_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """Upsert document chunks into ChromaDB.

        Uses deterministic IDs so re-running ingestion does not create duplicates.
        """
        if not ids or not documents:
            return 0

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas or [{} for _ in ids],
        )
        logger.info("Successfully upserted %d chunks to ChromaDB.", len(ids))
        return len(ids)

    def query(
        self,
        query_text: str,
        n_results: int = 4,
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB collection for top-k similar chunks."""
        count = self.collection.count()
        if count == 0:
            logger.warning("Vector store collection is empty. Please run ingestion first.")
            return []

        actual_n = min(n_results, count)
        results = self.collection.query(
            query_texts=[query_text],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )

        matched_items = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                matched_items.append(
                    {
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                        "similarity": round(1.0 - dist, 4) if dist is not None else 1.0,
                    }
                )

        return matched_items

    def count(self) -> int:
        """Return total document count in collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_status(self) -> Dict[str, Any]:
        """Return vector store health and statistics."""
        return {
            "persist_dir": self.persist_dir,
            "collection_name": self.collection_name,
            "document_count": self.count(),
            "status": "ready" if self.count() > 0 else "empty",
        }
