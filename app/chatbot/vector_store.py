"""Vector store module for Umang P. Kaklotar's Portfolio Chatbot.

Manages Pinecone serverless vector index using the PINECONE API key from
backend/.env. Handles index creation, namespace reset, document upserting
without duplicates, and cosine similarity search retrieval.
"""

import os
import re
import time
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


def get_pinecone_api_key() -> str:
    """Retrieve the Pinecone API key from environment variables."""
    return (os.getenv("PINECONE") or os.getenv("PINECONE_API_KEY") or "").strip()


def _normalize_index_name(raw_name: str) -> str:
    """Normalize index name to match Pinecone rules: lowercase [a-z0-9-], max 45 chars."""
    normalized = re.sub(r"[^a-z0-9-]+", "-", (raw_name or "").lower()).strip("-")
    if not normalized:
        return "portfolio-knowledge"
    return normalized[:45].rstrip("-") or "portfolio-knowledge"


def _extract_index_dimension(model: Any) -> Optional[int]:
    """Safely extract vector dimension from a Pinecone IndexModel."""
    schema = getattr(model, "schema", None)
    fields = getattr(schema, "fields", None) if schema is not None else None
    if isinstance(fields, dict):
        for field in fields.values():
            dim = getattr(field, "dimension", None)
            if dim is not None:
                return int(dim)
            embed = getattr(field, "embed", None)
            if embed is not None and getattr(embed, "dimension", None) is not None:
                return int(embed.dimension)
    try:
        dim = getattr(model, "dimension", None)
        return int(dim) if dim is not None else None
    except AttributeError:
        return None


def _sanitize_metadata(meta: Optional[Dict[str, Any]], text: str) -> Dict[str, Any]:
    """Sanitize metadata for Pinecone (primitive types + list[str], includes chunk text)."""
    clean: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        key = str(k)
        if isinstance(v, (str, int, float, bool)):
            clean[key] = v
        elif isinstance(v, list):
            clean[key] = [str(item) for item in v if item is not None]
        else:
            clean[key] = str(v)
    clean["text"] = text
    return clean


DEFAULT_INDEX_NAME = _normalize_index_name(os.getenv("PINECONE_INDEX_NAME", "portfolio-knowledge"))
DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "portfolio").strip()
DEFAULT_CLOUD = os.getenv("PINECONE_CLOUD", "aws").strip()
DEFAULT_REGION = os.getenv("PINECONE_REGION", "us-east-1").strip()


class PortfolioVectorStore:
    """Wrapper around Pinecone serverless index for portfolio knowledge."""

    def __init__(
        self,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        api_key: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.api_key = (api_key if api_key is not None else get_pinecone_api_key()).strip()
        self.index_name = _normalize_index_name(
            index_name or os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
        )
        self.namespace = (
            namespace if namespace is not None else os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)
        ).strip()
        self.cloud = (cloud or os.getenv("PINECONE_CLOUD", DEFAULT_CLOUD)).strip()
        self.region = (region or os.getenv("PINECONE_REGION", DEFAULT_REGION)).strip()
        self.embedder = PortfolioEmbedder()
        self._client = None
        self._index = None
        self._index_dimension: Optional[int] = None

    @property
    def client(self):
        """Lazy-loaded Pinecone control-plane client."""
        if self._client is None:
            api_key = self.api_key or get_pinecone_api_key()
            if not api_key:
                raise ValueError(
                    "Pinecone API key is not configured. Set PINECONE in backend/.env."
                )
            from pinecone import Pinecone

            self._client = Pinecone(api_key=api_key)
        return self._client

    def _index_exists(self) -> bool:
        """Check if the configured Pinecone index exists."""
        return bool(self.client.indexes.exists(self.index_name))

    def _describe_index(self):
        """Fetch IndexModel metadata for the configured index."""
        model = self.client.indexes.describe(self.index_name)
        dim = _extract_index_dimension(model)
        if dim is not None:
            self._index_dimension = dim
        return model

    @property
    def index(self):
        """Lazy-loaded Pinecone data-plane Index client."""
        if self._index is None:
            if not self._index_exists():
                raise RuntimeError(
                    f"Pinecone index '{self.index_name}' does not exist yet. "
                    "Run 'python -m app.chatbot.ingest' to create and populate it."
                )
            model = self._describe_index()
            if not model.host:
                raise RuntimeError(
                    f"Pinecone index '{self.index_name}' is still initializing and has no host yet."
                )
            self._index = self.client.index(host=model.host)
        return self._index

    def ensure_index(self, dimension: int) -> None:
        """Ensure the Pinecone serverless index exists with the required dimension."""
        if self._index_exists():
            model = self._describe_index()
            existing_dim = _extract_index_dimension(model)
            if existing_dim is not None and existing_dim != dimension:
                raise ValueError(
                    f"Pinecone index '{self.index_name}' has dimension {existing_dim}, "
                    f"which does not match embedding dimension {dimension}."
                )
            if not model.status.ready or not model.host:
                deadline = time.time() + 120
                while time.time() < deadline:
                    time.sleep(2)
                    model = self._describe_index()
                    if model.status.ready and model.host:
                        break
            if not model.host:
                raise RuntimeError(f"Pinecone index '{self.index_name}' did not become ready in time.")
            self._index = self.client.index(host=model.host)
            self._index_dimension = existing_dim or dimension
            return

        from pinecone import ServerlessSpec

        logger.info(
            "Creating Pinecone serverless index '%s' (dimension=%d, metric=cosine, cloud=%s, region=%s)...",
            self.index_name,
            dimension,
            self.cloud,
            self.region,
        )
        model = self.client.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            timeout=120,
        )
        if not model.host:
            model = self._describe_index()
        self._index_dimension = dimension
        self._index = self.client.index(host=model.host)
        logger.info("Pinecone index '%s' is ready.", self.index_name)

    def reset_namespace(self) -> None:
        """Clear all vectors in the configured namespace to avoid stale chunks."""
        if not self._index_exists():
            logger.info("Pinecone index '%s' does not exist yet; nothing to reset.", self.index_name)
            return

        current_count = self.count()
        if current_count == 0:
            logger.info(
                "Pinecone namespace '%s' in index '%s' is already empty.",
                self.namespace or "__default__",
                self.index_name,
            )
            return

        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            logger.info(
                "Cleared existing vectors from Pinecone index '%s' (namespace='%s').",
                self.index_name,
                self.namespace or "__default__",
            )
        except Exception as e:
            # Handle 404 NotFound cleanly if namespace was already absent
            if "404" in str(e) or "not found" in str(e).lower():
                logger.debug("Namespace already absent during delete: %s", e)
                return
            raise

        # Wait briefly for eventual consistency on serverless delete
        deadline = time.time() + 10
        while time.time() < deadline:
            if self.count() == 0:
                break
            time.sleep(0.5)

    def upsert_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
        wait_until_indexed: bool = True,
    ) -> int:
        """Embed and upsert document chunks into Pinecone.

        Uses deterministic IDs so re-running ingestion does not create duplicates.
        """
        if not ids or not documents:
            return 0
        if len(ids) != len(documents):
            raise ValueError("ids and documents must have the same length.")

        meta_list = metadatas or [{} for _ in ids]
        if len(meta_list) != len(ids):
            raise ValueError("metadatas must match ids length.")

        embeddings = self.embedder.embed_documents(documents)
        if not embeddings or not embeddings[0]:
            raise RuntimeError("Failed to generate embeddings for documents.")

        dimension = len(embeddings[0])
        self.ensure_index(dimension=dimension)

        pre_count = self.count() if wait_until_indexed else 0

        vectors = [
            {
                "id": doc_id,
                "values": vec,
                "metadata": _sanitize_metadata(meta, doc),
            }
            for doc_id, vec, meta, doc in zip(ids, embeddings, meta_list, documents)
        ]

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)

        if wait_until_indexed:
            target_count = max(len(ids), pre_count)
            deadline = time.time() + 15
            while time.time() < deadline:
                if self.count() >= target_count:
                    break
                time.sleep(0.5)

        logger.info(
            "Successfully upserted %d chunks to Pinecone index '%s' (namespace='%s').",
            len(ids),
            self.index_name,
            self.namespace or "__default__",
        )
        return len(ids)

    def query(
        self,
        query_text: str,
        n_results: int = 4,
    ) -> List[Dict[str, Any]]:
        """Query Pinecone index namespace for top-k similar chunks."""
        api_key = self.api_key or get_pinecone_api_key()
        if not api_key:
            logger.warning("Pinecone API key is not configured.")
            return []

        if self._index is None and not self._index_exists():
            logger.warning("Pinecone index '%s' does not exist. Please run ingestion first.", self.index_name)
            return []

        total = self.count()
        if total == 0:
            logger.warning("Pinecone vector store is empty. Please run ingestion first.")
            return []

        query_vector = self.embedder.embed_query(
            query_text,
            fallback_dim=self._index_dimension,
        )
        if not query_vector:
            return []

        actual_n = min(n_results, total)
        results = self.index.query(
            namespace=self.namespace,
            vector=query_vector,
            top_k=actual_n,
            include_metadata=True,
            include_values=False,
        )

        matched_items = []
        matches = getattr(results, "matches", None) or []
        for match in matches:
            meta = dict(getattr(match, "metadata", None) or {})
            content = meta.pop("text", None)
            if content is None:
                content = meta.pop("content", "")
            score = getattr(match, "score", None)
            score_val = float(score) if score is not None else 0.0

            matched_items.append(
                {
                    "id": getattr(match, "id", None),
                    "content": content,
                    "metadata": meta,
                    "distance": round(1.0 - score_val, 4),
                    "similarity": round(score_val, 4),
                }
            )

        return matched_items

    def count(self) -> int:
        """Return total vector count in the configured Pinecone namespace."""
        api_key = self.api_key or get_pinecone_api_key()
        if not api_key:
            return 0
        try:
            if self._index is None and not self._index_exists():
                return 0
            stats = self.index.describe_index_stats()
            namespaces = getattr(stats, "namespaces", None) or {}
            ns_key = self.namespace if self.namespace else "__default__"
            ns_summary = namespaces.get(ns_key)
            if ns_summary is None:
                return 0
            return int(getattr(ns_summary, "vector_count", 0))
        except Exception as e:
            logger.debug("Could not fetch Pinecone vector count: %s", e)
            return 0

    def get_status(self) -> Dict[str, Any]:
        """Return Pinecone vector store health and statistics."""
        configured = bool(self.api_key or get_pinecone_api_key())
        doc_count = self.count() if configured else 0
        if not configured:
            store_status = "not_configured"
        elif doc_count > 0:
            store_status = "ready"
        else:
            store_status = "empty"

        return {
            "backend": "pinecone",
            "pinecone_configured": configured,
            "index_name": self.index_name,
            "namespace": self.namespace or "__default__",
            "document_count": doc_count,
            "status": store_status,
        }
