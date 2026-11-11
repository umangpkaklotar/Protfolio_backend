"""Ingestion pipeline for Umang P. Kaklotar's Portfolio Chatbot.

Reads backend/app/chatbot/knowledge.md, splits the content into semantic,
section-aware chunks, generates embeddings, and upserts them into Pinecone.

Ensures no duplicate documents when run multiple times.
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Ensure the backend directory is in sys.path when run directly
backend_root = Path(__file__).resolve().parent.parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.chatbot.vector_store import PortfolioVectorStore

env_file = backend_root / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KNOWLEDGE_FILE = Path(__file__).resolve().parent / "knowledge.md"


def load_knowledge_file(filepath: Path = KNOWLEDGE_FILE) -> str:
    """Read the markdown knowledge file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Knowledge file not found at: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def chunk_markdown(content: str, max_chunk_chars: int = 850, overlap_chars: int = 120) -> List[Dict[str, Any]]:
    """Chunk markdown document by major sections (# and ##), then sub-chunk

    any sections that exceed max_chunk_chars while maintaining overlap.
    Preserves section titles in metadata and prepends section context to chunks.
    """
    chunks: List[Dict[str, Any]] = []

    # Split by level 1 or 2 markdown headers
    pattern = r"(?=(?:^|\n)#{1,2}\s+[^\n]+)"
    raw_sections = re.split(pattern, content.strip())

    chunk_idx = 0
    for section in raw_sections:
        section = section.strip()
        if not section:
            continue

        header_match = re.match(r"^#{1,2}\s+([^\n]+)", section)
        if header_match:
            section_title = header_match.group(1).strip()
        else:
            section_title = "General"

        if len(section) <= max_chunk_chars:
            chunks.append(
                {
                    "id": f"chunk_{chunk_idx:04d}",
                    "content": section,
                    "metadata": {
                        "section": section_title,
                        "source": "knowledge.md",
                        "chunk_index": chunk_idx,
                        "char_count": len(section),
                    },
                }
            )
            chunk_idx += 1
        else:
            # Split section into smaller units: paragraphs or bullet items
            units = re.split(r"\n\n+|\n(?=- |\d+\. )", section)
            current_text = f"### Section: {section_title}\n"

            for u in units:
                u = u.strip()
                if not u:
                    continue

                # If single unit itself is huge (> max_chunk_chars), split by character slices
                if len(u) > max_chunk_chars:
                    if len(current_text.strip()) > len(f"### Section: {section_title}"):
                        chunks.append(
                            {
                                "id": f"chunk_{chunk_idx:04d}",
                                "content": current_text.strip(),
                                "metadata": {
                                    "section": section_title,
                                    "source": "knowledge.md",
                                    "chunk_index": chunk_idx,
                                    "char_count": len(current_text.strip()),
                                },
                            }
                        )
                        chunk_idx += 1
                        current_text = f"### Section: {section_title} (cont.)\n"

                    start = 0
                    while start < len(u):
                        end = min(start + max_chunk_chars, len(u))
                        piece = u[start:end]
                        chunks.append(
                            {
                                "id": f"chunk_{chunk_idx:04d}",
                                "content": f"### Section: {section_title}\n{piece.strip()}",
                                "metadata": {
                                    "section": section_title,
                                    "source": "knowledge.md",
                                    "chunk_index": chunk_idx,
                                    "char_count": len(piece.strip()),
                                },
                            }
                        )
                        chunk_idx += 1
                        start += max_chunk_chars - overlap_chars
                    continue

                if len(current_text) + len(u) + 2 > max_chunk_chars and len(current_text.strip()) > len(f"### Section: {section_title}"):
                    chunks.append(
                        {
                            "id": f"chunk_{chunk_idx:04d}",
                            "content": current_text.strip(),
                            "metadata": {
                                "section": section_title,
                                "source": "knowledge.md",
                                "chunk_index": chunk_idx,
                                "char_count": len(current_text.strip()),
                            },
                        }
                    )
                    chunk_idx += 1
                    overlap_tail = current_text[-overlap_chars:] if len(current_text) > overlap_chars else ""
                    current_text = f"### Section: {section_title} (cont.)\n...{overlap_tail.strip()}\n\n{u}\n"
                else:
                    current_text += u + "\n\n"

            if len(current_text.strip()) > len(f"### Section: {section_title}"):
                chunks.append(
                    {
                        "id": f"chunk_{chunk_idx:04d}",
                        "content": current_text.strip(),
                        "metadata": {
                            "section": section_title,
                            "source": "knowledge.md",
                            "chunk_index": chunk_idx,
                            "char_count": len(current_text.strip()),
                        },
                    }
                )
                chunk_idx += 1

    return chunks


def run_ingest(force_reset: bool = True) -> Dict[str, Any]:
    """Execute the full ingestion pipeline.

    1. Reads knowledge.md
    2. Generates semantic chunks
    3. Clears Pinecone namespace to avoid duplicates
    4. Upserts chunks with embeddings
    """
    logger.info("Starting portfolio knowledge ingestion...")
    logger.info("Reading knowledge file: %s", KNOWLEDGE_FILE)

    content = load_knowledge_file(KNOWLEDGE_FILE)
    logger.info("Knowledge file read successfully (%d characters).", len(content))

    chunks = chunk_markdown(content)
    logger.info("Created %d semantic chunks.", len(chunks))

    vs = PortfolioVectorStore()

    if force_reset:
        logger.info(
            "Resetting Pinecone namespace '%s' in index '%s' to avoid duplicate documents...",
            vs.namespace or "__default__",
            vs.index_name,
        )
        vs.reset_namespace()

    ids = [c["id"] for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    logger.info("Embedding and upserting %d documents into Pinecone...", len(ids))
    count = vs.upsert_documents(ids=ids, documents=documents, metadatas=metadatas)

    total_in_db = vs.count()
    logger.info("Ingestion complete! Total documents in vector store: %d", total_in_db)

    return {
        "status": "success",
        "vector_store": "pinecone",
        "chunks_created": len(chunks),
        "documents_upserted": count,
        "total_in_index": total_in_db,
        "index_name": vs.index_name,
        "namespace": vs.namespace or "__default__",
    }


if __name__ == "__main__":
    result = run_ingest(force_reset=True)
    print("\n" + "=" * 60)
    print("PORTFOLIO CHATBOT INGESTION SUMMARY")
    print("=" * 60)
    for k, v in result.items():
        print(f" - {k}: {v}")
    print("=" * 60)
