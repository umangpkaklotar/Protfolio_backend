"""Chatbot API routes for Umang P. Kaklotar's Portfolio backend.

Provides:
- POST /api/chat : Ask a question and receive RAG-generated answer + sources
- GET  /api/chat/status : Check vector store health, count, and API key presence
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.chatbot.rag import get_rag_pipeline
from app.chatbot.embeddings import get_gemini_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Chatbot"])


class ChatRequest(BaseModel):
    """Chat request payload from frontend."""

    message: Optional[str] = Field(None, description="The user's question or message")
    question: Optional[str] = Field(None, description="Alias for message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What projects has Umang built?"
            }
        }
    }


class ChatResponse(BaseModel):
    """Chat response payload returned to frontend."""

    answer: str = Field(..., description="The AI response grounded in portfolio context")
    sources: List[str] = Field(default_factory=list, description="Referenced sections or source documents")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Umang has built four key projects including the PCB AI Defect Inspection System...",
                "sources": ["knowledge.md (Projects)", "knowledge.md (Project Features)"]
            }
        }
    }


class StatusResponse(BaseModel):
    """Vector store and chatbot configuration status."""

    status: str
    documents_indexed: int
    gemini_api_configured: bool
    chroma_db_dir: str
    collection_name: str


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with Umang's Portfolio AI Assistant",
)
async def chat_endpoint(payload: ChatRequest):
    """Accepts a question from a portfolio visitor, retrieves relevant

    sections from ChromaDB, and returns a verified, factual response with sources.
    """
    user_query = payload.message or payload.question
    if not user_query or not user_query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'message' or 'question' field is required and cannot be empty.",
        )

    try:
        rag = get_rag_pipeline()
        result = rag.answer_question(user_query)
        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
        )
    except Exception as e:
        logger.error("Chat endpoint error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}",
        )


@router.get(
    "/chat/status",
    response_model=StatusResponse,
    summary="Get Chatbot & Vector Store Status",
)
async def chat_status():
    """Returns the operational status of the ChromaDB vector store and API configuration."""
    try:
        rag = get_rag_pipeline()
        vs_status = rag.vector_store.get_status()
        has_key = bool(get_gemini_api_key())
        return StatusResponse(
            status="ready" if vs_status["document_count"] > 0 else "needs_ingestion",
            documents_indexed=vs_status["document_count"],
            gemini_api_configured=has_key,
            chroma_db_dir=vs_status["persist_dir"],
            collection_name=vs_status["collection_name"],
        )
    except Exception as e:
        logger.error("Status endpoint error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inspecting vector store: {str(e)}",
        )

