"""Main FastAPI application entry point for Umang P. Kaklotar's Portfolio backend.

Runs the backend API and provides the chatbot endpoint at /api/chat.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.routes.chatbot import router as chatbot_router
from app.chatbot.vector_store import PortfolioVectorStore

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Initializing Portfolio Backend & Vector Store...")
    try:
        vs = PortfolioVectorStore()
        count = vs.count()
        logger.info("ChromaDB vector store connected: %d documents in collection '%s'.", count, vs.collection_name)
        if count == 0:
            logger.warning("Vector store is empty! Run 'python -m app.chatbot.ingest' to index knowledge.md.")
    except Exception as e:
        logger.warning("Could not initialize ChromaDB at startup: %s", e)
    yield
    logger.info("Portfolio Backend shutting down.")


app = FastAPI(
    title="Umang P. Kaklotar — Portfolio API & AI Chatbot",
    description="Backend microservice providing RAG-powered chatbot endpoints and portfolio services.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS so the React frontend (Vite on localhost:5173 or deployed origin) can connect cleanly
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173").split(",")
allowed_origins = [orig.strip() for orig in allowed_origins if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if "*" not in allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(chatbot_router)


@app.get("/", tags=["General"])
async def root():
    """Root health and discovery endpoint."""
    return {
        "name": "Umang P. Kaklotar Portfolio API",
        "status": "online",
        "docs_url": "/docs",
        "chat_endpoint": "/api/chat",
        "status_endpoint": "/api/chat/status",
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Simple ping-pong health check."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)

