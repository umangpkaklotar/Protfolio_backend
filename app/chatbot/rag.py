"""Retrieval-Augmented Generation (RAG) module for Umang P. Kaklotar's Portfolio Chatbot.

Orchestrates similarity search via Pinecone vector store, constructs context-grounded
prompts, and generates accurate answers using Google Gemini LLM.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Ensure backend/.env is loaded with override=True
env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    load_dotenv(override=True)

from .vector_store import PortfolioVectorStore
from .embeddings import get_gemini_api_key

logger = logging.getLogger(__name__)

# Default Gemini model for answering questions (gemini-3.1-flash-lite is the active default)
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

SYSTEM_INSTRUCTION = """You are Umang AI, the official AI Portfolio Assistant for Umang P. Kaklotar.
Your role is to answer questions from visitors, recruiters, engineering managers, and clients about Umang's skills, projects, experience, education, background, and services.

STRICT GUIDELINES:
1. Base your answer strictly and accurately on the provided Context excerpts from Umang's portfolio knowledge base.
2. If the user asks something that is NOT covered in the context, do not invent or guess. Politely reply:
   "I don't have that specific information in Umang's portfolio records. You can contact Umang directly at upkaklotar7@gmail.com."
3. Highlight his flagship achievements when relevant:
   - PCB AI Defect Inspection & Smart Manufacturing System (Ultralytics YOLOv8, 7 defect classes, OpenCV, ROS 2 Jazzy, Gazebo Sim 8 digital twin, RViz2, FastAPI, MongoDB, React).
   - Document AI — Enterprise Multi-Format RAG Platform (Gemini Embedding-2, FAISS IndexFlatIP, Groq LPU, LangChain, Clerk JWT auth, SSE streaming).
   - Real Estate Valuation & Home Price Prediction (Scikit-Learn regression, EDA, R² score 0.942).
   - Frontend Intern at Jemistry Info Solutions LLP (6 months, 2025).
   - BCA graduate (2025) from Smt ZS Patel College of Management and Technology, Surat.
   - AI/ML Training: currently attending AI/ML classes at DataMites in Bangalore, India.
4. Keep answers professional, concise, articulate, and well-structured with bullet points where appropriate.
5. Never contradict the provided context.
"""


class PortfolioRAG:
    """RAG pipeline coordinator."""

    def __init__(
        self,
        vector_store: Optional[PortfolioVectorStore] = None,
        model_name: Optional[str] = None,
        top_k: int = 4,
    ):
        self.vector_store = vector_store or PortfolioVectorStore()
        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.top_k = top_k

    def retrieve_context(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks from Pinecone."""
        return self.vector_store.query(question, n_results=self.top_k)

    def _call_gemini_llm(self, prompt: str, api_key: str) -> str:
        """Call Gemini model via google.genai, google.generativeai, or REST API

        with automatic fallback to active models if the configured model is deprecated.
        """
        # Determine candidate models to try in order
        configured = (self.model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)).replace("models/", "").strip()
        candidates = [configured] if configured else []
        for fb in ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"]:
            if fb not in candidates:
                candidates.append(fb)

        last_error = None

        for model_to_try in candidates:
            # 1. Try google.genai SDK
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model_to_try,
                    contents=prompt,
                )
                if hasattr(response, "text") and response.text:
                    return response.text.strip()
            except Exception as e:
                last_error = e
                logger.debug("google.genai call failed for model '%s': %s", model_to_try, e)

            # 2. Try google.generativeai SDK
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=api_key)
                model = legacy_genai.GenerativeModel(model_to_try)
                response = model.generate_content(prompt)
                if hasattr(response, "text") and response.text:
                    return response.text.strip()
            except Exception as e:
                last_error = e
                logger.debug("google.generativeai call failed for model '%s': %s", model_to_try, e)

            # 3. Direct REST API fallback
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_to_try}:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 800,
                    },
                }
                res = requests.post(url, json=payload, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    candidates_data = data.get("candidates", [])
                    if candidates_data:
                        parts = candidates_data[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                else:
                    last_error = RuntimeError(f"HTTP {res.status_code}: {res.text[:150]}")
            except Exception as e:
                last_error = e
                logger.debug("Gemini REST API failed for model '%s': %s", model_to_try, e)

        raise RuntimeError(f"Failed to obtain response from Gemini API across candidates {candidates}: {last_error}")

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Execute RAG pipeline for a user question.

        Returns:
            dict with 'answer' (str) and 'sources' (list of str).
        """
        question = question.strip()
        if not question:
            return {
                "answer": "Please ask a question about Umang P. Kaklotar's portfolio, skills, or projects.",
                "sources": [],
            }

        # 1. Retrieve relevant context
        chunks = self.retrieve_context(question)

        # Build sources list from metadata
        sources: List[str] = []
        for c in chunks:
            sec = c.get("metadata", {}).get("section", "")
            src = c.get("metadata", {}).get("source", "knowledge.md")
            label = f"{src} ({sec})" if sec else src
            if label not in sources:
                sources.append(label)

        # 2. Check for empty vector store
        if not chunks:
            return {
                "answer": (
                    "The portfolio knowledge base is currently empty or has not been ingested yet. "
                    "Please run the ingestion script: `python -m app.chatbot.ingest` inside the backend directory."
                ),
                "sources": [],
            }

        # 3. Format context string
        context_blocks = []
        for i, c in enumerate(chunks, 1):
            sec = c.get("metadata", {}).get("section", "General")
            context_blocks.append(f"[Excerpt {i} - Section: {sec}]\n{c['content']}")
        context_str = "\n\n".join(context_blocks)

        # 4. Check for Gemini API key
        api_key = get_gemini_api_key()
        if not api_key:
            answer = (
                "⚠️ **Gemini API Key Required**\n\n"
                                "The portfolio similarity search successfully retrieved relevant context from Pinecone, "
                "but `GEMINI_API_KEY` is not yet configured in `backend/.env` to generate an LLM response.\n\n"
                "**Retrieved Context Summary:**\n"
                f"{chunks[0]['content'][:450]}...\n\n"
                "To enable live AI generation, add your API key to `backend/.env`:\n"
                "```env\nGEMINI_API_KEY=your_key_here\n```"
            )
            return {
                "answer": answer,
                "sources": sources,
            }

        # 5. Build full prompt
        prompt = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"=== CONTEXT FROM UMANG'S PORTFOLIO KNOWLEDGE BASE ===\n"
            f"{context_str}\n\n"
            f"=== USER QUESTION ===\n"
            f"{question}\n\n"
            f"=== YOUR ANSWER ==="
        )

        # 6. Generate answer
        try:
            llm_answer = self._call_gemini_llm(prompt, api_key)
            return {
                "answer": llm_answer,
                "sources": sources,
            }
        except Exception as e:
            logger.error("Error generating answer via LLM: %s", e)
            return {
                "answer": (
                    f"I encountered an error communicating with the Gemini model: {str(e)}. "
                    "Please verify your `GEMINI_API_KEY` and internet connectivity."
                ),
                "sources": sources,
            }


# Global default instance
_rag_instance: Optional[PortfolioRAG] = None


def get_rag_pipeline() -> PortfolioRAG:
    """Singleton getter for RAG pipeline."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = PortfolioRAG()
    return _rag_instance
