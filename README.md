# Umang P. Kaklotar — Portfolio AI Chatbot Backend

A production-ready FastAPI backend and Retrieval-Augmented Generation (RAG) chatbot integrated directly into Umang P. Kaklotar's portfolio codebase.

The chatbot provides verified, context-grounded answers about Umang's skills, experience, projects, education, and services to recruiters, clients, and portfolio visitors.

---
# To Run Backend
.\venv\Scripts\uvicorn.exe app.main:app --reload
 --port 8000


## 1. What the Chatbot Does

- **Domain-Specific Portfolio Knowledge**: Ingests comprehensive details from `backend/app/chatbot/knowledge.md` covering Computer Vision, Deep Learning (YOLOv8, AOI), Industrial Digital Twins (ROS 2, Gazebo Sim 8), Enterprise RAG, full-stack web development, and work experience.
- **Fast Similarity Search**: Uses a managed serverless **Pinecone** vector store to index and retrieve relevant portfolio excerpts in milliseconds.
- **Zero Hallucination Guardrails**: The system prompt strictly limits answers to the verified portfolio context. If a detail is missing, it transparently directs the user to Umang's direct email (`upkaklotar7@gmail.com`).
- **Google Gemini Integration**: Uses Google's official Gemini models (`gemini-1.5-flash` or `gemini-2.0-flash`) and embeddings (`models/text-embedding-004`).
- **Developer-Friendly Fallback**: If `GEMINI_API_KEY` is not yet configured, the similarity search still functions and returns retrieved context along with setup guidance, ensuring zero crashes during local setup.

---

## 2. Folder Structure

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application entry point & CORS
│   │
│   ├── chatbot/
│   │   ├── __init__.py
│   │   ├── knowledge.md            # Comprehensive source of portfolio truth
│   │   ├── ingest.py               # Ingestion & chunking script for Pinecone
│   │   ├── rag.py                  # RAG pipeline orchestration & prompt formatting
│   │   ├── embeddings.py           # Gemini embedding generator
│   │   └── vector_store.py         # Pinecone serverless index client & querying
│   │
│   └── routes/
│       ├── __init__.py
│       └── chatbot.py              # POST /api/chat and GET /api/chat/status
│
├── .env.example                    # Environment variable template
├── requirements.txt                # Python dependencies
└── README.md                       # Backend documentation
```

---

## 3. How to Install Dependencies

Make sure you are inside the `backend/` directory:

```bash
cd backend
```

### Option A: Using the existing virtual environment

Activate the existing virtual environment:

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### Option B: Creating a fresh virtual environment (optional)

```bash
python -m venv venv
# Activate it as shown above
```

### Install packages:

```bash
pip install -r requirements.txt
```

---

## 4. How to Configure `.env`

Copy the provided `.env.example` to create your local `.env`:

```bash
cp .env.example .env
```

*(On Windows Command Prompt: `copy .env.example .env`)*

Edit `backend/.env` with your settings:

```env
# Get your free Gemini API key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_actual_gemini_api_key_here

# LLM Generation Model
GEMINI_MODEL=gemini-1.5-flash

# Embedding Model
GEMINI_EMBEDDING_MODEL=models/text-embedding-004

# Pinecone Vector Database Configuration
PINECONE=your_actual_pinecone_api_key_here
PINECONE_INDEX_NAME=portfolio-knowledge
PINECONE_NAMESPACE=portfolio
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Server settings
HOST=0.0.0.0
PORT=8000

# CORS Allowed Origins
CORS_ORIGINS=http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173
```

---


## 5. How to Generate / Update the Pinecone Vector Store

Run the ingestion script from the `backend/` directory:

```bash
python -m app.chatbot.ingest
```

Or run directly:

```bash
python app/chatbot/ingest.py
```

### What this script does:
1. Loads `backend/app/chatbot/knowledge.md`.
2. Segments sections and headers into overlapping semantic chunks (max 850 characters, 120 overlap).
3. Connects to Pinecone using the `PINECONE` API key from `backend/.env` and creates the serverless index if needed.
4. Clears the configured namespace to **prevent duplicate documents** on repeated runs.
5. Embeds each chunk and upserts with section metadata (`section`, `source`, `chunk_index`) and chunk text.
6. Prints a summary of indexed documents.

Whenever you update `knowledge.md`, simply re-run the command above to refresh the vector store.

---

## 6. How to Start the Backend Server

From the `backend/` directory:

```bash
uvicorn app.main:app --reload --port 8000
```

Or run with Python directly:

```bash
python -m app.main
```

Once running:
- **API Base URL**: `http://localhost:8000`
- **Swagger Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **Chatbot Status**: `http://localhost:8000/api/chat/status`

---

## 7. How the RAG Pipeline Works

```text
knowledge.md
      ↓
Document chunking (900 chars / 150 overlap with section headings)
      ↓
Embeddings (Google Gemini text-embedding-004)
      ↓
Pinecone (Serverless index `portfolio-knowledge`, namespace `portfolio`)
      ↓
[ User Query: "What is Umang's PCB Defect project?" ]
      ↓
Similarity Search (Top-K closest cosine similarity excerpts)
      ↓
Context Grounding + System Instructions
      ↓
LLM (Google Gemini 1.5 Flash @ temperature 0.2)
      ↓
Answer + Source Section Citations
```

1. **Retrieval**: When a query hits `POST /api/chat`, the query is embedded and matched against Pinecone's index.
2. **Context Formatting**: Top 4 matching chunks are assembled with their respective section headers.
3. **Prompt Framing**: The context is wrapped inside strict system instructions preventing speculation or hallucinations.
4. **Generation**: The prompt is processed by Google Gemini, returning a concise, professional answer with referenced sources.

---

## 8. How to Test `/api/chat`

### Using cURL:

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What technologies does Umang use for Computer Vision?\"}"
```

### Example Response:

```json
{
  "answer": "Umang specializes in Computer Vision using Ultralytics YOLOv8 for object detection and defect classification, OpenCV (cv2) for image preprocessing and annotation, and PyTorch for deep learning. He also works with ROS 2 (Jazzy) and Gazebo Sim 8 for industrial conveyor automation and digital twin inspection.",
  "sources": [
    "knowledge.md (Skills)",
    "knowledge.md (Projects)",
    "knowledge.md (Project Technologies)"
  ]
}
```

### Testing with Swagger UI:
1. Open `http://localhost:8000/docs` in your browser.
2. Expand the `POST /api/chat` section.
3. Click **Try it out**, enter your question, and click **Execute**.

---

## 9. How the Frontend Connects to It

In your React frontend (e.g. from any React component or custom hook):

```javascript
const askPortfolioChatbot = async (userMessage) => {
  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  
  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: userMessage }),
    });

    if (!response.ok) {
      throw new Error(`Server returned status: ${response.status}`);
    }

    const data = await response.json();
    // data.answer => String with the AI answer
    // data.sources => Array of referenced sections
    return data;
  } catch (error) {
    console.error('Chatbot error:', error);
    throw error;
  }
};
```
