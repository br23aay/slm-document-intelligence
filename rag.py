"""
rag.py
------
RAG pipeline for the SLM Document Intelligence Service.

Ingestion:  Loads .txt documents → chunks → embeds → stores in ChromaDB
Retrieval:  Embeds query → finds top-k similar chunks → returns context
Generation: Passes context + question to Ollama Phi-3 Mini → returns answer

All runs locally. No external API calls. No GPU required.
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions

DOCUMENTS_DIR  = "documents"
CHROMA_DIR     = "db/chroma"
COLLECTION_NAME = "bharadwaj_docs"
CHUNK_SIZE     = 400    # characters per chunk
CHUNK_OVERLAP  = 80     # overlap between chunks
TOP_K          = 4      # number of chunks to retrieve
OLLAMA_MODEL   = "phi3" # model name in Ollama

# ── Embedding function (local, CPU, no API key) ───────────────────────────────
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# ── ChromaDB client (singleton) ───────────────────────────────────────────────
_client     = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client     = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_embedding_fn,
        )
    return _collection


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks."""
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 30]  # drop tiny fragments


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_documents(documents_dir: str = DOCUMENTS_DIR) -> dict:
    """
    Load all .txt files from documents_dir, chunk them,
    embed and store in ChromaDB.

    Returns: { "files_ingested": int, "chunks_stored": int }
    """
    collection   = _get_collection()
    files        = glob.glob(os.path.join(documents_dir, "*.txt"))
    total_chunks = 0

    for filepath in files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)

        ids       = [f"{filename}__chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        # Upsert — safe to re-run
        collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    return {"files_ingested": len(files), "chunks_stored": total_chunks}


def get_ingestion_status() -> dict:
    """Return how many chunks are currently stored in ChromaDB."""
    try:
        collection = _get_collection()
        count      = collection.count()
        return {"status": "ready", "chunks_in_db": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_context(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Find the top_k most relevant chunks for the question.

    Returns list of { "text": str, "source": str, "chunk_index": int, "distance": float }
    """
    collection = _get_collection()
    results    = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":        doc,
            "source":      meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "distance":    round(dist, 4),
        })

    return chunks


# ── Generation via Ollama ─────────────────────────────────────────────────────

def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Send question + retrieved context to Ollama Phi-3 Mini.
    Returns the generated answer string.
    """
    import requests

    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    prompt = f"""You are a helpful document assistant. Answer the question using ONLY the information provided in the context below. If the answer is not in the context, say "I don't have enough information in my documents to answer that."

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_answer(answer: str, context_chunks: list[dict]) -> dict:
    """
    Lightweight evaluation of the generated answer.

    Checks:
    - Groundedness: how many context words appear in the answer
    - Hallucination flag: answer contains content not traceable to context
    - Safety: checks for harmful patterns

    Returns: { groundedness_score, hallucination_flag, safety_score }
    """
    context_text = " ".join(c["text"] for c in context_chunks).lower()
    answer_lower  = answer.lower()
    answer_words  = set(answer_lower.split())

    # Groundedness — proportion of answer words found in context
    context_words    = set(context_text.split())
    overlap          = answer_words & context_words
    stop_words       = {"the", "a", "an", "is", "in", "of", "and", "to", "i", "it", "that", "was", "for"}
    content_words    = answer_words - stop_words
    groundedness     = round(len(overlap & content_words) / max(len(content_words), 1), 4)

    # Hallucination flag — low groundedness + confident-sounding answer
    confident_phrases = ["definitely", "certainly", "always", "never", "exactly", "precisely"]
    has_confident     = any(p in answer_lower for p in confident_phrases)
    hallucination     = groundedness < 0.25 and has_confident

    # Safety score — penalise harmful patterns
    harmful_patterns  = ["harm", "kill", "attack", "illegal", "weapon", "exploit"]
    safety_score      = 1.0 - (0.2 * sum(p in answer_lower for p in harmful_patterns))
    safety_score      = round(max(0.0, safety_score), 4)

    return {
        "groundedness_score": groundedness,
        "hallucination_flag": hallucination,
        "safety_score":       safety_score,
    }


# ── Full RAG pipeline ─────────────────────────────────────────────────────────

def query_pipeline(question: str) -> dict:
    """
    Full RAG pipeline: retrieve → generate → evaluate.

    Returns complete result dict for API response and DB logging.
    """
    import time
    start = time.time()

    # Retrieve
    chunks = retrieve_context(question)

    if not chunks:
        return {
            "question":          question,
            "answer":            "No documents have been ingested yet. Please call POST /ingest first.",
            "sources":           [],
            "groundedness_score": 0.0,
            "hallucination_flag": False,
            "safety_score":      1.0,
            "chunks_retrieved":  0,
            "processing_ms":     0,
        }

    # Generate
    answer = generate_answer(question, chunks)

    # Evaluate
    evaluation   = evaluate_answer(answer, chunks)
    processing_ms = int((time.time() - start) * 1000)

    sources = list({c["source"] for c in chunks})

    return {
        "question":           question,
        "answer":             answer,
        "sources":            sources,
        "groundedness_score": evaluation["groundedness_score"],
        "hallucination_flag": evaluation["hallucination_flag"],
        "safety_score":       evaluation["safety_score"],
        "chunks_retrieved":   len(chunks),
        "processing_ms":      processing_ms,
        "context_preview":    chunks[0]["text"][:200] + "..." if chunks else "",
    }
