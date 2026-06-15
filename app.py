"""
app.py
------
Flask REST API — On-Premise SLM Document Intelligence Service
Swayam Ltd · ML Engineering · Production Service

Privacy-preserving document Q&A using Phi-3 Mini (Ollama) + ChromaDB RAG.
All inference runs locally — no data leaves the machine.

Endpoints:
    POST /ingest        Ingest documents into the vector store
    POST /query         Ask a question — returns answer + evaluation scores
    POST /summarise     Summarise all ingested documents
    GET  /logs          Query recent logged queries from SQLite
    GET  /stats         Aggregate evaluation statistics
    GET  /status        RAG pipeline status (chunks in DB, Ollama health)
    GET  /health        Service health check
    GET  /              API info

Run:
    python app.py
"""

import os
import time
from flask import Flask, request, jsonify

from rag import query_pipeline, ingest_documents, get_ingestion_status, generate_answer, retrieve_context
from db.database import init_db, log_query, get_recent_queries, get_stats

app = Flask(__name__)

# ── Startup ───────────────────────────────────────────────────────────────────
with app.app_context():
    init_db()
    print("✓ Database initialised")


# ── Helper ────────────────────────────────────────────────────────────────────

def check_ollama() -> bool:
    """Check if Ollama is running locally."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service":     "SLM Document Intelligence Service",
        "version":     "1.0.0",
        "company":     "Swayam Ltd",
        "description": (
            "On-premise document Q&A and summarisation service. "
            "Uses Phi-3 Mini (SLM) via Ollama with a ChromaDB RAG pipeline. "
            "All inference runs locally — no data leaves the machine."
        ),
        "model":       "Phi-3 Mini (Ollama) — runs on CPU",
        "embeddings":  "all-MiniLM-L6-v2 (SentenceTransformers — local)",
        "endpoints": {
            "POST /ingest":    "Ingest .txt documents from documents/ folder",
            "POST /query":     "Ask a question — returns answer + hallucination score",
            "POST /summarise": "Summarise all ingested documents",
            "GET  /logs":      "Recent query logs from SQLite (?limit=50)",
            "GET  /stats":     "Aggregate evaluation statistics",
            "GET  /status":    "RAG pipeline status",
            "GET  /health":    "Service health check",
        },
        "author": "Bharadwaj Rachuri",
    })


@app.route("/health", methods=["GET"])
def health():
    checks = {
        "database": "ok",
        "ollama":   "ok" if check_ollama() else "unavailable — start Ollama with: ollama serve",
        "rag":      "ok",
    }
    try:
        get_stats()
    except Exception as e:
        checks["database"] = f"error: {str(e)}"

    status    = "healthy" if checks["ollama"] == "ok" and checks["database"] == "ok" else "degraded"
    http_code = 200 if status == "healthy" else 503
    return jsonify({"status": status, "checks": checks}), http_code


@app.route("/status", methods=["GET"])
def status():
    """RAG pipeline status — chunks in DB, Ollama health."""
    ingestion = get_ingestion_status()
    return jsonify({
        "rag_pipeline":  ingestion,
        "ollama_online": check_ollama(),
        "model":         "phi3",
        "embeddings":    "all-MiniLM-L6-v2",
    })


@app.route("/ingest", methods=["POST"])
def ingest():
    """
    POST /ingest
    ------------
    Ingest all .txt documents from the documents/ folder into ChromaDB.

    Response 200:
        { "files_ingested": int, "chunks_stored": int, "message": str }
    """
    try:
        result = ingest_documents()
        return jsonify({
            "files_ingested": result["files_ingested"],
            "chunks_stored":  result["chunks_stored"],
            "message":        f"Successfully ingested {result['files_ingested']} documents into {result['chunks_stored']} chunks.",
        }), 200
    except Exception as e:
        return jsonify({"error": f"Ingestion failed: {str(e)}"}), 500


@app.route("/query", methods=["POST"])
def query():
    """
    POST /query
    -----------
    Ask a question against the ingested documents.

    Request JSON:
        { "question": "What is Bharadwaj's published research about?" }

    Response 200:
        {
            "question":           str,
            "answer":             str,
            "sources":            [str],
            "groundedness_score": float,
            "hallucination_flag": bool,
            "safety_score":       float,
            "chunks_retrieved":   int,
            "processing_ms":      int,
            "query_id":           int
        }
    """
    data     = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Request must be JSON with a 'question' field."}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400
    if len(question) > 1000:
        return jsonify({"error": "Question too long. Maximum 1000 characters."}), 400

    if not check_ollama():
        return jsonify({
            "error": "Ollama is not running. Start it with: ollama serve",
            "hint":  "Also ensure phi3 model is pulled: ollama pull phi3"
        }), 503

    try:
        result = query_pipeline(question)

        # Log to SQLite
        query_id = log_query(
            question=result["question"],
            answer=result["answer"],
            context_used=result.get("context_preview", ""),
            groundedness_score=result["groundedness_score"],
            hallucination_flag=result["hallucination_flag"],
            safety_score=result["safety_score"],
            chunks_retrieved=result["chunks_retrieved"],
            processing_ms=result["processing_ms"],
        )

        return jsonify({
            "question":           result["question"],
            "answer":             result["answer"],
            "sources":            result["sources"],
            "groundedness_score": result["groundedness_score"],
            "hallucination_flag": result["hallucination_flag"],
            "safety_score":       result["safety_score"],
            "chunks_retrieved":   result["chunks_retrieved"],
            "processing_ms":      result["processing_ms"],
            "query_id":           query_id,
        }), 200

    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500


@app.route("/summarise", methods=["POST"])
def summarise():
    """
    POST /summarise
    ---------------
    Summarise all ingested documents in 3-5 sentences.

    Response 200:
        { "summary": str, "sources": [str], "processing_ms": int }
    """
    if not check_ollama():
        return jsonify({"error": "Ollama is not running. Start it with: ollama serve"}), 503

    try:
        start  = time.time()
        chunks = retrieve_context("summarise the main topics and key information in these documents", top_k=6)

        if not chunks:
            return jsonify({"error": "No documents ingested. Call POST /ingest first."}), 400

        summary = generate_answer(
            "Please provide a concise 3-5 sentence summary of the main topics and key information covered in these documents.",
            chunks,
        )
        processing_ms = int((time.time() - start) * 1000)
        sources       = list({c["source"] for c in chunks})

        return jsonify({
            "summary":        summary,
            "sources":        sources,
            "processing_ms":  processing_ms,
        }), 200

    except Exception as e:
        return jsonify({"error": f"Summarisation failed: {str(e)}"}), 500


@app.route("/logs", methods=["GET"])
def logs():
    """
    GET /logs
    ---------
    Retrieve recent query logs from SQLite.

    Query params:
        limit  int  Max rows (default 50, max 200)
    """
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    logs_data = get_recent_queries(limit=limit)
    return jsonify({"count": len(logs_data), "queries": logs_data}), 200


@app.route("/stats", methods=["GET"])
def stats():
    """
    GET /stats
    ----------
    Aggregate evaluation statistics across all logged queries.
    """
    return jsonify(get_stats()), 200


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found. See GET / for available endpoints."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print("\nSLM Document Intelligence Service")
    print(f"Running on http://0.0.0.0:{port}")
    print("Make sure Ollama is running: ollama serve")
    print("Make sure phi3 is pulled:   ollama pull phi3")
    app.run(host="0.0.0.0", port=port, debug=debug)
