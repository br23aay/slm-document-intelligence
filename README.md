# On-Premise SLM Document Intelligence Service

**Swayam Ltd — ML Engineering | Production Service**

A privacy-preserving document Q&A and summarisation service built for clients who cannot send sensitive data to external APIs. Uses **Phi-3 Mini** (Small Language Model) via Ollama with a **ChromaDB RAG pipeline**, served through a **Flask REST API**, with **SQLite logging** of every query and a built-in **hallucination and safety evaluation layer**.

All inference runs locally — no data leaves the machine.

---

## Stack

| Component | Technology |
|---|---|
| Small Language Model | Phi-3 Mini via Ollama (CPU, local) |
| RAG Pipeline | ChromaDB + SentenceTransformers (all-MiniLM-L6-v2) |
| API Framework | Flask 3.1 |
| Database Pipeline | SQLite — logs every query with evaluation scores |
| Evaluation Layer | Groundedness scoring, hallucination detection, safety scoring |
| CI/CD | GitHub Actions (lint → test) |

---

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest documents from documents/ folder into ChromaDB |
| `POST` | `/query` | Ask a question — returns answer + groundedness + hallucination score |
| `POST` | `/summarise` | Summarise all ingested documents |
| `GET` | `/logs` | Recent query logs from SQLite (?limit=50) |
| `GET` | `/stats` | Aggregate evaluation statistics |
| `GET` | `/status` | RAG pipeline status (chunks in DB, Ollama health) |
| `GET` | `/health` | Service health check |
| `GET` | `/` | API info |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/br23aay/slm-document-intelligence
cd slm-document-intelligence
pip install -r requirements.txt

# 2. Start Ollama and pull Phi-3 Mini
ollama serve
ollama pull phi3

# 3. Run tests
pytest tests/ -v

# 4. Start the API
python app.py

# 5. Ingest documents
curl -X POST http://localhost:5001/ingest

# 6. Query the documents
curl -X POST http://localhost:5001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Bharadwaj Rachuri published research about?"}'
```

---

## Knowledge Base

The default knowledge base contains:
- `bharadwaj_cv.txt` — Full CV including skills, experience, and education
- `ijres_paper.txt` — Full summary of the IJRES published research paper on Shadow Hand PPO

Add your own `.txt` documents to the `documents/` folder and call `POST /ingest`.

---

## API Usage

### Query — Ask a question

```bash
curl -X POST http://localhost:5001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What reinforcement learning algorithm was used in the Shadow Hand research?"}'
```

```json
{
  "question": "What reinforcement learning algorithm was used in the Shadow Hand research?",
  "answer": "Proximal Policy Optimisation (PPO) was used via Stable-Baselines3...",
  "sources": ["ijres_paper.txt"],
  "groundedness_score": 0.87,
  "hallucination_flag": false,
  "safety_score": 1.0,
  "chunks_retrieved": 4,
  "processing_ms": 1240,
  "query_id": 1
}
```

### Summarise all documents

```bash
curl -X POST http://localhost:5001/summarise
```

### View stats

```bash
curl http://localhost:5001/stats
```

```json
{
  "total_queries": 12,
  "avg_groundedness": 0.834,
  "avg_safety": 0.98,
  "hallucination_count": 1,
  "hallucination_rate": 0.083,
  "avg_processing_ms": 1180.4
}
```

---

## Evaluation Layer

Every query is automatically evaluated:

| Metric | Description |
|---|---|
| `groundedness_score` | 0–1. How well the answer is supported by retrieved context |
| `hallucination_flag` | True if answer contains confident claims not traceable to context |
| `safety_score` | 0–1. Penalises harmful content patterns |

All scores are logged to SQLite for audit and analytics.

---

## Skills Evidenced

- **SLMs** — Phi-3 Mini (3.8B parameter small language model) via Ollama
- **RAG Pipeline** — ChromaDB vector store, SentenceTransformer embeddings, semantic retrieval
- **Flask / REST API** — 7 production endpoints with full error handling
- **SQL Pipelines** — SQLite query logging with evaluation scores and aggregate analytics
- **Hallucination Scoring** — Built-in groundedness, hallucination detection, safety scoring layer

---

## Author

**Bharadwaj Rachuri** — ML & AI Engineer
[br23aay.github.io](https://br23aay.github.io) · [github.com/br23aay](https://github.com/br23aay)
*Built during employment at Swayam Ltd, London*
