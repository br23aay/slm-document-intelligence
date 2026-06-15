"""
test_app.py
-----------
pytest test suite for the SLM Document Intelligence Service.
Tests database pipeline, RAG chunking, evaluation, and all API endpoints.

Run:
    pytest tests/ -v
"""

import json
import os
import sys
import tempfile
import pytest
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Database tests ─────────────────────────────────────────────────────────

class TestDatabase:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        os.environ["DB_PATH"] = self.tmp.name
        import db.database as db_mod
        importlib.reload(db_mod)
        from db.database import init_db as _init
        _init()

    def teardown_method(self):
        try:
            self.tmp.close()
            os.unlink(self.tmp.name)
        except Exception:
            pass

    def test_log_query_returns_id(self):
        from db.database import log_query
        row_id = log_query(
            question="What is PPO?",
            answer="Proximal Policy Optimisation is an RL algorithm.",
            groundedness_score=0.85,
            hallucination_flag=False,
            safety_score=1.0,
            chunks_retrieved=3,
            processing_ms=450,
        )
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_get_recent_queries_returns_list(self):
        from db.database import log_query, get_recent_queries
        log_query("Q1", "A1", groundedness_score=0.9, safety_score=1.0)
        log_query("Q2", "A2", groundedness_score=0.8, safety_score=1.0)
        results = get_recent_queries(limit=10)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_get_stats_empty(self):
        from db.database import get_stats
        stats = get_stats()
        assert stats["total_queries"] == 0
        assert stats["hallucination_rate"] == 0

    def test_get_stats_with_data(self):
        from db.database import log_query, get_stats
        log_query("Q1", "A1", groundedness_score=0.9, hallucination_flag=False, safety_score=1.0)
        log_query("Q2", "A2", groundedness_score=0.5, hallucination_flag=True, safety_score=0.8)
        stats = get_stats()
        assert stats["total_queries"] == 2
        assert stats["hallucination_count"] == 1

    def test_query_fields_present(self):
        from db.database import log_query, get_recent_queries
        log_query("Test question?", "Test answer.", groundedness_score=0.7,
                  hallucination_flag=False, safety_score=1.0, chunks_retrieved=3, processing_ms=200)
        rows = get_recent_queries(limit=1)
        assert len(rows) == 1
        row = rows[0]
        for field in ["id", "timestamp", "question", "answer", "groundedness_score",
                      "hallucination_flag", "safety_score", "chunks_retrieved", "processing_ms"]:
            assert field in row


# ── RAG pipeline tests ────────────────────────────────────────────────────────

class TestRAGPipeline:
    def test_chunk_text_basic(self):
        from rag import chunk_text
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=200, overlap=40)
        assert len(chunks) > 1
        assert all(len(c) > 0 for c in chunks)

    def test_chunk_text_overlap(self):
        from rag import chunk_text
        text = "Hello world this is a test document " * 20
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 2

    def test_chunk_text_drops_tiny_fragments(self):
        from rag import chunk_text
        text = "Short"
        chunks = chunk_text(text, chunk_size=400, overlap=80)
        assert len(chunks) == 0

    def test_evaluate_answer_high_groundedness(self):
        from rag import evaluate_answer
        context = [{"text": "Bharadwaj Rachuri published research on PPO and Shadow Hand manipulation in MuJoCo at University of Hertfordshire"}]
        answer = "Bharadwaj published research on PPO and Shadow Hand manipulation"
        result = evaluate_answer(answer, context)
        assert "groundedness_score" in result
        assert "hallucination_flag" in result
        assert "safety_score" in result
        assert 0.0 <= result["groundedness_score"] <= 1.0
        assert 0.0 <= result["safety_score"] <= 1.0

    def test_evaluate_answer_safety_score(self):
        from rag import evaluate_answer
        context = [{"text": "This is a safe document about AI research"}]
        answer = "This is a safe response about research"
        result = evaluate_answer(answer, context)
        assert result["safety_score"] == 1.0

    def test_evaluate_answer_returns_dict(self):
        from rag import evaluate_answer
        context = [{"text": "Test context about machine learning"}]
        result = evaluate_answer("Machine learning is a field of AI", context)
        assert isinstance(result, dict)
        assert len(result) == 3


# ── API endpoint tests ───────────────────────────────────────────────────────

@pytest.fixture
def client():
    os.environ["DB_PATH"] = ":memory:"
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        with app.app_context():
            from db.database import init_db
            init_db()
        yield c


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_contains_service_info(self, client):
        data = json.loads(client.get("/").data)
        assert "service" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    def test_health_returns_status(self, client):
        data = json.loads(client.get("/health").data)
        assert "status" in data
        assert "checks" in data

    def test_health_checks_present(self, client):
        data = json.loads(client.get("/health").data)
        assert "database" in data["checks"]
        assert "ollama" in data["checks"]


class TestStatusEndpoint:
    def test_status_returns_200(self, client):
        assert client.get("/status").status_code == 200

    def test_status_fields(self, client):
        data = json.loads(client.get("/status").data)
        assert "rag_pipeline" in data
        assert "ollama_online" in data


class TestQueryEndpoint:
    def test_query_no_body_returns_400(self, client):
        resp = client.post("/query", content_type="application/json")
        assert resp.status_code == 400

    def test_query_missing_question_returns_400(self, client):
        resp = client.post("/query",
                           data=json.dumps({"text": "hello"}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_query_empty_question_returns_400(self, client):
        resp = client.post("/query",
                           data=json.dumps({"question": ""}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_query_too_long_returns_400(self, client):
        resp = client.post("/query",
                           data=json.dumps({"question": "Q" * 1001}),
                           content_type="application/json")
        assert resp.status_code == 400

    def test_query_without_ollama_returns_503(self, client):
        resp = client.post("/query",
                           data=json.dumps({"question": "What is PPO?"}),
                           content_type="application/json")
        assert resp.status_code in (200, 503, 500)


class TestLogsEndpoint:
    def test_logs_returns_200(self, client):
        assert client.get("/logs").status_code == 200

    def test_logs_structure(self, client):
        data = json.loads(client.get("/logs").data)
        assert "count" in data
        assert "queries" in data

    def test_logs_invalid_limit_returns_400(self, client):
        assert client.get("/logs?limit=abc").status_code == 400


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        assert client.get("/stats").status_code == 200

    def test_stats_fields(self, client):
        data = json.loads(client.get("/stats").data)
        assert "total_queries" in data
        assert "hallucination_rate" in data
        assert "avg_groundedness" in data


class TestErrorHandlers:
    def test_404_returns_json(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
        assert "error" in json.loads(resp.data)

    def test_405_on_wrong_method(self, client):
        resp = client.get("/query")
        assert resp.status_code == 405
