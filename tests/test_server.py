import pytest
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a FastAPI TestClient with temp memory."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    # Patch the memory module to use temp dir
    import core.memory as mem_module
    test_db = str(tmp_path / "test_server.db")
    monkeypatch.setattr("server.memory", mem_module.AgentMemory(test_db))
    monkeypatch.setattr("server.learner", None)  # will be set after import

    from fastapi.testclient import TestClient
    from server import app, ServerLearner, memory as server_memory

    # Re-patch after import
    import server
    server.memory = mem_module.AgentMemory(test_db)
    server.learner = ServerLearner(server.memory)

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "keys_set" in data
    assert "memory" in data


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "status" in r.json()


def test_learn_empty(client):
    r = client.get("/learn")
    assert r.status_code == 200
    data = r.json()
    assert data["total_runs"] == 0


def test_learn_lessons(client):
    r = client.get("/learn/lessons")
    assert r.status_code == 200
    assert "lessons" in r.json()


def test_learn_episodes(client):
    r = client.get("/learn/episodes")
    assert r.status_code == 200
    assert "episodes" in r.json()


def test_learn_recommend(client):
    r = client.get("/learn/recommend")
    assert r.status_code == 200
    data = r.json()
    assert "provider" in data


def test_learn_providers_empty(client):
    r = client.get("/learn/providers")
    assert r.status_code == 200


def test_keys_update(client):
    r = client.post("/keys", json={"keys": {"test_provider": "test-key"}})
    assert r.status_code == 200


def test_rate_nonexistent(client):
    r = client.post("/learn/rate/nonexistent", json={"episode_id": "x", "rating": 5})
    assert r.status_code == 404


def test_reset_memory(client):
    r = client.delete("/learn/reset")
    assert r.status_code == 200
    assert r.json()["ok"]
