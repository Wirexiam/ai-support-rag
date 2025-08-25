from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_ask_stub(monkeypatch):
    # подменяем generator.ask, чтобы не дёргать реальный API
    from app import main
    def fake_answer(q, ctx): return "stub answer"
    monkeypatch.setattr(main.generator, "ask", fake_answer)

    r = client.post("/ask", json={"question": "Тестовый вопрос"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert body["answer"] == "stub answer"
    assert "context" in body
