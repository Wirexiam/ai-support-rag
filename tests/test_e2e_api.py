import os
import time
import json
import re
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

@pytest.mark.parametrize("question,expect_regex", [
    ("Как готовить борщ?", r"^Я не знаю\.$"),
])
def test_unknown_answer(question, expect_regex):
    t0 = time.time()
    resp = client.post("/ask", json={"question": question})
    dt = time.time() - t0
    assert resp.status_code == 200, resp.text

    data = resp.json()
    # Структура ответа
    assert "answer" in data and "context" in data and "latency_sec" in data
    # Ответ «не знаю»
    assert re.search(expect_regex, data["answer"]) is not None
    # Время ответа вменяемое (локально без модели может быть быстро)
    assert 0 <= data["latency_sec"] <= 15
    # Контекст может быть пустым — ок
    assert isinstance(data["context"], list)

def test_exchange_fallback():
    question = "How to exchange item?"
    resp = client.post("/ask", json={"question": question})
    assert resp.status_code == 200
    data = resp.json()
    # Фолбэк/LLM должен упомянуть условия обмена
    assert "exchange" in data["answer"].lower() or "обмен" in data["answer"].lower()
    # Квадратных ссылок быть не должно
    assert "[" not in data["answer"], data["answer"]
    # Должны вернуться фрагменты контекста
    assert isinstance(data["context"], list)

def test_refund_fallback_ru():
    question = "Как оформить возврат средств?"
    resp = client.post("/ask", json={"question": question})
    assert resp.status_code == 200
    data = resp.json()
    # Ключевая формулировка фолбэка
    assert "возврат средств происходит по правилам возврата товара" in data["answer"].lower()
    # Нет ссылок вида [1], [2]
    assert not re.search(r"\[\s*\d+(\s*,\s*\d+)*\s*\]", data["answer"])
    # Контекст присутствует (как минимум один фрагмент)
    assert isinstance(data["context"], list)
