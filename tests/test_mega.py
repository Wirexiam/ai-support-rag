# tests/test_mega.py
from __future__ import annotations

import re
import time
from typing import List, Dict, Any

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# Набор сквозных сценариев (можно расширять)
CASES = [
    {
        "name": "unknown_ru",
        "question": "Как готовить борщ?",
        "expect_answer_regex": r"^Я не знаю\.$",
        "expect_ctx_min": 0,   # контекст может быть пуст
    },
    {
        "name": "exchange_en",
        "question": "How to exchange item?",
        "expect_substrings": ["exchange", "обмен"],  # хотя бы одно
        "expect_ctx_min": 0,
    },
    {
        "name": "refund_ru",
        "question": "Как оформить возврат средств?",
        "expect_substrings": ["возврат средств происходит по правилам возврата товара"],
        "expect_ctx_min": 1,
    },
]


def _assert_answer_matches(answer: str, case: Dict[str, Any]) -> None:
    # Либо строгая регулярка, либо подстроки
    if "expect_answer_regex" in case:
        rx = re.compile(case["expect_answer_regex"])
        assert rx.search(answer) is not None, f"answer regex mismatch: {answer!r}"
    if "expect_substrings" in case:
        subs = case["expect_substrings"]
        assert any(s.lower() in answer.lower() for s in subs), f"answer substrings not found: {answer!r}"


def _no_square_refs(answer: str) -> None:
    assert not re.search(r"\[\s*\d+(\s*,\s*\d+)*\s*\]", answer), f"answer contains [n]: {answer!r}"


def _assert_structure(data: Dict[str, Any]) -> None:
    assert "answer" in data and "context" in data and "latency_sec" in data
    assert isinstance(data["context"], list)
    assert isinstance(data["latency_sec"], (int, float))


def _run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    resp = client.post("/ask", json={"question": case["question"]})
    dt = time.time() - t0
    assert resp.status_code == 200, resp.text

    data = resp.json()
    _assert_structure(data)
    _assert_answer_matches(data["answer"], case)
    _no_square_refs(data["answer"])

    expect_ctx_min = int(case.get("expect_ctx_min", 0))
    assert len(data["context"]) >= expect_ctx_min

    # Технический лимит на время ответа, можно ослабить при реальном LLM
    assert 0 <= data["latency_sec"] <= 30

    return {
        "duration_sec": dt,
        "answer": data["answer"],
        "latency_sec": float(data["latency_sec"]),
        "context_len": len(data["context"]),
    }


def test_mega_suite(record_result):
    for case in CASES:
        result = _run_case(case)
        record_result(
            case=case["name"],
            status="passed",
            duration_sec=result["duration_sec"],
            question=case["question"],
            answer=result["answer"],
            latency_sec=result["latency_sec"],
            context_len=result["context_len"],
        )
