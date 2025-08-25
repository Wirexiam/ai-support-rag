# tests/conftest.py
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Callable, Tuple

import pytest


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH = RESULTS_DIR / "test_results.jsonl"
CSV_PATH   = RESULTS_DIR / "test_results.csv"


def _append_jsonl(record: Dict[str, Any]) -> None:
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_csv(record: Dict[str, Any]) -> None:
    exists = CSV_PATH.exists()
    # Жёсткий и стабильный порядок колонок
    fieldnames = [
        "ts", "nodeid", "case", "status", "duration_sec",
        "question", "answer", "latency_sec", "context_len",
    ]
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        row = {k: record.get(k) for k in fieldnames}
        w.writerow(row)


@pytest.fixture(scope="session", autouse=True)
def _clean_results_dir() -> None:
    # По желанию: очищать старые результаты в начале сессии
    # Закомментируй, если нужно накопление истории
    for p in (JSONL_PATH, CSV_PATH):
        if p.exists():
            p.unlink()


@pytest.fixture
def record_result(request) -> Callable[..., None]:
    """
    Фикстура возвращает функцию, которой можно передать
    произвольные поля для записи в JSONL/CSV.
    """
    nodeid = request.node.nodeid

    def _record(
        *,
        case: str,
        status: str,
        duration_sec: float,
        question: str = "",
        answer: str = "",
        latency_sec: float | None = None,
        context_len: int | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        rec = {
            "ts": ts,
            "nodeid": nodeid,
            "case": case,
            "status": status,
            "duration_sec": round(float(duration_sec), 3),
            "question": question,
            "answer": answer,
            "latency_sec": latency_sec,
            "context_len": context_len,
        }
        if extra:
            rec.update(extra)
        _append_jsonl(rec)
        _append_csv(rec)

    return _record
