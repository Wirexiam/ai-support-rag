# app/generator.py
from __future__ import annotations

import json
import re
import time
from typing import List, Optional

import requests

from .config import settings

# ==== УТИЛИТЫ ====

def _looks_unknown(s: str) -> bool:
    """Эвристика: ответ пустой или модель 'сдалась'."""
    if not s:
        return True
    t = s.strip().lower()
    return (
        len(t) < 5
        or "я не знаю" in t
        or "не знаю" in t
        or "insufficient" in t
        or "cannot answer" in t
        or "no context" in t
    )

def _clean_refs(text: str) -> str:
    """Убираем ссылки [1], [2] и т.п., нормализуем пробелы."""
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()

def _extract_qa(fragment: str) -> tuple[str, str]:
    """Достаём (вопрос, ответ) из строки 'Вопрос: ...\nОтвет: ...'."""
    if not isinstance(fragment, str):
        return "", ""
    q = ""
    a = ""
    mq = re.search(r"Вопрос:\s*(.+?)(?:\nОтвет:|$)", fragment, flags=re.IGNORECASE | re.DOTALL)
    ma = re.search(r"Ответ:\s*(.+)$", fragment, flags=re.IGNORECASE | re.DOTALL)
    if mq:
        q = mq.group(1).strip()
    if ma:
        a = ma.group(1).strip()
    return (q or ""), (a or "")

def _trim_context(context: List[str], max_total: int, max_one: int) -> List[str]:
    """Режем контекст по лимитам."""
    safe, total = [], 0
    for c in context:
        frag = (c or "")[:max_one]
        if total + len(frag) > max_total:
            frag = frag[: max(0, max_total - total)]
        if frag:
            safe.append(frag)
            total += len(frag)
        if total >= max_total:
            break
    return safe

# ==== FALLBACK-ЛОГИКА ====

def _compose_refund_fallback(question: str, context: List[str]) -> Optional[str]:
    q_l = (question or "").lower()
    if not any(x in q_l for x in ["возврат", "refund", "вернуть"]):
        return None
    for frag in context:
        fq, fa = _extract_qa(frag)
        if any(w in (fq+fa).lower() for w in ["возврат", "вернуть", "refund"]):
            return f"Возврат средств осуществляется по правилам возврата товара. {fa}".strip()
    return "Возврат средств возможен по стандартным правилам возврата товара."

def _compose_exchange_fallback(question: str, context: List[str]) -> Optional[str]:
    q_l = (question or "").lower()
    if not any(x in q_l for x in ["обмен", "exchange"]):
        return None
    for frag in context:
        fq, fa = _extract_qa(frag)
        if any(w in (fq+fa).lower() for w in ["обмен", "exchange"]):
            return f"Обмен товара возможен при соблюдении условий: {fa}".strip()
    return "Обмен товара возможен при наличии чека и сохранности товара."

def _compose_not_delivered_fallback(question: str, context: List[str]) -> Optional[str]:
    q_l = (question or "").lower()
    if not any(x in q_l for x in ["не приш", "не получил", "пропал", "not delivered", "не дош"]):
        return None
    return "Если заказ не был доставлен, рекомендуем связаться с поддержкой и указать номер заказа."

def _compose_compare_fallback(question: str, context: List[str]) -> Optional[str]:
    q_l = (question or "").lower()
    if not any(x in q_l for x in ["быстрее", "дольше", "что быстрее", "faster", "slower"]):
        return None
    refund_ans, exch_ans = None, None
    for frag in context:
        fq, fa = _extract_qa(frag)
        low = (fq+fa).lower()
        if refund_ans is None and any(w in low for w in ["возврат", "refund"]):
            refund_ans = fa
        if exch_ans is None and any(w in low for w in ["обмен", "exchange"]):
            exch_ans = fa
    parts = ["В базе знаний нет прямого сравнения сроков обмена и возврата."]
    if refund_ans:
        parts.append(f"• Возврат: {refund_ans}")
    if exch_ans:
        parts.append(f"• Обмен: {exch_ans}")
    return " ".join(parts)

def _compose_generic_fallback(question: str, context: List[str]) -> str:
    return "К сожалению, в базе знаний нет прямого ответа на этот вопрос. Попробуйте переформулировать запрос или уточнить детали."

# ==== ОСНОВНОЙ ГЕНЕРАТОР ====

class Generator:
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None, timeout: Optional[int] = None):
        self.url = url or settings.genapi_url
        self.key = key or settings.genapi_key
        self.timeout = timeout if timeout is not None else settings.request_timeout_sec

    def ask(self, question: str, context: List[str]) -> str:
        # ограничиваем контекст
        safe_ctx = _trim_context(context, settings.max_context_chars, settings.max_fragment_chars)
        ctx_for_llm = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(safe_ctx))

        system_prompt = (
            "Ты — русскоязычный специалист поддержки. Отвечай понятно и дружелюбно, "
            "опираясь только на факты из базы знаний. Если данных нет, честно скажи об этом, "
            "и добавь полезные советы. Не вставляй ссылки в квадратных скобках."
        )

        user_prompt = f"Контекст:\n{ctx_for_llm}\n\nВопрос: {question}\nОтвет:"

        payload = {
            "is_sync": True,
            "temperature": 0.0,
            "top_p": 0.9,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
        }

        answer = self._call_genapi(payload)

        # если модель "сдалась" — включаем fallback
        if _looks_unknown(answer):
            for fb in (
                _compose_compare_fallback,
                _compose_refund_fallback,
                _compose_exchange_fallback,
                _compose_not_delivered_fallback,
            ):
                res = fb(question, safe_ctx)
                if res:
                    return _clean_refs(res)
            return _compose_generic_fallback(question, safe_ctx)

        return _clean_refs(answer)

    def _call_genapi(self, payload: dict) -> str:
        if not self.key:
            return "[GenAPI error] Missing GENAPI_KEY"
        try:
            resp = requests.post(
                self.url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.key}",
                },
                json=payload,
                timeout=self.timeout,
            )
        except Exception as e:
            return f"[GenAPI exception] {e}"

        if resp.status_code != 200:
            try:
                return f"[GenAPI HTTP {resp.status_code}] {resp.json()}"
            except Exception:
                return f"[GenAPI HTTP {resp.status_code}] {resp.text}"

        try:
            data = resp.json()
        except Exception as e:
            return f"[GenAPI parse error] {e}"

        # унифицированный парсинг
        if isinstance(data, dict):
            if "response" in data:
                r = data["response"]
                if isinstance(r, list) and r:
                    msg = r[0].get("message") or r[0].get("delta") or {}
                    return msg.get("content", "").strip()
            if "choices" in data:
                ch = data["choices"]
                if ch and "message" in ch[0]:
                    return ch[0]["message"].get("content", "").strip()
            for key in ("output", "text", "message"):
                if key in data and isinstance(data[key], str):
                    return data[key].strip()

        return f"[GenAPI unexpected] {data}"
