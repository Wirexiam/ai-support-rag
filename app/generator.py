# app/generator.py
from __future__ import annotations

import json
import re
import time
from typing import List, Optional

import requests

from .config import settings


# служебные утилиты #

def _looks_unknown(s: str) -> bool:
    """Эвристика: модель «сдалась» или ответ слишком пустой/общий."""
    if not s:
        return True
    t = s.strip().lower()
    return (
        len(t) < 15
        or "я не знаю" in t
        or "не знаю" in t
        or "insufficient" in t
        or "no context" in t
        or "cannot answer" in t
    )


def _clean_refs(text: str) -> str:
    """Удаляем ссылки вида [1], [ 2 ], [1,2] и т.п. + нормализуем пробелы."""
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def _extract_qa(fragment: str) -> tuple[str, str]:
    """
    На входе фрагмент вида:
      "Вопрос: ...\nОтвет: ..."
    Возвращает (question, answer). Если не нашли — пустые строки.
    """
    if not isinstance(fragment, str):
        return "", ""
    q = ""
    a = ""
    # НЕЖАДНЫЕ матчи и явные маркеры
    mq = re.search(r"Вопрос:\s*(.+?)(?:\nОтвет:|$)", fragment, flags=re.IGNORECASE | re.DOTALL)
    ma = re.search(r"Ответ:\s*(.+)$", fragment, flags=re.IGNORECASE | re.DOTALL)
    if mq:
        q = mq.group(1).strip()
    if ma:
        a = ma.group(1).strip()
    return (q or "").strip(), (a or "").strip()


def _trim_context(context: List[str], max_total: int, max_one: int) -> List[str]:
    """
    Режем каждый фрагмент и общий контекст по лимитам из конфигов,
    чтобы не раздувать промпт.
    """
    safe = []
    total = 0
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


# безопасные фолбэки #
def _compose_refund_fallback(question: str, context: List[str]) -> Optional[str]:
    q_l = (question or "").lower()
    if "возврат" not in q_l and "refund" not in q_l:
        return None
    refund_text = exchange_text = not_deliv_text = None
    for frag in context:
        fq, fa = _extract_qa(frag)
        f_all = (fq + "\n" + fa).lower()
        if refund_text is None and any(x in f_all for x in ["возврат", "вернуть", "вернете", "вернуть товар", "refund"]):
            refund_text = fa if fa else frag
        if exchange_text is None and any(x in f_all for x in ["обмен", "обменять", "exchange"]):
            exchange_text = fa if fa else frag
        if not_deliv_text is None and any(x in f_all for x in ["не приш", "неполуч", "пропал заказ", "not delivered", "didn't arrive"]):
            not_deliv_text = fa if fa else frag
    if refund_text:
        parts = [
            "Возврат средств осуществляется по правилам возврата товара.",
            refund_text.rstrip(".") + "."
        ]
        if exchange_text:
            parts.append("Обмен возможен при соблюдении условий: " + exchange_text.rstrip(".") + ".")
        if not_deliv_text and any(w in q_l for w in ["заказ", "достав", "order", "delivery"]):
            parts.append("Если заказ не пришёл — действуйте так: " + not_deliv_text.rstrip(".") + ".")
        return " ".join(parts).strip()
    return None


def _compose_exchange_fallback(question: str, context: List[str]) -> Optional[str]:
    q_l = (question or "").lower()
    if "обмен" not in q_l and "exchange" not in q_l:
        return None
    for frag in context:
        _, fa = _extract_qa(frag)
        f_all = (frag or "").lower()
        if any(x in f_all for x in ["обмен", "exchange"]):
            text = fa if fa else frag
            return ("Обмен возможен при соблюдении условий: " + text.rstrip(".") + ".").strip()
    return None


def _compose_not_delivered_fallback(question: str, context: List[str]) -> Optional[str]:
    q_l = (question or "").lower()
    if not any(x in q_l for x in ["не приш", "не получил", "пропал", "not delivered", "didn't arrive", "не дош"]):
        return None
    for frag in context:
        _, fa = _extract_qa(frag)
        f_all = (frag or "").lower()
        if any(x in f_all for x in ["не приш", "свяжитесь с поддержкой", "номер заказа", "not delivered", "support"]):
            text = fa if fa else frag
            return ("Если заказ не пришёл — свяжитесь с поддержкой и укажите номер заказа: " + text.rstrip(".") + ".").strip()
    return None


def _compose_compare_fallback(question: str, context: List[str]) -> Optional[str]:
    q_l = (question or "").lower()
    if not any(x in q_l for x in ["быстрее", "дольше", "что быстрее", "faster", "slower"]):
        return None
    refund_text = exchange_text = None
    for frag in context:
        fq, fa = _extract_qa(frag)
        f_all = (fq + "\n" + fa).lower()
        if refund_text is None and any(x in f_all for x in ["возврат", "вернуть", "refund"]):
            refund_text = fa if fa else frag
        if exchange_text is None and any(x in f_all for x in ["обмен", "exchange"]):
            exchange_text = fa if fa else frag
    if refund_text or exchange_text:
        parts = ["В базе нет данных о скорости («быстрее/дольше»)."]
        if refund_text:
            parts.append("Возврат: " + refund_text.rstrip(".") + ".")
        if exchange_text:
            parts.append("Обмен: " + exchange_text.rstrip(".") + ".")
        return " ".join(parts).strip()
    return None


# основной генератор #

class Generator:
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None, timeout: Optional[int] = None):
        self.url = url or settings.genapi_url
        self.key = key or settings.genapi_key
        self.timeout = timeout if timeout is not None else settings.request_timeout_sec

    def ask(self, question: str, context: List[str]) -> str:
        """
        1) Зовём LLM с жёстким system-промптом.
        2) Если ответ пустой/«я не знаю» — включаем фолбэки.
        3) Чистим следы ссылок вида [2] и нормализуем пробелы.
        """
        # Применяем лимиты по конфигу
        safe_ctx = _trim_context(
            context=context,
            max_total=settings.max_context_chars,
            max_one=settings.max_fragment_chars,
        )

        # Нумерованный контекст: модели видят «слоты», но запрещаем вставлять ссылки в итоговый ответ.
        ctx_for_llm = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(safe_ctx))

        system_prompt = (
            "Ты — русскоязычный специалист поддержки. Отвечай коротко и предметно, "
            "используя ТОЛЬКО факты из контекста. Если формулировка вопроса отличается, "
            "но в контексте есть близкие правила (например, «возврат средств» ↔ «возврат товара»), "
            "применяй их и прямо говори, что возврат средств происходит по правилам возврата товара. "
            "Не вставляй в текст ответа ссылки/цитаты в квадратных скобках ([1], [2] и т. п.) — выводи чистый текст. "
            "Если в контексте нет информации — отвечай: «Я не знаю.» "
            "Не делай сравнений «быстрее/дольше/сроки», если в контексте нет явных данных о времени."
        )

        user_prompt = (
            f"Контекст:\n{ctx_for_llm}\n\n"
            f"Вопрос пользователя: {question}\n"
            "Ответ (без ссылок в квадратных скобках):"
        )

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

        # Если LLM «сдалась» — безопасные фолбэки
        if _looks_unknown(answer):
            for fb_fn in (
                _compose_compare_fallback,
                _compose_refund_fallback,
                _compose_exchange_fallback,
                _compose_not_delivered_fallback,
            ):
                fb = fb_fn(question, safe_ctx)
                if fb:
                    return _clean_refs(fb)

        return _clean_refs(answer)

    # низкоуровневый вызов GenAPI #
    def _call_genapi(self, payload: dict) -> str:
        # 0) Проверка ключа — чтобы не посылать Bearer None
        if not self.key:
            return "[GenAPI error] Missing GENAPI_KEY (set env var)"

        # 1) Ретраи на 429/5xx
        attempts = 3
        backoff = 0.75  # сек
        last_err = None

        for attempt in range(1, attempts + 1):
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
                last_err = f"[GenAPI exception] {e}"
                if attempt < attempts:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return last_err

            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"[GenAPI HTTP {resp.status_code}] {resp.text}"
                if attempt < attempts:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                # упали после ретраев — вернём последнее
                return last_err

            if resp.status_code != 200:
                # Попробуем показать полезную ошибку
                try:
                    data = resp.json()
                    return f"[GenAPI HTTP {resp.status_code}] {json.dumps(data, ensure_ascii=False)}"
                except Exception:
                    return f"[GenAPI HTTP {resp.status_code}] {resp.text}"

            # Унифицированный парсинг под разные форматы
            try:
                data = resp.json()
            except Exception as e:
                return f"[GenAPI parse error] {e}"

            # 1) Новый формат GenAPI: {"response":[{"message":{"content":"..."}}]} или delta-ветка
            if isinstance(data, dict) and "response" in data:
                r = data["response"]
                if isinstance(r, list) and r:
                    msg = r[0]
                    if isinstance(msg, dict):
                        m = msg.get("message") or msg.get("delta") or {}
                        content = m.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()

            # 2) OpenAI-подобный: {"choices":[{"message":{"content":"..."}}]}
            if "choices" in data:
                ch = data["choices"]
                if isinstance(ch, list) and ch:
                    msg = ch[0].get("message", {}) or {}
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

            # 3) Простые ключи: {"output":"..."}, {"text":"..."}, {"message":"..."}
            for key in ("output", "text", "message"):
                if key in data and isinstance(data[key], str) and data[key].strip():
                    return data[key].strip()

            # если дошли сюда — формат неожиданный, вернём диагностическую инфу
            return f"[GenAPI unexpected] {json.dumps(data, ensure_ascii=False)}"

        # теоретически не дойдём сюда, но на всякий случай
        return last_err or "[GenAPI] Unknown error"
