# AI Support RAG

Retrieval-Augmented Generation для справки/FAQ: гибридный поиск (FAISS + BM25) + генерация ответа. В интерфейсе показываются **и ответ**, и **источники** (фрагменты из базы).

---

## Возможности

- Гибридный поиск: dense + sparse (FAISS + BM25)
- Безопасные фолбэки на ключевые запросы («обмен», «возврат», «не пришёл», «быстрее/дольше»)
- Очистка ссылок `[1], [2]` и нормализация текста
- UI на **Streamlit**
- API на **FastAPI**
- Автотесты на **pytest** с сохранением результатов в CSV/JSONL

---

## Требования

- Python **3.10**
- pip или poetry
- (опционально) Docker / Docker Compose
- Ключ GenAPI (необязателен для тестов, нужен для реальных запросов)

---

## Установка

```bash
git clone <YOUR_REPO_URL>
cd rag_support

pip install -r requirements.txt

cp .env.example .env
# в .env укажи GENAPI_KEY=... и путь к data/faq.csv
```

---

## Индексация базы

```bash
python indexer.py --csv data/faq.csv
```

После этого появятся файлы `faq.index`, `faq_meta.pkl`, `bm25.pkl`.

---

## Запуск

### API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)

### UI
```bash
streamlit run ui_streamlit.py
```

UI: [http://localhost:8501](http://localhost:8501)

---

## Пример запроса

```bash
curl -s -X POST http://localhost:8000/ask   -H "Content-Type: application/json"   -d '{"question":"Как оформить возврат средств?"}'
```

Ответ:

```json
{
  "answer": "Возврат средств происходит по правилам возврата товара...",
  "context": ["Вопрос: ...\nОтвет: ..."],
  "latency_sec": 3.27
}
```

---

## Тесты

```bash
pytest
```

Результаты сохраняются в:

- `results/test_results.jsonl`
- `results/test_results.csv`

---

## Docker

```bash
docker build -t ai-support-rag .
docker run -p 8000:8000 --env-file .env ai-support-rag
```

---

## Лицензия

MIT
