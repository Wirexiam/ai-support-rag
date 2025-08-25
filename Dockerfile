FROM python:3.10-slim

# рабочая директория внутри контейнера
WORKDIR /app

# устанавливаем зависимости
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# копируем проект
COPY . /app

# переменные окружения
ENV PYTHONUNBUFFERED=1 \
    DOCS_PATH=/app/data \
    DATABASE_URL=sqlite:////app/rag_logs.db

EXPOSE 8000

# запуск
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
