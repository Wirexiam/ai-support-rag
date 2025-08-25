import os
import pandas as pd, numpy as np, faiss, pickle
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi

CSV_PATH = os.getenv("FAQ_CSV_PATH", "data/faq.csv")  # по умолчанию рядом с проектом

# === 1. Загружаем данные
df = pd.read_csv(CSV_PATH)
records = df.to_dict(orient="records")

def make_doc(r):
    q = str(r.get("question_ru", "")).strip()
    a = str(r.get("answer_ru", "")).strip()
    return f"Вопрос: {q}\nОтвет: {a}"

corpus = [make_doc(r) for r in records]

# === 2. Эмбеддинги (BGE-M3, мультиязычная)
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
enc = model.encode(corpus, batch_size=32)
emb = np.array(enc["dense_vecs"]).astype("float32")
faiss.normalize_L2(emb)

# === 3. FAISS (inner product по L2-нормированным векторам)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

# === 4. BM25 по тем же текстам (вопрос+ответ)
def tokenize(text: str):
    text = "".join([c.lower() if (c.isalnum() or c.isspace()) else " " for c in text])
    return [t for t in text.split() if t]

bm25 = BM25Okapi([tokenize(doc) for doc in corpus])

# === 5. Сохранение артефактов
with open("faq_meta.pkl", "wb") as f:
    pickle.dump(records, f)

with open("faq_embeddings.pkl", "wb") as f:
    pickle.dump(emb, f)

faiss.write_index(index, "faq.index")

with open("bm25.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "corpus": corpus}, f)

print("✅ Индексация завершена: encoded (вопрос+ответ), FAISS + BM25 готовы")
