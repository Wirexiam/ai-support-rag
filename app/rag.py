import faiss, pickle, numpy as np
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel

class Retriever:
    def __init__(self, index_path: str, meta_path: str, bm25_path: str, alpha: float = 0.6, faiss_k: int = 50):
        # Индексы/метаданные
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)
        with open(bm25_path, "rb") as f:
            pack = pickle.load(f)
        self.bm25: BM25Okapi = pack["bm25"]
        self.corpus = pack["corpus"]  # тексты "Вопрос:\nОтвет:\n" в том же порядке, что и meta
        # Модель энкодера
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        # Гиперпараметры гибридного скора
        self.alpha = float(alpha)  # вес FAISS
        self.faiss_k = int(faiss_k)

    def _encode(self, text: str) -> np.ndarray:
        v = self.model.encode([text])["dense_vecs"].astype("float32")
        faiss.normalize_L2(v)
        return v

    @staticmethod
    def _tokenize(text: str):
        text = "".join([c.lower() if (c.isalnum() or c.isspace()) else " " for c in text])
        return [t for t in text.split() if t]

    def search(self, query_ru: str, k: int = 3):
        # 1) FAISS
        qvec = self._encode(query_ru)
        sims, ids = self.index.search(qvec, self.faiss_k)  # побольше кандидатов
        sims = sims[0]
        ids = ids[0]

        # нормализация FAISS-скор
        faiss_max = float(np.max(sims)) if sims.size else 1.0
        faiss_scores = {int(i): (float(s)/faiss_max if faiss_max > 0 else 0.0)
                        for i, s in zip(ids, sims)}

        # 2) BM25
        bm25_scores_arr = self.bm25.get_scores(self._tokenize(query_ru))
        # топ-N BM25 (берём такое же N, как faiss_k)
        bm25_top_idx = np.argsort(bm25_scores_arr)[::-1][:self.faiss_k]
        bm25_max = float(np.max(bm25_scores_arr)) if bm25_scores_arr.size else 1.0
        bm25_scores = {int(i): (float(bm25_scores_arr[i])/bm25_max if bm25_max > 0 else 0.0)
                       for i in bm25_top_idx}

        # 3) Смешиваем
        all_ids = set(list(faiss_scores.keys()) + list(bm25_scores.keys()))
        mixed = []
        for doc_id in all_ids:
            s_f = faiss_scores.get(doc_id, 0.0)
            s_b = bm25_scores.get(doc_id, 0.0)
            score = self.alpha * s_f + (1.0 - self.alpha) * s_b
            mixed.append((score, doc_id))

        mixed.sort(key=lambda x: x[0], reverse=True)
        top = [doc_id for _, doc_id in mixed[:k]]

        # Возвращаем метаданные (records) в порядке убывания смешанного скора
        return [self.meta[i] for i in top]
