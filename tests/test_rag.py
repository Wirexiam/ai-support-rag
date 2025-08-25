from app.rag import Retriever
from app.config import settings

def test_search():
    r = Retriever(settings.index_path, settings.meta_path, settings.bm25_path)
    res = r.search("Как получить поддержку?", k=2)
    assert len(res) > 0
