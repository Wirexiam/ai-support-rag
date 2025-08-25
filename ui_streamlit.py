import os
import time
import html
import requests
import streamlit as st

# === Настройки ===
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000/ask")
APP_TITLE = "🤖 AI Support RAG"
APP_DESC = "Задай вопрос — получи ответ от RAG (FAISS+BM25) + GPT-4 Omni (GenAPI), с показом контекста."

# === Внешний вид ===
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.markdown(
    """
<style>
:root {
  --pri: #4F46E5;
  --radius: 12px;
}

/* базовая геометрия */
.block-container { padding-top: 2rem; padding-bottom: 1rem; }
.small { font-size: 0.85rem; opacity: 0.85; }

/* LIGHT THEME */
@media (prefers-color-scheme: light) {
  .answer { background: #f8fafc; border: 1px solid #e5e7eb; color: #0f172a; border-radius: var(--radius); padding: 14px; }
  .ctx    { background: #f9fafb; border: 1px solid #e5e7eb; color: #111827; border-radius: 10px; padding: 12px; }
  .divider { border-top: 1px solid #e5e7eb; margin: 8px 0 16px 0; }
  button[kind="primary"] { background: var(--pri); }
}

/* DARK THEME */
@media (prefers-color-scheme: dark) {
  .answer { background: #0b1220; border: 1px solid #1f2937; color: #e5e7eb; border-radius: var(--radius); padding: 14px; }
  .ctx    { background: #0f172a; border: 1px solid #243244; color: #e5e7eb; border-radius: 10px; padding: 12px; }
  .divider { border-top: 1px solid #1f2937; margin: 8px 0 16px 0; }
  button[kind="primary"] { background: var(--pri); }
}

/* инпут покрупнее */
.stTextInput > div > div > input { font-size: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# === Сайдбар ===
with st.sidebar:
    st.markdown(f"### {APP_TITLE}")
    st.write(APP_DESC)
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    st.markdown("---")
    st.markdown("**Подсказки:**")
    st.markdown("- Убедись, что FastAPI запущен: `uvicorn app.main:app --reload`")
    st.markdown("- Индекс собран: `python indexer.py`")
    st.markdown("- Вопрос можно задавать на любом языке — бэкенд разберётся.")
    st.markdown("---")
    clear_btn = st.button("Очистить историю")

# === Состояние ===
if "history" not in st.session_state:
    # элементы: (question, answer_html, context_list, latency_sec, ok)
    st.session_state.history = []

if clear_btn:
    st.session_state.history.clear()

# === Заголовок ===
st.title(APP_TITLE)
st.caption("Поддержка на базе retrieval-augmented generation. В ответе показываем и сгенерированный текст, и источники (фрагменты из базы).")

# === Форма ввода ===
col_inp, col_btn = st.columns([4, 1])
with col_inp:
    user_q = st.text_input("Ваш вопрос:", placeholder="Например: Как оформить возврат средств?")
with col_btn:
    ask_clicked = st.button("Спросить", type="primary", use_container_width=True)

def call_api(question: str, url: str) -> dict:
    """POST /ask FastAPI. Возвращает dict с answer/context либо ошибку."""
    try:
        t0 = time.time()
        r = requests.post(url, json={"question": question}, timeout=60)
        latency = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            data["_ok"] = True
            data["_latency"] = latency
            return data
        return {"_ok": False, "_latency": latency, "error": f"HTTP {r.status_code}: {r.text}"}
    except Exception as e:
        return {"_ok": False, "_latency": 0.0, "error": str(e)}

def as_html_with_br(text: str) -> str:
    """Экранируем HTML и сохраняем переводы строк как <br/>."""
    safe = html.escape(text).replace("\n", "<br/>")
    return safe

def html_to_plain(a_html: str) -> str:
    """Преобразует нашу HTML-строку (escape + <br/>) обратно в обычный текст для st.error()."""
    return html.unescape(a_html.replace("<br/>", "\n"))

# === Обработка запроса ===
if ask_clicked and user_q.strip():
    resp = call_api(user_q.strip(), api_url)
    if resp.get("_ok"):
        st.session_state.history.append((
            user_q.strip(),
            as_html_with_br(resp.get("answer", "")),
            resp.get("context", []),
            resp.get("_latency", 0.0),
            True
        ))
    else:
        st.session_state.history.append((
            user_q.strip(),
            as_html_with_br(f"Ошибка: {resp.get('error','unknown')}"),
            [],
            resp.get("_latency", 0.0),
            False
        ))

# === Рендер истории (последние сверху) ===
for q, a_html, ctx_list, lat, ok in reversed(st.session_state.history):
    with st.container():
        st.markdown(f"**❓ Вопрос:** {html.escape(q)}")
        if ok:
            st.markdown(f"""<div class="answer"><b>✅ Ответ:</b><br/>{a_html}</div>""", unsafe_allow_html=True)
            with st.expander("📚 Показать контекст (фрагменты из базы знаний)"):
                if ctx_list:
                    for i, ctx in enumerate(ctx_list, start=1):
                        ctx_html = html.escape(ctx).replace("\n", "<br/>")
                        st.markdown(f"**Источник [{i}]**")
                        st.markdown(f"<div class='ctx'>{ctx_html}</div>", unsafe_allow_html=True)
                        st.markdown("<span class='small'>Совпадение извлечено FAISS/BM25</span>",
                                    unsafe_allow_html=True)
                        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                else:
                    st.info("Контекст не возвращён (возможно, индекс пуст или запрос нерелевантен).")

            st.caption(f"⏱ Время ответа: {lat:.2f} c • API: {api_url}")
        else:
            plain_err = html_to_plain(a_html)
            st.error(plain_err)
            st.caption(f"⏱ Попытка запроса заняла: {lat:.2f} c • API: {api_url}")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
