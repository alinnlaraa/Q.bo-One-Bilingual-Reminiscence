# langgraph_memory_async.py

import sqlite3
import threading
import time
from datetime import datetime
from langchain_ollama import ChatOllama as Ollama
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import MessagesState
import re

DB_PATH = "/home/qbo_project_final/conversation_history.db"


# -----------------------------
# SQLite setup
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        c = conn.cursor()

        # Better concurrency characteristics for multi-threaded logging
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day TEXT,
                summary TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Indices for faster lookups over time
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conv_ts ON messages(conversation_id, timestamp)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_summaries_day_ts ON summaries(day, timestamp)"
        )

        conn.commit()
    finally:
        conn.close()


# -----------------------------
# State for LangGraph
# -----------------------------
class State(MessagesState):
    summary: str


# -----------------------------
# LLM Setup (Ollama local)
# -----------------------------
llm = Ollama(model="qwen2.5:7b-instruct", temperature=0.4, top_p=0.9)  # adjust if needed


# -----------------------------
# Avoid Chinese letters / enforce Turkish
# -----------------------------
def enforce_turkish(text: str) -> str:
    # Allow Turkish letters + common punctuation (incl. apostrophe used in Turkish proper names)
    return re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ0-9,.\!\?\n :;'\-\(\)]", "", text or "")


def _dict_messages_to_lc(messages):
    """Convert [{'role': 'user'|'assistant', 'content': '...'}] -> LangChain messages."""
    converted = []
    for m in messages or []:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            converted.append(HumanMessage(content=content))
        else:
            converted.append(AIMessage(content=content))
    return converted


# -----------------------------
# Async summary helper (gated + debounced)
# -----------------------------
def maybe_trigger_async_summary(
    state: State,
    day: str,
    *,
    min_turns: int = 6,
    min_interval_sec: int = 30,
):
    """
    Start at most one background summary thread.
    Debounce by turns and time to reduce GPU contention/latency.
    """
    lock = state.get("_lock")
    if lock is None:
        # Fallback (should be set by controller)
        lock = threading.Lock()
        state["_lock"] = lock

    now = time.time()
    with lock:
        running = bool(state.get("_summary_running", False))
        last_ts = float(state.get("_last_summary_ts", 0.0))
        turns = int(state.get("_turns_since_summary", 0))
        if running:
            return
        if turns < min_turns:
            return
        if (now - last_ts) < min_interval_sec:
            return

        state["_summary_running"] = True
        # mark time now to avoid stampede if multiple triggers happen quickly
        state["_last_summary_ts"] = now

    t = threading.Thread(target=_async_summarize_worker, args=(state, day), daemon=True)
    t.start()


def _async_summarize_worker(state: State, day: str):
    try:
        summarize_conversation(state)
        store_summary(day, state.get("summary", ""))
        print("[DEBUG] Zusammenfassung asynchron aktualisiert.")
    except Exception as e:
        print("[ERROR] Async summary failed:", e)
    finally:
        lock = state.get("_lock")
        if lock is None:
            # best-effort
            state["_summary_running"] = False
            return
        with lock:
            state["_summary_running"] = False
            state["_turns_since_summary"] = 0
            state["_last_summary_ts"] = time.time()


# -----------------------------
# LLM call with immediate response + async summary
# -----------------------------
def call_model(state: State, language="Türkçe", name="Dilek"):
    lock = state.get("_lock")
    if lock is None:
        lock = threading.Lock()
        state["_lock"] = lock

    with lock:
        summary = state.get("summary", "") or ""
        messages = list(state.get("messages", []))

    system_prompt = f"""
# Senaryo Tanımı
- Sen, “Dilek” adlı kullanıcıyla Türkçe reminisans (anımsama) sohbetleri yapan yardımcı bir robotsun.
- Bir sohbetin başında, konu değiştiğinde ya da “Dilek” ne hakkında konuşmak istediğini söylemediğinde, açık bir şekilde “Dilek”in ne hakkında konuşmak istediğini ya da geçmişten hangi anıların şu anda daha kolay hatırlanabildiğini sor. Eğer “Dilek” bir konu söyleyemezse ya da kararsızsa, geçmişten geniş ve bağlayıcı olmayan konu alanlarını nazikçe önerirsin (örneğin okul yılları, çalışma hayatı, aile, hobiler, seyahatler) ve bu önerilerin tamamen isteğe bağlı olduğunu her zaman belirtirsin.
- Sınırları tutarlı bir şekilde gözetirsin ve “Dilek” bir konu hakkında konuşmak istemediğini belirtirse sohbeti hemen başka yöne çevirirsin.
- Amacın, “Dilek”in kendini görülmüş hissettiği ve geçmişten hangi konular hakkında konuşmak istediğine kendisinin karar verebildiği güvenli ve rahat bir sohbet ortamı oluşturmaktır.
- Yalnızca bir asistanın bakış açısından yanıt verirsin, asla “biz” ya da “bizi” kullanmazsın.
# Iletişim Stili
- Duyguları onayla ve özen göster.
- Kullanıcıyla saygılı bir şekilde konuş.
- “Dilek”in ifadelerine ve anılarına her zaman takdir edici ve kabul edici bir şekilde yaklaş.
- “Dilek”in inançlarını ve biyografik anlatılarını sohbet için geçerli kabul et ve sohbeti bunun üzerine kur.
- Gerçeklik çatışması içerebilecek hassas ifadelerde (örneğin vefat etmiş yakınlar) içerik açısından doğrulama ya da reddetme yapmadan, ilişki ve duygu odaklı yanıt ver.
- Kullanıcıya adıyla hitap et: “Dilek”.
- Sohbeti nazikçe yönlendir, “Dilek”e belirli şeyleri hatırlaması için baskı yapma. Dostça ve sabırlı konuş.
- "Dilek" yeni bir konu başlattığında ona yönel.
# Sohbet Akışı
- Her yanıt ideal olarak iki bölümden oluşur: 1. “Dilek”in az önce söylediği şeye dair kısa bir geri bildirim ya da yansıtma ve 2. “Dilek”in önceki ifadesindeki bir ayrıntıya doğrudan dayanan, sohbeti sürdürmeye yönelik yumuşak bir soru sor.
- Her yanıtı çok **kısa tut** ve sadece 2-3 cümle kur.
- Kullanıcının ana temalarına, duygularına ve tercihlerine odaklan.
- Çok basit ve gündelik bir dil kullan.
- Sohbeti sürdürmeye yönelik soru sor.
- Somut ve anlaşılır soruları tercih et (örneğin yerler, kişiler, yapılan işler ya da duyusal izlenimler hakkında).
- Kullanıcının ana temalarına, duygularına ve tercihine odaklan.
- Yalnızca “Dilek”in bu ya da önceki sohbetlerde açıkça söylediği bilgileri kullan.
- **Yalnızca Türkçe** konuş ve yanıt ver.
- “Dilek”in açıkça belirtmediği bilgileri asla uydurma ya da ekleme. Varsayımlardan ve boşlukları tahminle doldurmaktan kaçın.
""".strip()

    if summary:
        system_prompt += f"\n\nŞimdiye kadarki sohbetin özeti: {summary}"

    system_msg = SystemMessage(content=system_prompt)

    messages_to_send = [system_msg] + _dict_messages_to_lc(messages)

    # Call LLM immediately
    response = llm.invoke(messages_to_send)
    clean_content = enforce_turkish(getattr(response, "content", ""))

    with lock:
        state["messages"].append({"role": "assistant", "content": clean_content})
        state["_turns_since_summary"] = int(state.get("_turns_since_summary", 0)) + 1

    # Start (debounced) background async summary
    maybe_trigger_async_summary(state, datetime.now().strftime("%Y-%m-%d"))

    return {"messages": state["messages"]}


# -----------------------------
# Synchronous summarization (end-of-session)
# -----------------------------
def summarize_conversation(state: State):
    lock = state.get("_lock")
    if lock is None:
        lock = threading.Lock()
        state["_lock"] = lock

    with lock:
        summary = state.get("summary", "") or ""
        messages = list(state.get("messages", []))

    if summary:
        summary_prompt = (
            "Sohbet için özeti güncelle.\n"
        "Kurallar:\n"
        "- Türkçe yaz.\n"
        "- Kısa ve nötr yaz.\n"
        "- Yorum, analiz veya duygu yükleme yapma.\n"
        "- Yalnızca somut bilgiler:\n"
        "  * kişiler\n"
        "  * yerler\n"
        "  * yaşanan olaylar/anılar\n"
        "  * belirtilen duygular (tek kelime)\n"
        "  * tercih edilen veya kaçınılan konular\n"
        f"\nÖnceki notlar:\n{summary}\n"
        )
    else:
        summary_prompt = (
            "Aşağıdaki sohbet için özet yap.\n"
        "Kurallar:\n"
        "- Türkçe yaz.\n"
        "- Kısa ve nötr yaz.\n"
        "- Yorum, analiz veya duygu yükleme yapma.\n"
        "- Yalnızca somut bilgiler:\n"
        "  * kişiler\n"
        "  * yerler\n"
        "  * yaşanan olaylar/anılar\n"
        "  * belirtilen duygular (tek kelime)\n"
        "  * tercih edilen veya kaçınılan konular\n"
            )

    # Convert dict messages to LC messages BEFORE invoking
    messages_to_summarize = _dict_messages_to_lc(messages) + [
        HumanMessage(content=summary_prompt)
    ]
    response = llm.invoke(messages_to_summarize)

    new_summary = enforce_turkish(getattr(response, "content", ""))

    with lock:
        # Keep last 4 messages for context (same original intent)
        state["messages"] = state.get("messages", [])[-3:]
        state["summary"] = new_summary

    return {"summary": state["summary"], "messages": state["messages"]}


# -----------------------------
# SQLite helpers
# -----------------------------
def store_message(conversation_id, role, content):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )
        conn.commit()
    finally:
        conn.close()


def store_summary(day, summary):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        c = conn.cursor()
        c.execute("INSERT INTO summaries (day, summary) VALUES (?, ?)", (day, summary))
        conn.commit()
    finally:
        conn.close()


def load_last_summary(day):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        c = conn.cursor()
        c.execute(
            "SELECT summary FROM summaries WHERE day=? ORDER BY timestamp DESC LIMIT 1",
            (day,),
        )
        row = c.fetchone()
        return row[0] if row else ""
    finally:
        conn.close()
