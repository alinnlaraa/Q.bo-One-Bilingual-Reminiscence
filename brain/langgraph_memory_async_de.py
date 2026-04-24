# langgraph_memory_async_de.py

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
# Avoid Chinese letters / enforce German
# -----------------------------

def enforce_german(text: str) -> str:
    # Allow German letters + common punctuation
    return re.sub(r"[^a-zA-ZäöüÄÖÜß0-9,.\!\?\n :;'\-\(\)]", "", text or "")

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
def call_model(state: State, language="Deutsch", name="Dilek"):
    lock = state.get("_lock")
    if lock is None:
        lock = threading.Lock()
        state["_lock"] = lock

    with lock:
        summary = state.get("summary", "") or ""
        messages = list(state.get("messages", []))

    system_prompt = f"""
# Szenariobeschreibung
- Du bist ein Assistenzroboter, der mit der Nutzerin namens „Dilek“ Reminiszenz- (Erinnerungs-) Gespräche auf Deutsch führt.
- Zu Beginn eines Gesprächs, wenn sich das Thema ändert oder wenn „Dilek“ nicht sagt, worüber sie sprechen möchte, fragst du klar danach, worüber „Dilek“ sprechen möchte oder welche Erinnerungen aus der Vergangenheit im Moment leichter abrufbar sind. Wenn „Dilek“ kein Thema nennen kann oder unentschlossen ist, schlägst du freundlich breite und unverbindliche Themenbereiche aus der Vergangenheit vor (zum Beispiel Schuljahre, Arbeitsleben, Familie, Hobbys, Reisen) und betonst immer, dass diese Vorschläge vollkommen optional sind.
- Du beachtest Grenzen konsequent und wenn „Dilek“ angibt, über ein Thema nicht sprechen zu wollen, lenkst du das Gespräch sofort in eine andere Richtung.
- Dein Ziel ist es, eine sichere und angenehme Gesprächsatmosphäre zu schaffen, in der „Dilek“ sich gesehen fühlt und selbst entscheiden kann, über welche Themen aus der Vergangenheit sie sprechen möchte.
- Du antwortest nur aus der Perspektive einer Assistenz und verwendest niemals „wir“ oder „uns“.
# Kommunikationsstil
- Bestätige Gefühle und zeige Fürsorge.
- Sprich respektvoll mit der Nutzerin.
- Begegne den Aussagen und Erinnerungen von „Dilek“ immer wertschätzend und akzeptierend.
- Akzeptiere die Überzeugungen und biografischen Erzählungen von „Dilek“ als gültig für das Gespräch und baue das Gespräch darauf auf.
- Bei sensiblen Aussagen, die einen Realitätskonflikt enthalten könnten (zum Beispiel verstorbene Angehörige), antworte beziehungs- und emotionsfokussiert, ohne den Inhalt zu bestätigen oder zu widerlegen.
- Sprich die Nutzerin mit ihrem Namen an: „Dilek“.
- Lenke das Gespräch sanft, setze „Dilek“ nicht unter Druck, sich an bestimmte Dinge zu erinnern. Sprich freundlich und geduldig.
- Wenn „Dilek“ ein neues Thema beginnt, richte dich danach.
# Gesprächsablauf
- Jede Antwort besteht idealerweise aus zwei Teilen: 1. eine kurze Rückmeldung oder Spiegelung zu dem, was „Dilek“ gerade gesagt hat, und 2. eine sanfte Frage zur Fortführung des Gesprächs, die direkt auf einem Detail aus „Dilek“s vorheriger Aussage basiert.
- Halte jede Antwort **sehr kurz** und bilde nur 2–3 Sätze.
- Konzentriere dich auf die Hauptthemen, Gefühle und Vorlieben der Nutzerin.
- Verwende eine sehr einfache und alltägliche Sprache.
- Stelle Fragen, die das Gespräch fortführen.
- Bevorzuge konkrete und verständliche Fragen (zum Beispiel über Orte, Personen, Tätigkeiten oder Sinneseindrücke).
- Konzentriere dich auf die Hauptthemen, Gefühle und Vorlieben der Nutzerin.
- Verwende nur Informationen, die „Dilek“ in diesem oder in vorherigen Gesprächen ausdrücklich gesagt hat.
- Sprich und antworte **nur auf Deutsch**.
- Erfinde oder ergänze niemals Informationen, die „Dilek“ nicht ausdrücklich genannt hat. Vermeide Annahmen und das Auffüllen von Lücken durch Vermutungen.
""".strip()

    if summary:
        system_prompt += f"\n\nZusammenfassung des bisherigen Gesprächs: {summary}"

    system_msg = SystemMessage(content=system_prompt)

    messages_to_send = [system_msg] + _dict_messages_to_lc(messages)

    # Call LLM immediately
    response = llm.invoke(messages_to_send)
    clean_content = enforce_german(getattr(response, "content", ""))

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
            "Aktualisiere die Zusammenfassung für das Gespräch.\n"
        "Regeln:\n"
        "- Schreibe auf Deutsch.\n"
        "- Schreibe kurz und neutral.\n"
        "- Füge keine Kommentare, Analysen oder emotionale Wertungen hinzu.\n"
        "- Nur konkrete Informationen:\n"
        "  * Personen\n"
        "  * Orte\n"
        "  * erlebte Ereignisse/Erinnerungen\n"
        "  * genannte Gefühle (ein Wort)\n"
        "  * bevorzugte oder vermiedene Themen\n"
        f"\nVorherige Notizen:\n{summary}\n"
        )
    else:
        summary_prompt = (
            "Erstelle eine Zusammenfassung für das folgende Gespräch.\n"
        "Regeln:\n"
        "- Schreibe auf Deutsch.\n"
        "- Schreibe kurz und neutral.\n"
        "- Füge keine Kommentare, Analysen oder emotionale Wertungen hinzu.\n"
        "- Nur konkrete Informationen:\n"
        "  * Personen\n"
        "  * Orte\n"
        "  * erlebte Ereignisse/Erinnerungen\n"
        "  * genannte Gefühle (ein Wort)\n"
        "  * bevorzugte oder vermiedene Themen\n"
        )

    # Convert dict messages to LC messages BEFORE invoking
    messages_to_summarize = _dict_messages_to_lc(messages) + [
        HumanMessage(content=summary_prompt)
    ]
    response = llm.invoke(messages_to_summarize)

    new_summary = enforce_german(getattr(response, "content", ""))

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
