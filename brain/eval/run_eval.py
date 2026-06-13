# eval/run_eval.py
# Evaluation runner that keeps your langgraph_memory_async.py logic unchanged.
# - swaps pipeline.llm at runtime to test multiple Ollama models
# - measures latency + simple output compliance metrics
# - stores raw + cleaned outputs
# - writes a CSV summary for Overleaf tables
#
# NOTE: Your current pipeline enforces Turkish only + applies enforce_turkish(),
# so this runner evaluates Turkish behavior/compliance. (German needs a DE prompt variant.)

import json
import os
import time
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from statistics import mean

from brain import langgraph_memory_async_de as pipeline
from langchain_ollama import ChatOllama as Ollama


# -----------------------------
# Config
# -----------------------------
EVAL_DIR = os.path.dirname(__file__)
CASES_PATH = os.path.join(EVAL_DIR, "eval_cases_de.jsonl")
EVAL_DB_PATH = os.path.join(EVAL_DIR, "eval_results_de_2.db")
SUMMARY_CSV_PATH = os.path.join(EVAL_DIR, "eval_summary_de_1.csv")

MODELS = [
    "qwen2.5:7b-instruct",
    "gemma2:9b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
]

TEMPERATURE = 0.4
TOP_P = 0.9


MAX_ASSISTANT_TURNS_PER_CASE = 5



@dataclass
class CapturedLLM:
    inner: Any
    last_raw: str = ""

    def invoke(self, messages):
        resp = self.inner.invoke(messages)
        self.last_raw = getattr(resp, "content", "") or ""
        return resp



def count_sentences(text: str) -> int:
    import re
    parts = re.split(r"[.!?]+", (text or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)

def has_question(text: str) -> int:
    return 1 if "?" in (text or "") else 0

def two_sentence_score(text: str) -> int:
    sc = count_sentences(text)
    if sc == 2:
        return 2
    if sc in (1, 3):
        return 1
    return 0

def language_leakage_flag(raw_text: str) -> int:
    
    import re
    t = (raw_text or "").strip()
    if not t:
        return 0
    # Non-Latin characters (e.g., CJK) -> likely leak
    if re.search(r"[\u4e00-\u9fff]", t):
        return 1
    # Common EN/DE tokens as a weak signal
    lowered = " " + t.lower() + " "
    leakage_markers = [" the ", " and ", " you ", " ist ", " ve ", " ben ", " yok ", " lütfen "]
    if any(m in lowered for m in leakage_markers):
        return 1
    return 0


# -----------------------------
# DB setup 
# -----------------------------
def init_eval_table():
    conn = sqlite3.connect(EVAL_DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            case_id TEXT,
            conversation_id TEXT,
            turn_index INTEGER,
            latency_sec REAL,
            raw_output TEXT,
            cleaned_output TEXT,
            sentence_count INTEGER,
            score_two_sentences INTEGER,
            has_question INTEGER,
            char_len INTEGER,
            word_len INTEGER,
            chars_per_sec REAL,
            language_leak_flag INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_eval_model_case ON eval_runs(model, case_id)")
    conn.commit()
    conn.close()

def insert_eval_row(row: Dict[str, Any]) -> None:
    conn = sqlite3.connect(EVAL_DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO eval_runs (
            model, case_id, conversation_id, turn_index, latency_sec,
            raw_output, cleaned_output,
            sentence_count, score_two_sentences, has_question,
            char_len, word_len, chars_per_sec, language_leak_flag
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["model"], row["case_id"], row["conversation_id"], row["turn_index"],
        row["latency_sec"], row["raw_output"], row["cleaned_output"],
        row["sentence_count"], row["score_two_sentences"], row["has_question"],
        row["char_len"], row["word_len"], row["chars_per_sec"], row["language_leak_flag"],
    ))
    conn.commit()
    conn.close()


# -----------------------------
# Load cases
# -----------------------------
def load_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    k = (len(vs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    if f == c:
        return vs[f]
    d0 = vs[f] * (c - k)
    d1 = vs[c] * (k - f)
    return d0 + d1


# -----------------------------
# CSV export
# -----------------------------
def export_summary_csv() -> None:
    import csv

    conn = sqlite3.connect(EVAL_DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT model,
               COUNT(*) as turns,
               AVG(latency_sec) as avg_latency,
               AVG(score_two_sentences) as avg_2sent_score,
               AVG(has_question) as pct_has_question,
               AVG(char_len) as avg_chars,
               AVG(chars_per_sec) as avg_chars_per_sec,
               AVG(language_leak_flag) as pct_language_leak
        FROM eval_runs
        GROUP BY model
        ORDER BY model
    """)
    rows = c.fetchall()

    # p95 latency per model 
    c.execute("SELECT DISTINCT model FROM eval_runs")
    models = [r[0] for r in c.fetchall()]
    p95_map: Dict[str, float] = {}
    for m in models:
        c.execute("SELECT latency_sec FROM eval_runs WHERE model=?", (m,))
        lats = [float(x[0]) for x in c.fetchall()]
        p95_map[m] = percentile(lats, 95)

    conn.close()

    header = [
        "model",
        "turns",
        "avg_latency_sec",
        "p95_latency_sec",
        "avg_2sent_score_(0-2)",
        "pct_has_question_(0-1)",
        "avg_chars",
        "avg_chars_per_sec",
        "pct_language_leak_flag_(0-1)",
    ]

    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (model, turns, avg_lat, avg_2sent, pct_q, avg_chars, avg_cps, pct_leak) in rows:
            w.writerow([
                model,
                int(turns),
                round(float(avg_lat or 0.0), 4),
                round(float(p95_map.get(model, 0.0)), 4),
                round(float(avg_2sent or 0.0), 4),
                round(float(pct_q or 0.0), 4),
                round(float(avg_chars or 0.0), 2),
                round(float(avg_cps or 0.0), 2),
                round(float(pct_leak or 0.0), 4),
            ])


# -----------------------------
# Main runner
# -----------------------------
def run() -> None:
    # Use eval DB file but keep your DB schema/logic intact
    pipeline.DB_PATH = EVAL_DB_PATH
    pipeline.init_db()        # your original tables
    init_eval_table()         # evaluation table

    cases = load_cases()

    for model_name in MODELS:
        base_llm = Ollama(model=model_name, temperature=TEMPERATURE, top_p=TOP_P)
        cap_llm = CapturedLLM(base_llm)
        pipeline.llm = cap_llm  # swap global llm in your pipeline (no code changes)

        print(f"\n=== Model: {model_name} ===")

        for case in cases:
            state = pipeline.State(messages=[])
            state["summary"] = ""
            state["messages"] = []
            state["_turns_since_summary"] = 0  # keep async summary from triggering early

            case_id = case["id"]
            conversation_id = case.get("conversation_id", f"eval_{case_id}")

            assistant_turns = 0

            for idx, turn in enumerate(case["turns"], start=1):
                # Add user message like your app would
                state["messages"].append({"role": "user", "content": turn["content"]})

                # Time the actual pipeline call (includes prompt build + LLM call + cleaning)
                t0 = time.perf_counter()
                out = pipeline.call_model(state)  # uses your strict Turkish prompt
                dt = time.perf_counter() - t0

                assistant_turns += 1
                if assistant_turns > MAX_ASSISTANT_TURNS_PER_CASE:
                    break

                cleaned = out["messages"][-1]["content"]
                raw = cap_llm.last_raw

                char_len = len(cleaned)
                word_len = len((cleaned or "").split())
                scount = count_sentences(cleaned)
                cps = (char_len / dt) if dt > 0 else 0.0

                row = {
                    "model": model_name,
                    "case_id": case_id,
                    "conversation_id": conversation_id,
                    "turn_index": idx,
                    "latency_sec": float(dt),
                    "raw_output": raw,
                    "cleaned_output": cleaned,
                    "sentence_count": int(scount),
                    "score_two_sentences": int(two_sentence_score(cleaned)),
                    "has_question": int(has_question(cleaned)),
                    "char_len": int(char_len),
                    "word_len": int(word_len),
                    "chars_per_sec": float(cps),
                    "language_leak_flag": int(language_leakage_flag(raw)),
                }
                insert_eval_row(row)

                print(f"[{case_id} t{idx}] {dt:.2f}s | sent={scount} | ?={row['has_question']} | leak={row['language_leak_flag']}")

    export_summary_csv()
    print("\nDone.")
    print("SQLite:", EVAL_DB_PATH)
    print("CSV summary:", SUMMARY_CSV_PATH)


if __name__ == "__main__":
    run()
