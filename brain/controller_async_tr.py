# controller_async_tr.py

from brain.whisper_engine import Transcriber
from brain.piper_tts import PiperTTS
from brain.langgraph_memory_async_tr import (
    State,
    call_model,
    store_message,
    store_summary,
    load_last_summary,
    init_db,
    summarize_conversation,
    maybe_trigger_async_summary,
)
from datetime import datetime
import uuid
import threading
import re


class BrainController:
    def __init__(
        self,
        stt_model="base",
        stt_device="cuda",
        tts_model_path="/home/qbo_project_final/piper1-gpl/models/piper_tr/tr_TR-dfki-medium.onnx",
        tts_device="cpu",
        language="tr",
    ):
        self.stt = Transcriber(model_size=stt_model, device=stt_device)
        self.tts = PiperTTS(tts_model_path, device=tts_device)
        self.language = language

        self.state = State()
        self.state["messages"] = []
        self.state["summary"] = ""
        # Thread-safety / async summary coordination
        self.state["_lock"] = threading.Lock()
        self.state["_summary_running"] = False
        self.state["_last_summary_ts"] = 0.0
        self.state["_turns_since_summary"] = 0

        self.conversation_id = str(uuid.uuid4())
        self.day = datetime.now().strftime("%Y-%m-%d")

        init_db()
        self.state["summary"] = load_last_summary(self.day) or ""

    def process_text(self, text: str):
        print("[STT]", text)

        text = (text or "").strip()
        if not text:
            print("[DEBUG] Empty transcription, ignored.")
            return None, None

        # Remove last assistant reply from user transcription (safer than global replace)
        with self.state["_lock"]:
            last_assistant = (
                self.state["messages"][-1]["content"] if self.state["messages"] else ""
            )

        la = (last_assistant or "").strip()
        if la:
            # Only strip if it appears as a prefix or suffix (common "echo" STT behavior)
            if text.startswith(la):
                text = text[len(la) :].strip()
            elif text.endswith(la):
                text = text[: -len(la)].strip()

        text = text.strip()
        if not text:
            print("[DEBUG] Empty transcription after cleanup, ignored.")
            return None, None

        # Stop-words (word-boundary safe)
        stop_pattern = re.compile(r"\b(dur|gülegüle|güle\s+güle)\b", re.IGNORECASE)
        if stop_pattern.search(text):
            # End-of-session synchronous summary
            try:
                with self.state["_lock"]:
                    summarize_conversation(self.state)
                    store_summary(self.day, self.state["summary"])
                print("[DEBUG] End-of-session summary stored:", self.state["summary"])
            except Exception as e:
                print("[ERROR] Failed to summarize/store end-of-session summary:", e)
            return "STOP", None

        # Append user message + store
        try:
            with self.state["_lock"]:
                self.state["messages"].append({"role": "user", "content": text})
                self.state["_turns_since_summary"] += 1
            store_message(self.conversation_id, "user", text)
        except Exception as e:
            print("[ERROR] Failed to store user message:", e)

        # LLM response (reply now; summary is triggered in langgraph file)
        try:
            reply_dict = call_model(self.state, language=self.language, name="User")
            reply_text = reply_dict["messages"][-1]["content"]
        except Exception as e:
            print("[ERROR] LLM call failed:", e)
            reply_text = "Üzgünüm Çiğdem, şu an bir sorun var."

        try:
            store_message(self.conversation_id, "assistant", reply_text)
        except Exception as e:
            print("[ERROR] Failed to store assistant message:", e)

        print("[LLM]", reply_text)

        # Optional: extra periodic async summary trigger (gated + debounced)
        # Keeps the same overall behavior, but prevents duplicate/racing summaries.
        try:
            maybe_trigger_async_summary(self.state, self.day)
        except Exception as e:
            print("[ERROR] Failed to trigger async summary:", e)

        # TTS
        try:
            audio_path = self.tts.speak(reply_text)
            print("[TTS]", audio_path)
            return reply_text, audio_path
        except Exception as e:
            print("[ERROR] TTS failed:", e)
            return reply_text, None
