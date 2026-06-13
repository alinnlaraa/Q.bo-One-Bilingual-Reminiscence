#!/usr/bin/env python3
import socket
import struct
import io
import os
import time
import threading
import queue
import subprocess

import soundfile as sf
import numpy as np
import librosa
import difflib
import traceback
import signal

# ---------------------------------------------------------
# AUTO-KILL PREVIOUS SERVER PROCESSES 
# ---------------------------------------------------------
PORTS_TO_FREE = [50007, 60007]

for port in PORTS_TO_FREE:
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                print(f"[AUTO-KILL] Killing PID {pid} using port {port}")
                os.kill(int(pid), signal.SIGTERM)
    except Exception as e:
        print(f"[AUTO-KILL] Error freeing port {port}: {e}")


from brain.controller_async_de import BrainController  # your existing module

HOST = "0.0.0.0"
PORT = 50007

# ---------------------------------------------------------
# Command bytes for QBO client (LED + shutdown)
# ---------------------------------------------------------
CMD_SPEAKING = b'\x00'   # speaking (blue)
CMD_LISTENING = b'\x01'  # listening (green)
CMD_STOP_CONVERSATION = b'\x02'   # stop convo
CMD_THINKING = b'\x10'   # thinking (blinking blue)
CMD_OFF = b'\x03' # LED off completly
# ---------------------------------------------------------
# Wake/stop phrases
# ---------------------------------------------------------
WAKE_PHRASES_DE = [
    "hallo qbo",
    "ich bin bereit",

]

STOP_PHRASES_DE = [
    "stopp",
    "stop",
    "tschüss",
]

WAKE_PHRASES_TR = [
    "merhaba qbo",
    "hazırım",
]

STOP_PHRASES_TR = [
    "dur",
    "gülegüle",
    "güle güle",
]


# ---------------------------------------------------------
# Brain initialization
# ---------------------------------------------------------
brain = BrainController(
    stt_model="base",
    stt_device="cuda",
    language="de" #turkish tr / german de
)

# ---------------------------------------------------------
# Audio conversion for QBO
# ---------------------------------------------------------
def convert_for_qbo(path):
    try:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        pcm16 = (audio * 32767).astype(np.int16)
        sf.write(path, pcm16, 16000, subtype="PCM_16")
    except Exception as e:
        print("[SERVER] Audio conversion error:", e)

# ---------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------
def fuzzy_match(text, phrases, threshold=0.6):
    for p in phrases:
        if difflib.SequenceMatcher(None, text, p).ratio() >= threshold:
            return True
    return False

# ---------------------------------------------------------
# Send audio to QBO speaker (port 60007)
# and relay playback start/stop to QBO client
# ---------------------------------------------------------
def send_audio_to_qbo(qbo_ip, wav_path, conn_to_qbo_client, vad_muted):
    with open(wav_path, "rb") as f:
        data = f.read()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((qbo_ip, 60007))
        s.sendall(struct.pack("!I", len(data)))
        s.sendall(data)

        # QBO audio receiver sends 0xFE at playback start, 0xFF at end
        while True:
            sig = s.recv(1)
            if not sig:
                break
            try:
                if sig == b'\xFE':
                    vad_muted.set()  # MUTE AirHug VAD
                    conn_to_qbo_client.sendall(CMD_SPEAKING)

                elif sig == b'\xFF':
                    time.sleep(0.9)
                    vad_muted.clear()  # UNMUTE AirHug VAD
                    conn_to_qbo_client.sendall(CMD_LISTENING)
                    break

            except BrokenPipeError:
                print("[SERVER] QBO client disconnected before unmute.")
                break

# ---------------------------------------------------------
# AirHug VAD thread (with echo prevention)
# ---------------------------------------------------------
def airhug_vad_loop(audio_queue, stop_event, vad_muted):
    SAMPLE_RATE = 16000
    BLOCK_SIZE = 2000          # 0.25 s per block
    SILENCE_THRESHOLD = 0.03   # tune upward if needed
    SILENCE_DURATION = 2.5     # seconds of silence to end utterance
    MIN_AUDIO_DURATION = 1   # ignore very short blips
    COOLDOWN = 0.5             # pause before next utterance

    recording = []
    silence_time = 0.0
    spoken_flag = False

    # small “ramp” to avoid ending on 1–2 quiet blocks mid-sentence
    SILENCE_BLOCKS_TO_END = 6 # 6 * 0.125s ≈ 0.75s
    silence_blocks = 0

    cmd = [
        "arecord",
        "-D", "plughw:A21,0",
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", "1",
        "-q"
    ]
    print("[VAD] Starting AirHug recording:", " ".join(cmd))
    arecord = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    try:
        target_bytes = BLOCK_SIZE * 2  # int16 -> 2 bytes

        while not stop_event.is_set():
            raw = arecord.stdout.read(target_bytes)
            if not raw:
                break

            # Hard mute when QBO is speaking
            if vad_muted.is_set():
                if recording:
                    print("[VAD] Clearing buffer due to mute (QBO speaking).")
                recording = []
                silence_time = 0.0
                silence_blocks = 0
                spoken_flag = False
                continue

            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            if chunk.size == 0:
                continue
            rms = np.sqrt(np.mean(chunk ** 2))

            # Debug: see the energy pattern
            print("[VAD] RMS={:.4f} spoken={} silence_time={:.2f}".format(rms, spoken_flag, silence_time))

            if rms >= SILENCE_THRESHOLD:
                if not spoken_flag:
                    print("[VAD] Speech start detected (RMS {:.4f})".format(rms))
                spoken_flag = True
                silence_time = 0.0
                silence_blocks = 0
            else:
                if spoken_flag:
                    silence_time += BLOCK_SIZE / float(SAMPLE_RATE)
                    silence_blocks +=1

            if spoken_flag:
                recording.append(chunk)

            # End of utterance?
            if spoken_flag and silence_time >= SILENCE_DURATION:
                if recording:
                    audio = np.concatenate(recording)
                    duration = len(audio) / float(SAMPLE_RATE)
                else:
                    audio= None
                    duration = 0.0

                print("[VAD] Utterance end. Duration={:.2f}s".format(duration))

                recording = []
                silence_time = 0.0
                silence_blocks = 0
                spoken_flag = False

                if audio is not None and duration >= MIN_AUDIO_DURATION:
                    print("[VAD] Utterance accepted ({:.2f}s), sending to queue.".format(duration))
                    audio_queue.put(audio)
                    time.sleep(COOLDOWN)
                else:
                    print("[VAD] Utterance rejected as too short ({:.2f}s).".format(duration))

    finally:
        print("[VAD] Stopping AirHug recording.")
        try:
            arecord.terminate()
        except Exception:
            pass


# ---------------------------------------------------------
# Main state machine
# ---------------------------------------------------------
def main():
    qbo_ip = "qbo.local"

    audio_queue = queue.Queue()
    stop_event = threading.Event()
    vad_muted = threading.Event()  # NEW: echo prevention flag

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[SERVER] Listening for QBO client on {HOST}:{PORT}...")
#--------------------- TESTS ----------------------
        # Add this to mark server fully ready
        ready_file = "/tmp/qbo_server_ready"
        if os.path.exists(ready_file):
            os.remove(ready_file)
#--------------------- END ----------------------
        print("Waiting for client to connect")
        conn, addr = s.accept()
        print("[SERVER] QBO client connected:", addr)
#--------------------- TESTS ----------------------
        with open(ready_file, "w") as f:
            f.write("ready")
        print(f"[SERVER] Ready file created at {ready_file}")
#--------------------- END ----------------------
        # Start VAD thread
        vad_thread = threading.Thread(
            target=airhug_vad_loop,
            args=(audio_queue, stop_event, vad_muted),
            daemon=True
        )
        vad_thread.start()

        # -------------------------
        # GREETING
        # -------------------------
        greeting_text = (
            "Hallo, Dilek! Schön, dass du da bist. "
            "Wenn du bereit bist, sag einfach: Ich bin bereit."

        )
        greeting_wav = brain.tts.speak(greeting_text)
        convert_for_qbo(greeting_wav)
        send_audio_to_qbo(qbo_ip, greeting_wav, conn, vad_muted)
        os.remove(greeting_wav)

        mode = "WAKE"
        print("[SERVER] Entering WAKE mode.")
        try:
            conn.sendall(CMD_LISTENING)
        except BrokenPipeError:
            print("SERVER: Client disconnected too early.")
            return

        try:
            while True:
                audio = audio_queue.get()
                if audio is None:
                    continue

                
                try:
                    text = (brain.stt.transcribe(audio) or "").lower().strip()
                except Exception as e:
                    print("[SERVER] STT error:", e)
                    try:
                        conn.sendall(CMD_LISTENING)
                    except BrokenPipeError:
                        print("[SERVER] Client disconnected during STT error handling.")
                        break
                    vad_muted.clear()
                    continue

                if not text:
                    # Nothing meaningful detected — recover state
                    try:
                        conn.sendall(CMD_LISTENING)
                    except BrokenPipeError:
                        print("[SERVER] Client disconnected after empty STT result.")
                        break
                    vad_muted.clear()
                    continue
              
                # -------------------------
                # WAKE MODE
                # -------------------------
                if mode == "WAKE":
                    print("[WAKE]", text)

                    if fuzzy_match(text, WAKE_PHRASES_DE):
                        vad_muted.set()
                        conn.sendall(CMD_THINKING)

                        ack_text = "Ich höre zu. Wie geht es dir? Über was möchtest du sprechen?"
                        ack_wav = brain.tts.speak(ack_text)
                        convert_for_qbo(ack_wav)
                        send_audio_to_qbo(qbo_ip, ack_wav, conn, vad_muted)
                        os.remove(ack_wav)

                        mode = "CHAT"
                        print("[SERVER] Switched to CHAT mode.")
                    continue

                # -------------------------
                # CHAT MODE
                # -------------------------
                print("[CHAT]", text)

                if fuzzy_match(text, STOP_PHRASES_DE):
                    vad_muted.set()
                    conn.sendall(CMD_THINKING)

                    bye_text = "Auf Wiedersehen, Dilek. Bis zu unserem nächsten Gespräch."
                    bye_wav = brain.tts.speak(bye_text)
                    convert_for_qbo(bye_wav)
                    send_audio_to_qbo(qbo_ip, bye_wav, conn, vad_muted)
                    os.remove(bye_wav)
                    
                    print("Server: Convo ended. Stopping all loops.")

                    conn.sendall(CMD_OFF)
                    break

                
                conn.sendall(CMD_THINKING)
                vad_muted.set()

                reply_text, reply_wav = brain.process_text(text)
              
                if reply_wav is None:
                    print(f"[SERVER] Warning: TTS failed for text: {reply_text}")
                    fallback_text = "Es tut mir leid, ich hatte gerade Schwierigkeiten beim Nachdenken."
                    reply_wav = brain.tts.speak(fallback_text)
                    continue  # skip sending audio, move to next iteration
                
                convert_for_qbo(reply_wav)
                send_audio_to_qbo(qbo_ip, reply_wav, conn, vad_muted)
                os.remove(reply_wav)

        finally:
            stop_event.set()
            print("[SERVER] Shutting down server main loop.")

if __name__ == "__main__": 
    try:
        main()
    except Exception as e:
        with open("/home/qbo_project_final/server_official_crash.log", "a") as f:
            f.write("Server crashed:\n")
            f.write(traceback.format_exc())
        raise
