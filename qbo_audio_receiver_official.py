#qbo_audio_receiver_official.py

#!/usr/bin/env python3
import socket
import struct
import tempfile
import subprocess
import time
import os # Addedd: for manual file deletion

HOST = "0.0.0.0"  # listen on all interfaces
PORT = 60007      

def play_audio(audio_bytes, conn):
    # Signal start of playback
    conn.sendall(b'\xFE') #playback start
    tmp= tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        # Blocking call: returns when playback finishes
        subprocess.run(["aplay", "-D", "plughw:1,0", tmp.name])
        # Optional: small buffer to ensure ALSA drains completely
        time.sleep(0.2)
    finally:
        # Ensure file is removed after playback
        try:
            os.remove(tmp.name)
        except OSError:
            pass
    # Signal finished playback
    conn.sendall(b'\xFF')

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
   # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print("[QBO] Listening for audio on port {}...".format(PORT))
   
    while True:
        conn, addr = s.accept()
        print("[QBO] Connected by", addr)
        with conn:
            while True:
                raw_size = conn.recv(4)
                if not raw_size:
                    break
                size = struct.unpack("!I", raw_size)[0]
                audio_bytes = b""
                while len(audio_bytes) < size:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    audio_bytes += chunk
                if audio_bytes:
                    #print("[QBO] Playing {} bytes".format(len(audio_bytes)))
                    play_audio(audio_bytes, conn)
