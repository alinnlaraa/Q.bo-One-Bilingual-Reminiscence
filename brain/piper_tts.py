# brain/piper_tts.py
from pathlib import Path
import subprocess
import tempfile

class PiperTTS:
    def __init__(self, model_path, device="cpu"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")
        self.device = device

    def speak(self, text: str):

        if text.lower().startswith("text"):
            text = text[4:].strip(" :.-\n")
            
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

        PIPER_BIN = "/home/qbo_project_final/venv-qbo/bin/piper"

        cmd = [
            PIPER_BIN,
            "-m", str(self.model_path), 
            "-f", str(tmp_wav.name),
            text
        ]

        if self.device.lower() == "cuda":
            cmd.append("--cuda")

        # run Piper
        subprocess.run(cmd, check=True)

        return tmp_wav.name
