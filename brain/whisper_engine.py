"""
Whisper Engine - STT
Runs Faster-Whisper
Supports tiny on GPU and CPU, small & base on CPU only, due to VRAM limitations. 
"""
from faster_whisper import WhisperModel
import tempfile, torchaudio
import numpy as np
import torch
import soundfile as sf

class Transcriber:
    def __init__(self, model_size="base", device="cuda", compute_type="int8"): #base did not work gave me '!!!!' prob because it uses to much VRAM. Small also did not work, only tiny worked. Should I use distil?
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"[Whisper] Loading {model_size} model...")


    def transcribe(self, audio:bytes):
        """
        transcribe an audio file. audio: numpy array of float32
        """
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_file.name, audio, 16000)
        segments, _ = self.model.transcribe(tmp_file.name, language="tr", vad_filter=False, beam_size=1) #mabye add: condition_on_previous_text=False
        text = " ".join([s.text for s in segments])
        return text
        
        
        #print(f"[Whisper] Transcribing {audio_path}...")
        #segments, info = self.model.transcribe(audio_path)
        #text=" ".join(segment.text for segment in segments)
        #print("[STT Result]:", text)
        #return text




    

