# recorder.py
import sounddevice as sd
from scipy.io.wavfile import write
import uuid
import os

RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

def record_audio(duration: int = 15, samplerate: int = 44100) -> str:
    """
    Records audio from default microphone
    Returns path to WAV file
    """
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(RECORDINGS_DIR, filename)

    print("🎙 Recording started...")
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    print("✅ Recording finished")

    write(filepath, samplerate, audio)
    return filepath

if __name__ == "__main__":
    path = record_audio()
    print(f"Recorded file saved at: {path}")
