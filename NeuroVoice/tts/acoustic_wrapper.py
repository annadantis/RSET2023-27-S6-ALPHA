import torch
import numpy as np
from TTS.api import TTS

from encoders.speaker_encoder import get_speaker_embedding
from emotion_training.infer import extract_emotion_embedding
from tts.fusion import EmotionFusion


class EmotionTTS:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/your_tts"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load YourTTS backbone
        self.tts = TTS(model_name=model_name, gpu=torch.cuda.is_available())

        # Initialize fusion module
        self.fusion = EmotionFusion().to(self.device)

    def synthesize(
        self,
        text,
        reference_audio,
        language="en",
        output_path="output.wav",
        alpha=0.3
    ):
        # --- Speaker embedding (256-d) ---
        speaker_emb = get_speaker_embedding(reference_audio)
        speaker_emb = torch.tensor(speaker_emb, dtype=torch.float32).to(self.device)

        # --- Emotion embedding (128-d) ---
        emotion_emb = extract_emotion_embedding(reference_audio)
        emotion_emb = torch.tensor(emotion_emb, dtype=torch.float32).to(self.device)

        # --- Fuse ---
        fused_emb = self.fusion(
            speaker_emb.unsqueeze(0),
            emotion_emb.unsqueeze(0),
            alpha=alpha
        )

        fused_emb = fused_emb.squeeze(0).detach().cpu().numpy()

        # --- Generate speech ---
        self.tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio,  # satisfy multi-speaker requirement
            speaker_embedding=fused_emb,  # override embedding internally
            language=language,
            file_path=output_path
        )

        print(f"Generated audio saved to {output_path}")
