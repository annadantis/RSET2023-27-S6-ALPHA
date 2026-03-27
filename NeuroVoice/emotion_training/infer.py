import torch
import librosa
import numpy as np
from emotion_training.model import EmotionCNN
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNN()
model_path = os.path.join(os.path.dirname(__file__), "emotion_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def extract_emotion_embedding(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=64
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    # Pad or truncate to match training size (216 if you padded earlier)
    max_len = 216
    if mel.shape[1] < max_len:
        pad_width = max_len - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)))
    else:
        mel = mel[:, :max_len]

    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mel = mel.to(device)

    with torch.no_grad():
        embedding = model(mel, return_embedding=True)

    return embedding.squeeze(0).cpu().numpy()
