import torch
import librosa
import numpy as np
import sys
import os
import torch.nn.functional as F

# Add emotion_training to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from emotion_training.model import EmotionCNN

# Emotion label mapping
EMOTION_LABELS = [
    "Neutral",
    "Calm", 
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised"
]

# VAD values for each emotion (Valence, Arousal, Dominance)
EMOTION_VAD = {
    "Neutral": (0.5, 0.3, 0.5),    # (0.5, 0.3, 0.5)
    "Calm": (0.6, 0.2, 0.4),     # (0.6, 0.2, 0.4)
    "Happy": (0.8, 0.7, 0.6),      # (0.8, 0.7, 0.6)
    "Sad": (0.2, 0.3, 0.3),       # (0.2, 0.3, 0.3)
    "Angry": (0.3, 0.8, 0.7),     # (0.3, 0.8, 0.7)
    "Fearful": (0.4, 0.6, 0.5),   # (0.4, 0.6, 0.5)
    "Disgust": (0.2, 0.5, 0.4),    # (0.2, 0.5, 0.4)
    "Surprised": (0.7, 0.6, 0.6)    # (0.7, 0.6, 0.6)
}

def get_emotion_embedding(audio_path):
    print(f"Extracting emotion embedding from {audio_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained emotion model
    model = EmotionCNN()
    model_path = os.path.join(os.path.dirname(__file__), '..', 'emotion_training', 'emotion_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Process audio
    y, sr = librosa.load(audio_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel = librosa.power_to_db(mel, ref=np.max)
    
    # Pad to match training size
    max_len = 216
    if mel.shape[1] < max_len:
        mel = np.pad(mel, ((0, 0), (0, max_len - mel.shape[1])), mode='constant')
    else:
        mel = mel[:, :max_len]
    
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Extract both embedding and logits
    with torch.no_grad():
        embedding = model(mel, return_embedding=True)  # 128-dim embedding
        logits = model(mel, return_embedding=False)  # 8-class logits
    
    # Convert logits to probabilities and get prediction
    probs = F.softmax(logits, dim=1)
    pred_index = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_index].item()
    
    # Get emotion label and VAD values
    emotion_label = EMOTION_LABELS[pred_index]
    valence, arousal, dominance = EMOTION_VAD[emotion_label]
    
    print(f"Emotion: {emotion_label} (confidence: {confidence:.3f})")
    print(f"VAD: V={valence:.3f}, A={arousal:.3f}, D={dominance:.3f}")
    print(f"Emotion embedding shape: {embedding.shape}")
    
    return {
        'embedding': embedding.squeeze(0).cpu().numpy().tolist(),
        'emotion_label': emotion_label,
        'confidence': confidence,
        'valence': valence,
        'arousal': arousal,
        'dominance': dominance
    }
