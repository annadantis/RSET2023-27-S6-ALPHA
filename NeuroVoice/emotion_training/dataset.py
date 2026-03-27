import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

EMOTION_MAP = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
    "07": 6,  # disgust
    "08": 7   # surprised
}

class RAVDESSDataset(Dataset):
    def __init__(self, root_dir, sr=16000, n_mels=64, max_length=216):
        self.root_dir = root_dir
        self.sr = sr
        self.n_mels = n_mels
        self.max_length = max_length
        self.files = []

        for actor in os.listdir(root_dir):
            actor_path = os.path.join(root_dir, actor)
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(actor_path, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_name = os.path.basename(file_path)

        emotion_code = file_name.split("-")[2]
        label = EMOTION_MAP[emotion_code]

        y, sr = librosa.load(file_path, sr=self.sr)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels
        )

        mel = librosa.power_to_db(mel, ref=np.max)

        # Pad or truncate to max_length
        if mel.shape[1] < self.max_length:
            mel = np.pad(mel, ((0, 0), (0, self.max_length - mel.shape[1])), mode='constant')
        else:
            mel = mel[:, :self.max_length]

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        return mel, torch.tensor(label, dtype=torch.long), file_path
