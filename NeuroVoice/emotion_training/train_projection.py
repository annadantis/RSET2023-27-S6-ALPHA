import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import RAVDESSDataset
from infer import extract_emotion_embedding
from model import EmotionCNN
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encoders.speaker_encoder import get_speaker_embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Projection(nn.Module):
    def __init__(self, emotion_dim=128, speaker_dim=256):
        super().__init__()
        self.linear = nn.Linear(emotion_dim, speaker_dim)

    def forward(self, x):
        return self.linear(x)


# Load trained emotion model
emotion_model = EmotionCNN()
emotion_model.load_state_dict(
    torch.load(os.path.join(os.path.dirname(__file__), "emotion_model.pth"),
               map_location=device)
)
emotion_model.to(device)
emotion_model.eval()

projection = Projection().to(device)
optimizer = optim.Adam(projection.parameters(), lr=0.001)
criterion = nn.MSELoss()

dataset = RAVDESSDataset(root_dir="../audio_speech_actors_01-24")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for mel, _, file_path in dataloader:
        mel = mel.to(device)

        # Get emotion embedding
        with torch.no_grad():
            emotion_emb = emotion_model(mel, return_embedding=True)

        # Get speaker embedding from raw file path
        speaker_emb = get_speaker_embedding(file_path[0])
        speaker_emb = torch.tensor(speaker_emb, dtype=torch.float32).to(device)

        # Predict speaker space
        pred = projection(emotion_emb)

        loss = criterion(pred.squeeze(0), speaker_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")

    print(f"Epoch {epoch+1} complete")

torch.save(projection.state_dict(), "projection.pth")
print("Projection training complete")
