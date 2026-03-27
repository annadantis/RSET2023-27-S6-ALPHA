import torch
import torch.nn as nn
import os

class EmotionFusion(nn.Module):
    def __init__(self, speaker_dim=256, emotion_dim=128):
        super().__init__()
        self.linear = nn.Linear(emotion_dim, speaker_dim)

        # Load trained projection weights
        proj_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "emotion_training",
            "projection.pth"
        )

        if os.path.exists(proj_path):
            state_dict = torch.load(proj_path, map_location="cpu")
            # No need to rename since we're using 'linear' now
            self.load_state_dict(state_dict)
            print("Loaded trained projection weights.")
        else:
            print("WARNING: projection.pth not found. Using random weights.")

    def forward(self, speaker_emb, emotion_emb, alpha=1.3):
        projected_emotion = self.linear(emotion_emb)
        fused = speaker_emb + alpha * projected_emotion
        
        # Normalize fused embedding to prevent distortion
        fused = fused / torch.norm(fused, dim=1, keepdim=True)
        
        return fused
