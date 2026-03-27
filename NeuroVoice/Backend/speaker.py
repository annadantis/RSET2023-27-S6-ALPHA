import torch
from resemblyzer import VoiceEncoder, preprocess_wav

def get_speaker_embedding(audio_path):
    print(f"Extracting speaker embedding from {audio_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = VoiceEncoder(device=device)
    
    # Preprocess audio
    wav = preprocess_wav(audio_path)
    
    # Extract 256-dim speaker embedding
    embedding = encoder.embed_utterance(wav)
    
    print(f"Speaker embedding shape: {embedding.shape}")
    return embedding.tolist()
