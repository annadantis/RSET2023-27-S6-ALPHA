"""
Speaker Identity Extraction Module
Extracts unique voice characteristics from reference audio using Resemblyzer
"""

from resemblyzer import VoiceEncoder, preprocess_wav
import torch
import numpy as np
import logging
from typing import Union, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeakerEncoder:
    """
    Speaker encoder that extracts voice identity characteristics from audio
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the speaker encoder
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing speaker encoder on device: {self.device}")
        
        # Initialize the Resemblyzer voice encoder
        self.encoder = VoiceEncoder(device=self.device)
        logger.info("Speaker encoder initialized successfully")
    
    def get_speaker_embedding(self, wav_path: Union[str, Path]) -> np.ndarray:
        """
        Extract speaker embedding from audio file
        
        Args:
            wav_path: Path to WAV audio file
            
        Returns:
            Speaker embedding vector (numpy array)
        """
        wav_path = Path(wav_path)
        
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        
        if wav_path.suffix.lower() != '.wav':
            raise ValueError(f"Only WAV files are supported. Got: {wav_path.suffix}")
        
        try:
            # Preprocess the audio file
            logger.info(f"Processing audio file: {wav_path}")
            wav = preprocess_wav(wav_path)
            
            # Extract speaker embedding
            embedding = self.encoder.embed_utterance(wav)
            
            logger.info(f"Speaker embedding extracted. Shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            raise
    
    def get_speaker_embedding_from_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract speaker embedding from audio data array
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Speaker embedding vector (numpy array)
        """
        try:
            # Ensure audio is in the correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Extract speaker embedding directly from audio data
            embedding = self.encoder.embed_utterance(audio_data)
            
            logger.info(f"Speaker embedding extracted from audio data. Shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting speaker embedding from audio data: {e}")
            raise
    
    def compare_speakers(self, wav_path1: Union[str, Path], wav_path2: Union[str, Path]) -> float:
        """
        Compare two audio files and return similarity score
        
        Args:
            wav_path1: Path to first audio file
            wav_path2: Path to second audio file
            
        Returns:
            Cosine similarity between speaker embeddings (0-1)
        """
        emb1 = self.get_speaker_embedding(wav_path1)
        emb2 = self.get_speaker_embedding(wav_path2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        logger.info(f"Speaker similarity: {similarity:.4f}")
        return similarity
    
    def get_embedding_info(self) -> dict:
        """
        Get information about the speaker encoder
        
        Returns:
            Dictionary with encoder information
        """
        return {
            "device": self.device,
            "embedding_dim": 256,  # Resemblyzer default embedding dimension
            "model_type": "Resemblyzer VoiceEncoder",
            "supported_formats": [".wav"],
            "recommended_audio_length": "5-10 seconds"
        }


# Global instance for easy access
_speaker_encoder = None


def get_speaker_encoder() -> SpeakerEncoder:
    """
    Get or create global speaker encoder instance
    
    Returns:
        SpeakerEncoder instance
    """
    global _speaker_encoder
    if _speaker_encoder is None:
        _speaker_encoder = SpeakerEncoder()
    return _speaker_encoder


def get_speaker_embedding(wav_path: Union[str, Path]) -> np.ndarray:
    """
    Convenience function to get speaker embedding
    
    Args:
        wav_path: Path to WAV audio file
        
    Returns:
        Speaker embedding vector
    """
    encoder = get_speaker_encoder()
    return encoder.get_speaker_embedding(wav_path)


if __name__ == "__main__":
    # Testing code
    encoder = SpeakerEncoder()
    
    # Print encoder info
    info = encoder.get_embedding_info()
    print("Speaker Encoder Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with a sample file (if available)
    test_file = "test_reference.wav"
    if Path(test_file).exists():
        try:
            embedding = encoder.get_speaker_embedding(test_file)
            print(f"\nTest successful!")
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding dtype: {embedding.dtype}")
            print(f"Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print(f"\nTest file '{test_file}' not found. Create it to test the encoder.")
