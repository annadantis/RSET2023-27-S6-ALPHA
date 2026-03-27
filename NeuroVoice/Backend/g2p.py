from phonemizer.backend import EspeakBackend
from phonemizer import Phonemizer

def text_to_phonemes(text, language="en"):
    """
    Convert text to phonemes using phonemizer with espeak backend
    
    Args:
        text: Input text string
        language: Language code (en, es, fr, etc.)
    
    Returns:
        String of phonemes
    """
    print(f"Converting text to phonemes: {text}")
    
    # Initialize phonemizer with espeak backend
    phonemizer = Phonemizer(
        backend=EspeakBackend(),
        language=language,
        preserve_punctuation=True,
        with_stress=True
    )
    
    try:
        phonemes = phonemizer.phonemize([text])[0]
        print(f"Phonemes: {phonemes}")
        return phonemes
    except Exception as e:
        print(f"Error in phonemization: {e}")
        return text  # Fallback to original text
