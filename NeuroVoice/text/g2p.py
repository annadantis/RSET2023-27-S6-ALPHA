"""
Multilingual Grapheme-to-Phoneme Converter
Converts text to phoneme representations for multilingual TTS synthesis
"""

import re
import phonemizer
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualG2P:
    """
    Multilingual Grapheme-to-Phoneme converter supporting multiple languages
    """
    
    # Supported languages and their espeak language codes
    SUPPORTED_LANGUAGES = {
        'en': 'en-us',    # English (US)
        'hi': 'hi',       # Hindi
        'ja': 'ja',       # Japanese
        'fr': 'fr-fr',    # French
        'de': 'de',       # German
        'es': 'es',       # Spanish
        'pt': 'pt-br',    # Portuguese (Brazilian)
        'zh': 'zh',       # Chinese
        'it': 'it',       # Italian
        'ru': 'ru',       # Russian
    }
    
    def __init__(self):
        """Initialize the multilingual G2P converter"""
        self.backends = {}
        self._initialize_backends()
        
    def _initialize_backends(self):
        """Initialize phonemizer backends for all supported languages"""
        logger.info("Initializing phonemizer backends...")
        
        for lang_code, espeak_lang in self.SUPPORTED_LANGUAGES.items():
            try:
                backend = EspeakBackend(espeak_lang, 
                                       with_stress=True,
                                       ties=True,
                                       language_switch='remove-flags')
                self.backends[lang_code] = backend
                logger.info(f"Backend initialized for {lang_code} ({espeak_lang})")
            except Exception as e:
                logger.warning(f"Failed to initialize backend for {lang_code}: {e}")
                
    def text_to_phonemes(self, text: str, language: str = 'en') -> List[str]:
        """
        Convert text string to phoneme sequence
        
        Args:
            text: Input text string
            language: Target language code (default: 'en')
            
        Returns:
            List of phonemes
        """
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language '{language}' not supported. "
                           f"Supported languages: {list(self.SUPPORTED_LANGUAGES.keys())}")
        
        if language not in self.backends:
            raise RuntimeError(f"Backend for language '{language}' not initialized")
        
        # Clean and normalize text
        cleaned_text = self.clean_text(text)
        
        try:
            # Convert to phonemes
            phonemized = self.backends[language].phonemize(
                [cleaned_text],
                separator=Separator(word='|', syllable='', phone=' ')
            )
            
            # Split into individual phonemes and clean
            phoneme_sequence = phonemized[0].split()
            phoneme_sequence = [p for p in phoneme_sequence if p.strip()]
            
            logger.info(f"Converted text to {len(phoneme_sequence)} phonemes for language {language}")
            return phoneme_sequence
            
        except Exception as e:
            logger.error(f"Error converting text to phonemes: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
        
        # Handle numbers (convert to words if needed)
        text = self._handle_numbers(text)
        
        # Handle abbreviations and special cases
        text = self._handle_abbreviations(text)
        
        return text
    
    def _handle_numbers(self, text: str) -> str:
        """
        Convert numbers to words (basic implementation)
        
        Args:
            text: Text containing numbers
            
        Returns:
            Text with numbers converted to words
        """
        # Simple number to words conversion for basic cases
        # This is a placeholder - in production, you'd use a proper number-to-words library
        number_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        
        for num, word in number_words.items():
            text = re.sub(r'\b' + num + r'\b', word, text)
        
        return text
    
    def _handle_abbreviations(self, text: str) -> str:
        """
        Handle common abbreviations
        
        Args:
            text: Text with abbreviations
            
        Returns:
            Text with abbreviations expanded
        """
        abbreviations = {
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Dr.': 'Doctor',
            'Prof.': 'Professor',
            'St.': 'Saint',
            'Ave.': 'Avenue',
            'Blvd.': 'Boulevard',
            'Rd.': 'Road',
            'etc.': 'etcetera'
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
        
        return text
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes
        
        Returns:
            List of supported language codes
        """
        return list(self.SUPPORTED_LANGUAGES.keys())
    
    def is_language_supported(self, language: str) -> bool:
        """
        Check if a language is supported
        
        Args:
            language: Language code to check
            
        Returns:
            True if language is supported, False otherwise
        """
        return language in self.SUPPORTED_LANGUAGES
    
    def phonemes_to_text(self, phonemes: List[str]) -> str:
        """
        Convert phoneme sequence back to text (for debugging)
        
        Args:
            phonemes: List of phonemes
            
        Returns:
            Space-separated phoneme string
        """
        return ' '.join(phonemes)


# Global instance for easy access
g2p_converter = None


def get_g2p_converter() -> MultilingualG2P:
    """
    Get or create global G2P converter instance
    
    Returns:
        MultilingualG2P instance
    """
    global g2p_converter
    if g2p_converter is None:
        g2p_converter = MultilingualG2P()
    return g2p_converter


if __name__ == "__main__":
    # Testing code
    converter = MultilingualG2P()
    
    # Test different languages
    test_texts = {
        'en': "Hello, this is a test of the multilingual system.",
        'hi': "नमस्ते, यह बहुभाषी प्रणाली का परीक्षण है।",
        'ja': "こんにちは、これは多言語システムのテストです。",
        'fr': "Bonjour, ceci est un test du système multilingue."
    }
    
    for lang, text in test_texts.items():
        try:
            print(f"\nTesting {lang}:")
            print(f"Original: {text}")
            phonemes = converter.text_to_phonemes(text, lang)
            print(f"Phonemes: {' '.join(phonemes)}")
        except Exception as e:
            print(f"Error with {lang}: {e}")
