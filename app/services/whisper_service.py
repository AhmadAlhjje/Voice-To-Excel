"""
Whisper Speech-to-Text Service.
Uses OpenAI's Whisper model locally for Arabic speech recognition.
"""

import whisper
import torch
import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import soundfile as sf

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""
    text: str
    language: str
    confidence: float
    processing_time_ms: int


class WhisperError(Exception):
    """Custom exception for Whisper-related errors."""
    pass


class WhisperService:
    """
    Service for converting speech to text using Whisper.
    Runs completely offline after initial model download.
    """

    _instance: Optional["WhisperService"] = None
    _model: Optional[whisper.Whisper] = None

    def __new__(cls):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the Whisper service."""
        if self._model is None:
            self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {settings.whisper_model}")
            logger.info(f"Device: {settings.whisper_device}")

            # Determine device
            device = settings.whisper_device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"

            # Load model
            self._model = whisper.load_model(
                settings.whisper_model,
                device=device
            )

            logger.info(f"Whisper model loaded successfully on {device}")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise WhisperError(f"Failed to load Whisper model: {e}")

    def transcribe(
        self,
        audio_path: str,
        language: str = None
    ) -> TranscriptionResult:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file
            language: Language code (default: Arabic "ar")

        Returns:
            TranscriptionResult with text and metadata

        Raises:
            WhisperError: If transcription fails
        """
        if self._model is None:
            self._load_model()

        language = language or settings.whisper_language
        start_time = time.time()

        try:
            # Verify file exists
            if not Path(audio_path).exists():
                raise WhisperError(f"Audio file not found: {audio_path}")

            logger.info(f"Transcribing: {audio_path}")

            # Load and preprocess audio
            audio = self._load_audio(audio_path)

            # Transcribe
            result = self._model.transcribe(
                audio,
                language=language,
                task="transcribe",
                fp16=torch.cuda.is_available(),
                verbose=False
            )

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Extract confidence (average of segment probabilities)
            confidence = self._calculate_confidence(result)

            text = result["text"].strip()
            detected_language = result.get("language", language)

            logger.info(f"Transcription complete: {len(text)} characters in {processing_time_ms}ms")

            return TranscriptionResult(
                text=text,
                language=detected_language,
                confidence=confidence,
                processing_time_ms=processing_time_ms
            )

        except WhisperError:
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise WhisperError(f"Transcription failed: {e}")

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file for Whisper.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio as numpy array
        """
        try:
            # Read audio file
            audio, sample_rate = sf.read(audio_path)

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample if necessary (Whisper expects 16kHz)
            if sample_rate != 16000:
                # Simple resampling using numpy
                duration = len(audio) / sample_rate
                target_length = int(duration * 16000)
                indices = np.linspace(0, len(audio) - 1, target_length).astype(int)
                audio = audio[indices]

            # Ensure float32
            audio = audio.astype(np.float32)

            # Normalize
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            return audio

        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise WhisperError(f"Failed to load audio file: {e}")

    def _calculate_confidence(self, result: dict) -> float:
        """
        Calculate average confidence from transcription result.

        Args:
            result: Whisper transcription result

        Returns:
            Confidence score between 0 and 1
        """
        segments = result.get("segments", [])
        if not segments:
            return 0.0

        # Calculate average probability across segments
        probabilities = []
        for segment in segments:
            avg_logprob = segment.get("avg_logprob", -1.0)
            # Convert log probability to probability
            prob = np.exp(avg_logprob)
            probabilities.append(min(prob, 1.0))

        return float(np.mean(probabilities)) if probabilities else 0.0

    async def transcribe_async(
        self,
        audio_path: str,
        language: str = None
    ) -> TranscriptionResult:
        """
        Async wrapper for transcription.
        Note: Whisper itself is synchronous, this just wraps it.

        Args:
            audio_path: Path to the audio file
            language: Language code

        Returns:
            TranscriptionResult
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.transcribe(audio_path, language)
        )

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": settings.whisper_model,
            "device": settings.whisper_device,
            "language": settings.whisper_language,
            "loaded": self.is_loaded()
        }


# Global service instance
_whisper_service: Optional[WhisperService] = None


def get_whisper_service() -> WhisperService:
    """Get or create the Whisper service instance."""
    global _whisper_service
    if _whisper_service is None:
        _whisper_service = WhisperService()
    return _whisper_service
