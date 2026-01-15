"""
AudioLog model for tracking audio recordings and transcriptions.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class AudioStatus(str, Enum):
    """Audio processing status."""
    PENDING = "pending"
    TRANSCRIBED = "transcribed"
    PROCESSED = "processed"
    FAILED = "failed"


class TranscriptionInfo(BaseModel):
    """Transcription result information."""
    text: str = ""
    confidence: float = 0.0
    processing_time_ms: int = 0


class AudioLog(BaseModel):
    """Audio log model for tracking voice recordings."""
    id: Optional[str] = Field(None, alias="_id")
    session_id: str
    row_number: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    audio_file_path: str
    duration_seconds: float = 0.0
    transcription: Optional[TranscriptionInfo] = None
    status: AudioStatus = AudioStatus.PENDING

    class Config:
        populate_by_name = True
        use_enum_values = True


class AudioLogCreate(BaseModel):
    """Schema for creating an audio log entry."""
    session_id: str
    row_number: int
    audio_file_path: str
    duration_seconds: float = 0.0


class AudioLogResponse(BaseModel):
    """Response schema for audio log data."""
    session_id: str
    row_number: int
    created_at: datetime
    duration_seconds: float
    transcription: Optional[TranscriptionInfo] = None
    status: str

    class Config:
        from_attributes = True
