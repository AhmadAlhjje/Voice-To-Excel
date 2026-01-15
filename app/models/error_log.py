"""
ErrorLog model for tracking errors during processing.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class ErrorType(str, Enum):
    """Error type enumeration."""
    WHISPER = "whisper"
    LLM = "llm"
    EXCEL = "excel"
    VALIDATION = "validation"
    SYSTEM = "system"


class ErrorContext(BaseModel):
    """Context information for the error."""
    row_number: Optional[int] = None
    audio_file: Optional[str] = None
    transcription: Optional[str] = None
    additional_info: Optional[str] = None


class ErrorLog(BaseModel):
    """Error log model for tracking processing errors."""
    id: Optional[str] = Field(None, alias="_id")
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error_type: ErrorType
    error_message: str
    stack_trace: Optional[str] = None
    context: Optional[ErrorContext] = None
    resolved: bool = False

    class Config:
        populate_by_name = True
        use_enum_values = True


class ErrorLogCreate(BaseModel):
    """Schema for creating an error log entry."""
    session_id: str
    error_type: ErrorType
    error_message: str
    stack_trace: Optional[str] = None
    context: Optional[ErrorContext] = None


class ErrorLogResponse(BaseModel):
    """Response schema for error log data."""
    session_id: str
    created_at: datetime
    error_type: str
    error_message: str
    context: Optional[ErrorContext] = None
    resolved: bool

    class Config:
        from_attributes = True
