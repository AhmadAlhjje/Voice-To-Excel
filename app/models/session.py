"""
Session model for tracking Excel file processing sessions.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class SessionStatus(str, Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"


class ExcelFileInfo(BaseModel):
    """Information about the uploaded Excel file."""
    original_name: str
    stored_path: str
    headers: List[str] = Field(default_factory=list)
    total_rows: int = 0
    current_row: int = 1  # Start from row 1 (after headers)


class SessionSettings(BaseModel):
    """Session-specific settings."""
    language: str = "ar"
    auto_advance: bool = True  # Auto-advance to next row after confirmation


class Session(BaseModel):
    """Session model representing a data entry session."""
    id: Optional[str] = Field(None, alias="_id")
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: SessionStatus = SessionStatus.ACTIVE
    excel_file: Optional[ExcelFileInfo] = None
    settings: SessionSettings = Field(default_factory=SessionSettings)

    class Config:
        populate_by_name = True
        use_enum_values = True


class SessionCreate(BaseModel):
    """Schema for creating a new session."""
    settings: Optional[SessionSettings] = None


class SessionUpdate(BaseModel):
    """Schema for updating a session."""
    status: Optional[SessionStatus] = None
    current_row: Optional[int] = None
    settings: Optional[SessionSettings] = None


class SessionResponse(BaseModel):
    """Response schema for session data."""
    session_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    excel_file: Optional[ExcelFileInfo] = None
    settings: SessionSettings

    class Config:
        from_attributes = True
