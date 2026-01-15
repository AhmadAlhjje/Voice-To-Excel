"""
ParsedRow model for tracking extracted data from voice input.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RowStatus(str, Enum):
    """Row processing status."""
    DRAFT = "draft"  # Extracted but not confirmed
    CONFIRMED = "confirmed"  # User confirmed the data
    WRITTEN = "written"  # Data written to Excel


class LLMResponseInfo(BaseModel):
    """LLM extraction response information."""
    raw_json: str = ""
    parsed_data: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: int = 0


class ParsedRow(BaseModel):
    """Parsed row model for extracted data."""
    id: Optional[str] = Field(None, alias="_id")
    session_id: str
    row_number: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    original_transcription: str = ""
    llm_response: Optional[LLMResponseInfo] = None
    final_data: Dict[str, Any] = Field(default_factory=dict)
    status: RowStatus = RowStatus.DRAFT
    written_to_excel: bool = False

    class Config:
        populate_by_name = True
        use_enum_values = True


class ParsedRowCreate(BaseModel):
    """Schema for creating a parsed row."""
    session_id: str
    row_number: int
    original_transcription: str
    llm_response: Optional[LLMResponseInfo] = None
    final_data: Optional[Dict[str, Any]] = None


class ParsedRowUpdate(BaseModel):
    """Schema for updating a parsed row."""
    final_data: Optional[Dict[str, Any]] = None
    status: Optional[RowStatus] = None


class ParsedRowResponse(BaseModel):
    """Response schema for parsed row data."""
    session_id: str
    row_number: int
    created_at: datetime
    updated_at: datetime
    original_transcription: str
    llm_response: Optional[LLMResponseInfo] = None
    final_data: Dict[str, Any]
    status: str
    written_to_excel: bool

    class Config:
        from_attributes = True
