"""
Configuration settings for Voice To Excel application.
All settings can be overridden via environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Voice To Excel"
    debug: bool = False

    # MongoDB
    mongodb_url: str = Field(alias="MONGODB_URL")
    mongodb_database: str = Field(alias="MONGODB_DATABASE")

    # Ollama (LLM)
    ollama_base_url: str = Field(alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(alias="OLLAMA_MODEL")
    llm_timeout: int = Field(alias="LLM_TIMEOUT")

    # Whisper
    whisper_model: str = Field(alias="WHISPER_MODEL")  # tiny, base, small, medium, large-v3
    whisper_device: str = Field(alias="WHISPER_DEVICE")  # cpu, cuda
    whisper_language: str = Field(alias="WHISPER_LANGUAGE")

    # Storage paths
    storage_path: Path = Field(alias="STORAGE_PATH")
    audio_storage_path: Path = Field(alias="AUDIO_STORAGE_PATH")
    excel_storage_path: Path = Field(alias="EXCEL_STORAGE_PATH")

    # Audio settings
    max_audio_duration: int = Field(alias="MAX_AUDIO_DURATION")
    audio_sample_rate: int = Field(alias="AUDIO_SAMPLE_RATE")

    # API settings
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
