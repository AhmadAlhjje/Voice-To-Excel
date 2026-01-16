"""
LLM Service for text understanding and data extraction.
Uses Ollama with Qwen2.5-7B-Instruct model running locally.
"""

import httpx
import json
import time
import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from app.config import settings
from app.prompts.extraction_prompt import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_MULTI_ROW,
    ROW_SEPARATOR_KEYWORDS,
    get_extraction_prompt,
    get_correction_prompt,
    get_multi_row_extraction_prompt
)
from app.services.arabic_numbers import process_extracted_data

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of data extraction from text."""
    raw_json: str
    parsed_data: Dict[str, Any]
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class MultiRowExtractionResult:
    """Result of multi-row data extraction from text."""
    raw_json: str
    rows: List[Dict[str, Any]]
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


class LLMService:
    """
    Service for extracting structured data from Arabic text.
    Uses Ollama API to communicate with local Qwen2.5 model.
    """

    def __init__(self):
        """Initialize the LLM service."""
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        self.timeout = settings.llm_timeout

    async def check_health(self) -> bool:
        """
        Check if Ollama is running and the model is available.

        Returns:
            True if service is healthy
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    if any(self.model.split(":")[0] in m for m in models):
                        return True
                    logger.warning(f"Model {self.model} not found. Available: {models}")
                return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def extract_data(
        self,
        transcription: str,
        headers: List[str]
    ) -> ExtractionResult:
        """
        Extract structured data from transcribed Arabic text.

        Args:
            transcription: The transcribed text from speech
            headers: List of column headers from Excel

        Returns:
            ExtractionResult with parsed data

        Raises:
            LLMError: If extraction fails
        """
        start_time = time.time()

        try:
            # Generate the prompt
            user_prompt = get_extraction_prompt(headers, transcription)

            # Call Ollama API
            response_text = await self._call_ollama(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )

            # Parse the JSON response
            parsed_data = self._parse_json_response(response_text, headers)

            # Process Arabic numbers in the extracted data
            processed_data = process_extracted_data(parsed_data, headers)

            processing_time_ms = int((time.time() - start_time) * 1000)

            logger.info(f"Data extraction complete in {processing_time_ms}ms")

            return ExtractionResult(
                raw_json=response_text,
                parsed_data=processed_data,
                processing_time_ms=processing_time_ms,
                success=True
            )

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Data extraction failed: {e}")

            return ExtractionResult(
                raw_json="",
                parsed_data={},
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )

    def _has_row_separators(self, transcription: str) -> bool:
        """
        Check if transcription contains row separator keywords.

        Args:
            transcription: The transcribed text

        Returns:
            True if separator keywords are found
        """
        transcription_lower = transcription.lower()
        for keyword in ROW_SEPARATOR_KEYWORDS:
            if keyword in transcription_lower:
                return True
        return False

    async def extract_multi_row_data(
        self,
        transcription: str,
        headers: List[str]
    ) -> MultiRowExtractionResult:
        """
        Extract structured data for multiple rows from transcribed Arabic text.

        Args:
            transcription: The transcribed text from speech (may contain multiple rows)
            headers: List of column headers from Excel

        Returns:
            MultiRowExtractionResult with parsed data for each row

        Raises:
            LLMError: If extraction fails
        """
        start_time = time.time()

        try:
            # Generate the multi-row prompt
            user_prompt = get_multi_row_extraction_prompt(headers, transcription)

            # Call Ollama API with multi-row system prompt
            response_text = await self._call_ollama(
                system_prompt=SYSTEM_PROMPT_MULTI_ROW,
                user_prompt=user_prompt
            )

            # Parse the JSON array response
            rows = self._parse_multi_row_response(response_text, headers)

            # Process Arabic numbers in each row
            processed_rows = [
                process_extracted_data(row, headers)
                for row in rows
            ]

            processing_time_ms = int((time.time() - start_time) * 1000)

            logger.info(f"Multi-row extraction complete: {len(processed_rows)} rows in {processing_time_ms}ms")

            return MultiRowExtractionResult(
                raw_json=response_text,
                rows=processed_rows,
                processing_time_ms=processing_time_ms,
                success=True
            )

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Multi-row data extraction failed: {e}")

            return MultiRowExtractionResult(
                raw_json="",
                rows=[],
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )

    def _parse_multi_row_response(
        self,
        response: str,
        headers: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Parse JSON array from LLM response for multi-row extraction.

        Args:
            response: Raw LLM response text
            headers: Expected column headers

        Returns:
            List of parsed and validated dictionaries

        Raises:
            LLMError: If JSON parsing fails
        """
        # Clean the response
        cleaned = response.strip()

        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            cleaned = "\n".join(json_lines)

        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', cleaned)
        if json_match:
            cleaned = json_match.group()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nResponse: {cleaned}")
            raise LLMError(f"Invalid JSON response from LLM: {e}")

        if not isinstance(data, list):
            # If single object returned, wrap in list
            if isinstance(data, dict):
                data = [data]
            else:
                raise LLMError("LLM response is not a JSON array or object")

        # Validate each row
        validated_rows = []
        for row in data:
            if not isinstance(row, dict):
                continue
            validated = {}
            for header in headers:
                if header in row:
                    validated[header] = row[header]
                else:
                    validated[header] = None
            validated_rows.append(validated)

        return validated_rows

    async def correct_data(
        self,
        headers: List[str],
        original_data: Dict[str, Any],
        correction: str
    ) -> ExtractionResult:
        """
        Correct previously extracted data based on user instruction.

        Args:
            headers: List of column headers
            original_data: The original extracted data
            correction: User's correction instruction

        Returns:
            ExtractionResult with corrected data
        """
        start_time = time.time()

        try:
            user_prompt = get_correction_prompt(headers, original_data, correction)

            response_text = await self._call_ollama(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )

            parsed_data = self._parse_json_response(response_text, headers)
            processed_data = process_extracted_data(parsed_data, headers)

            processing_time_ms = int((time.time() - start_time) * 1000)

            return ExtractionResult(
                raw_json=response_text,
                parsed_data=processed_data,
                processing_time_ms=processing_time_ms,
                success=True
            )

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Data correction failed: {e}")

            return ExtractionResult(
                raw_json="",
                parsed_data=original_data,  # Return original on failure
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )

    async def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Make a request to Ollama API.

        Args:
            system_prompt: The system instruction
            user_prompt: The user's request

        Returns:
            Generated text response

        Raises:
            LLMError: If the API call fails
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": user_prompt,
                        "system": system_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temperature for consistent output
                            "top_p": 0.9,
                            "num_predict": 1024,
                        }
                    }
                )

                if response.status_code != 200:
                    raise LLMError(f"Ollama API error: {response.status_code} - {response.text}")

                data = response.json()
                return data.get("response", "")

        except httpx.TimeoutException:
            raise LLMError("Ollama request timed out")
        except httpx.ConnectError:
            raise LLMError("Could not connect to Ollama. Make sure it's running.")
        except Exception as e:
            raise LLMError(f"Ollama API error: {e}")

    def _parse_json_response(
        self,
        response: str,
        headers: List[str]
    ) -> Dict[str, Any]:
        """
        Parse JSON from LLM response and validate against headers.

        Args:
            response: Raw LLM response text
            headers: Expected column headers

        Returns:
            Parsed and validated dictionary

        Raises:
            LLMError: If JSON parsing fails
        """
        # Clean the response
        cleaned = response.strip()

        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            # Find the end of the code block
            lines = cleaned.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            cleaned = "\n".join(json_lines)

        # Try to find JSON object in response
        json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nResponse: {cleaned}")
            raise LLMError(f"Invalid JSON response from LLM: {e}")

        if not isinstance(data, dict):
            raise LLMError("LLM response is not a JSON object")

        # Validate and filter to only include valid headers
        validated = {}
        for header in headers:
            if header in data:
                validated[header] = data[header]
            else:
                validated[header] = None

        return validated

    def get_service_info(self) -> dict:
        """Get information about the LLM service configuration."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout
        }


# Global service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
