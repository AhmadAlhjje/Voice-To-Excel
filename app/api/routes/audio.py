"""
Audio processing API routes.
Handles voice recording upload, transcription, and data extraction.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from typing import Optional
import os
import uuid
from datetime import datetime
from pathlib import Path
import traceback

from app.config import settings
from app.services.whisper_service import get_whisper_service, WhisperError
from app.services.llm_service import get_llm_service, LLMError, MultiRowExtractionResult
from app.services.session_service import get_session_service
from app.prompts.extraction_prompt import ROW_SEPARATOR_KEYWORDS
from app.models.error_log import ErrorType, ErrorContext
from app.models.parsed_row import LLMResponseInfo

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/process/{session_id}")
async def process_audio(
    session_id: str,
    file: UploadFile = File(...),
    row_number: Optional[int] = Form(None)
):
    """
    Process an audio file: transcribe and extract data.

    This is the main endpoint that:
    1. Saves the audio file
    2. Transcribes using Whisper
    3. Extracts data using LLM
    4. Returns structured data for confirmation

    Args:
        session_id: Session ID
        file: Audio file (wav, mp3, webm, etc.)
        row_number: Optional row number (uses current_row if not provided)

    Returns:
        Transcription and extracted data
    """
    session_service = get_session_service()
    whisper_service = get_whisper_service()
    llm_service = get_llm_service()

    # Get session and validate
    session = await session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if not session.excel_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Excel file uploaded. Please upload an Excel file first."
        )

    # Determine row number
    current_row = row_number if row_number else session.excel_file.current_row
    headers = session.excel_file.headers

    try:
        # 1. Save audio file
        audio_path = await _save_audio_file(session_id, current_row, file)

        # Create audio log
        await session_service.create_audio_log(
            session_id=session_id,
            row_number=current_row,
            audio_file_path=audio_path
        )

        # 2. Transcribe with Whisper
        try:
            transcription_result = await whisper_service.transcribe_async(audio_path)
            transcription_text = transcription_result.text

            # Update audio log with transcription
            await session_service.update_audio_transcription(
                session_id=session_id,
                row_number=current_row,
                text=transcription_text,
                confidence=transcription_result.confidence,
                processing_time_ms=transcription_result.processing_time_ms
            )

        except WhisperError as e:
            await session_service.log_error(
                session_id=session_id,
                error_type=ErrorType.WHISPER,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                context=ErrorContext(
                    row_number=current_row,
                    audio_file=audio_path
                )
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {e}"
            )

        # 3. Extract data with LLM
        try:
            extraction_result = await llm_service.extract_data(
                transcription=transcription_text,
                headers=headers
            )

            if not extraction_result.success:
                await session_service.log_error(
                    session_id=session_id,
                    error_type=ErrorType.LLM,
                    error_message=extraction_result.error_message or "Unknown LLM error",
                    context=ErrorContext(
                        row_number=current_row,
                        transcription=transcription_text
                    )
                )

            extracted_data = extraction_result.parsed_data

        except LLMError as e:
            await session_service.log_error(
                session_id=session_id,
                error_type=ErrorType.LLM,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                context=ErrorContext(
                    row_number=current_row,
                    transcription=transcription_text
                )
            )
            # Continue with empty data - user can still edit manually
            extracted_data = {header: None for header in headers}
            extraction_result = type('obj', (object,), {
                'raw_json': '',
                'processing_time_ms': 0
            })()

        # 4. Create parsed row
        llm_response_info = LLMResponseInfo(
            raw_json=extraction_result.raw_json if hasattr(extraction_result, 'raw_json') else '',
            parsed_data=extracted_data,
            processing_time_ms=extraction_result.processing_time_ms if hasattr(extraction_result, 'processing_time_ms') else 0
        )

        await session_service.create_parsed_row(
            session_id=session_id,
            row_number=current_row,
            transcription=transcription_text,
            llm_response=llm_response_info,
            final_data=extracted_data
        )

        return {
            "success": True,
            "session_id": session_id,
            "row_number": current_row,
            "transcription": transcription_text,
            "extracted_data": extracted_data,
            "headers": headers,
            "transcription_confidence": transcription_result.confidence,
            "processing_time_ms": {
                "whisper": transcription_result.processing_time_ms,
                "llm": extraction_result.processing_time_ms if hasattr(extraction_result, 'processing_time_ms') else 0
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        await session_service.log_error(
            session_id=session_id,
            error_type=ErrorType.SYSTEM,
            error_message=str(e),
            stack_trace=traceback.format_exc(),
            context=ErrorContext(row_number=current_row)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {e}"
        )


def _has_row_separators(text: str) -> bool:
    """Check if text contains row separator keywords."""
    text_lower = text.lower()
    for keyword in ROW_SEPARATOR_KEYWORDS:
        if keyword in text_lower:
            return True
    return False


@router.post("/process-multi/{session_id}")
async def process_audio_multi_row(
    session_id: str,
    file: UploadFile = File(...),
    start_row: Optional[int] = Form(None)
):
    """
    Process an audio file that may contain multiple rows of data.

    The user can say keywords like "السطر التالي" or "صف جديد" to indicate
    they're moving to a new row.

    Args:
        session_id: Session ID
        file: Audio file (wav, mp3, webm, etc.)
        start_row: Starting row number (uses current_row if not provided)

    Returns:
        List of transcriptions and extracted data for each row
    """
    session_service = get_session_service()
    whisper_service = get_whisper_service()
    llm_service = get_llm_service()

    # Get session and validate
    session = await session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if not session.excel_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Excel file uploaded. Please upload an Excel file first."
        )

    # Determine starting row number
    current_row = start_row if start_row else session.excel_file.current_row
    headers = session.excel_file.headers

    try:
        # 1. Save audio file
        audio_path = await _save_audio_file(session_id, current_row, file)

        # Create audio log
        await session_service.create_audio_log(
            session_id=session_id,
            row_number=current_row,
            audio_file_path=audio_path
        )

        # 2. Transcribe with Whisper
        try:
            transcription_result = await whisper_service.transcribe_async(audio_path)
            transcription_text = transcription_result.text

            # Update audio log with transcription
            await session_service.update_audio_transcription(
                session_id=session_id,
                row_number=current_row,
                text=transcription_text,
                confidence=transcription_result.confidence,
                processing_time_ms=transcription_result.processing_time_ms
            )

        except WhisperError as e:
            await session_service.log_error(
                session_id=session_id,
                error_type=ErrorType.WHISPER,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                context=ErrorContext(
                    row_number=current_row,
                    audio_file=audio_path
                )
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {e}"
            )

        # 3. Check if multi-row or single-row
        is_multi_row = _has_row_separators(transcription_text)

        if is_multi_row:
            # Extract data for multiple rows
            try:
                extraction_result = await llm_service.extract_multi_row_data(
                    transcription=transcription_text,
                    headers=headers
                )

                if not extraction_result.success:
                    await session_service.log_error(
                        session_id=session_id,
                        error_type=ErrorType.LLM,
                        error_message=extraction_result.error_message or "Unknown LLM error",
                        context=ErrorContext(
                            row_number=current_row,
                            transcription=transcription_text
                        )
                    )

                rows_data = extraction_result.rows

            except LLMError as e:
                await session_service.log_error(
                    session_id=session_id,
                    error_type=ErrorType.LLM,
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    context=ErrorContext(
                        row_number=current_row,
                        transcription=transcription_text
                    )
                )
                rows_data = [{header: None for header in headers}]
                extraction_result = MultiRowExtractionResult(
                    raw_json='',
                    rows=rows_data,
                    processing_time_ms=0,
                    success=False
                )

            # Create parsed rows for each extracted row
            results = []
            for i, row_data in enumerate(rows_data):
                row_num = current_row + i

                llm_response_info = LLMResponseInfo(
                    raw_json=extraction_result.raw_json if i == 0 else '',
                    parsed_data=row_data,
                    processing_time_ms=extraction_result.processing_time_ms if i == 0 else 0
                )

                await session_service.create_parsed_row(
                    session_id=session_id,
                    row_number=row_num,
                    transcription=transcription_text if i == 0 else f"(من نفس التسجيل - صف {i + 1})",
                    llm_response=llm_response_info,
                    final_data=row_data
                )

                results.append({
                    "row_number": row_num,
                    "extracted_data": row_data
                })

            return {
                "success": True,
                "session_id": session_id,
                "multi_row": True,
                "start_row": current_row,
                "transcription": transcription_text,
                "rows": results,
                "total_rows": len(results),
                "headers": headers,
                "transcription_confidence": transcription_result.confidence,
                "processing_time_ms": {
                    "whisper": transcription_result.processing_time_ms,
                    "llm": extraction_result.processing_time_ms
                }
            }

        else:
            # Single row - use original extraction
            try:
                extraction_result = await llm_service.extract_data(
                    transcription=transcription_text,
                    headers=headers
                )

                if not extraction_result.success:
                    await session_service.log_error(
                        session_id=session_id,
                        error_type=ErrorType.LLM,
                        error_message=extraction_result.error_message or "Unknown LLM error",
                        context=ErrorContext(
                            row_number=current_row,
                            transcription=transcription_text
                        )
                    )

                extracted_data = extraction_result.parsed_data

            except LLMError as e:
                await session_service.log_error(
                    session_id=session_id,
                    error_type=ErrorType.LLM,
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    context=ErrorContext(
                        row_number=current_row,
                        transcription=transcription_text
                    )
                )
                extracted_data = {header: None for header in headers}
                extraction_result = type('obj', (object,), {
                    'raw_json': '',
                    'processing_time_ms': 0
                })()

            # Create parsed row
            llm_response_info = LLMResponseInfo(
                raw_json=extraction_result.raw_json if hasattr(extraction_result, 'raw_json') else '',
                parsed_data=extracted_data,
                processing_time_ms=extraction_result.processing_time_ms if hasattr(extraction_result, 'processing_time_ms') else 0
            )

            await session_service.create_parsed_row(
                session_id=session_id,
                row_number=current_row,
                transcription=transcription_text,
                llm_response=llm_response_info,
                final_data=extracted_data
            )

            return {
                "success": True,
                "session_id": session_id,
                "multi_row": False,
                "start_row": current_row,
                "transcription": transcription_text,
                "rows": [{
                    "row_number": current_row,
                    "extracted_data": extracted_data
                }],
                "total_rows": 1,
                "headers": headers,
                "transcription_confidence": transcription_result.confidence,
                "processing_time_ms": {
                    "whisper": transcription_result.processing_time_ms,
                    "llm": extraction_result.processing_time_ms if hasattr(extraction_result, 'processing_time_ms') else 0
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        await session_service.log_error(
            session_id=session_id,
            error_type=ErrorType.SYSTEM,
            error_message=str(e),
            stack_trace=traceback.format_exc(),
            context=ErrorContext(row_number=current_row)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {e}"
        )


@router.post("/transcribe/{session_id}")
async def transcribe_only(
    session_id: str,
    file: UploadFile = File(...)
):
    """
    Transcribe audio without data extraction.
    Useful for testing or when only transcription is needed.

    Args:
        session_id: Session ID
        file: Audio file

    Returns:
        Transcription text
    """
    session_service = get_session_service()
    whisper_service = get_whisper_service()

    session = await session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    try:
        audio_path = await _save_audio_file(session_id, 0, file, temp=True)
        result = await whisper_service.transcribe_async(audio_path)

        # Clean up temp file
        try:
            os.remove(audio_path)
        except:
            pass

        return {
            "success": True,
            "transcription": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "processing_time_ms": result.processing_time_ms
        }

    except WhisperError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {e}"
        )


async def _save_audio_file(
    session_id: str,
    row_number: int,
    file: UploadFile,
    temp: bool = False
) -> str:
    """
    Save uploaded audio file to storage.

    Args:
        session_id: Session ID
        row_number: Row number
        file: Uploaded file
        temp: If True, save to temp location

    Returns:
        Path to saved file
    """
    # Create directory
    if temp:
        audio_dir = Path(settings.audio_storage_path) / "temp"
    else:
        audio_dir = Path(settings.audio_storage_path) / session_id

    audio_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    ext = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    filename = f"row_{row_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    file_path = audio_dir / filename

    # Save file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return str(file_path)


@router.get("/health")
async def check_audio_services():
    """
    Check health of audio processing services.

    Returns:
        Service status
    """
    whisper_service = get_whisper_service()
    llm_service = get_llm_service()

    whisper_loaded = whisper_service.is_loaded()
    llm_healthy = await llm_service.check_health()

    return {
        "whisper": {
            "loaded": whisper_loaded,
            "info": whisper_service.get_model_info()
        },
        "llm": {
            "healthy": llm_healthy,
            "info": llm_service.get_service_info()
        },
        "overall_healthy": whisper_loaded and llm_healthy
    }
