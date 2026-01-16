"""
Row management API routes.
Handles row confirmation, editing, and writing to Excel.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, Optional

from app.services.session_service import get_session_service
from app.services.excel_service import get_excel_service, ExcelError
from app.services.llm_service import get_llm_service
from app.models.parsed_row import ParsedRowUpdate, RowStatus
from app.models.error_log import ErrorType, ErrorContext

router = APIRouter(prefix="/rows", tags=["rows"])


class RowUpdateRequest(BaseModel):
    """Request model for updating row data."""
    data: Dict[str, Any]


class RowConfirmRequest(BaseModel):
    """Request model for confirming a row."""
    data: Dict[str, Any]
    auto_advance: bool = True


class CorrectionRequest(BaseModel):
    """Request model for voice correction."""
    correction_text: str


@router.get("/{session_id}")
async def get_all_rows(session_id: str):
    """
    Get all parsed rows for a session.

    Args:
        session_id: Session ID

    Returns:
        List of all rows with their data
    """
    session_service = get_session_service()

    session = await session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    rows = await session_service.get_all_parsed_rows(session_id)

    return {
        "session_id": session_id,
        "headers": session.excel_file.headers if session.excel_file else [],
        "current_row": session.excel_file.current_row if session.excel_file else 1,
        "total_rows": len(rows),
        "rows": [
            {
                "row_number": row.row_number,
                "data": row.final_data,
                "status": row.status,
                "written_to_excel": row.written_to_excel
            }
            for row in rows
        ]
    }


@router.get("/{session_id}/{row_number}")
async def get_row(session_id: str, row_number: int):
    """
    Get parsed row data.

    Args:
        session_id: Session ID
        row_number: Row number

    Returns:
        Row data
    """
    session_service = get_session_service()

    session = await session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    row = await session_service.get_parsed_row(session_id, row_number)

    if not row:
        return {
            "session_id": session_id,
            "row_number": row_number,
            "exists": False,
            "headers": session.excel_file.headers if session.excel_file else []
        }

    return {
        "session_id": session_id,
        "row_number": row_number,
        "exists": True,
        "original_transcription": row.original_transcription,
        "extracted_data": row.llm_response.parsed_data if row.llm_response else {},
        "final_data": row.final_data,
        "status": row.status,
        "written_to_excel": row.written_to_excel,
        "headers": session.excel_file.headers if session.excel_file else []
    }


@router.patch("/{session_id}/{row_number}")
async def update_row(
    session_id: str,
    row_number: int,
    request: RowUpdateRequest
):
    """
    Update row data manually and write to Excel file.

    Args:
        session_id: Session ID
        row_number: Row number
        request: New data

    Returns:
        Updated row data
    """
    session_service = get_session_service()
    excel_service = get_excel_service()

    # Get session to access Excel file info
    session = await session_service.get_session(session_id)
    if not session or not session.excel_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session or Excel file not found"
        )

    # Update in database
    row = await session_service.update_parsed_row(
        session_id,
        row_number,
        ParsedRowUpdate(final_data=request.data)
    )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Row not found"
        )

    try:
        # Write changes to Excel file immediately
        excel_service.write_row(
            file_path=session.excel_file.stored_path,
            row_number=row_number,
            data=request.data,
            headers=session.excel_file.headers
        )

        # Mark as written
        await session_service.mark_row_written(session_id, row_number)

        return {
            "success": True,
            "session_id": session_id,
            "row_number": row_number,
            "final_data": row.final_data,
            "status": "written",
            "written_to_excel": True
        }
    except ExcelError as e:
        await session_service.log_error(
            session_id=session_id,
            error_type=ErrorType.EXCEL,
            error_message=str(e),
            context=ErrorContext(row_number=row_number)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write to Excel: {e}"
        )


@router.post("/{session_id}/{row_number}/confirm")
async def confirm_row(
    session_id: str,
    row_number: int,
    request: RowConfirmRequest
):
    """
    Confirm row data and write to Excel.

    Args:
        session_id: Session ID
        row_number: Row number
        request: Final data and options

    Returns:
        Confirmation result
    """
    session_service = get_session_service()
    excel_service = get_excel_service()

    # Get session
    session = await session_service.get_session(session_id)
    if not session or not session.excel_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session or Excel file not found"
        )

    try:
        # 1. Confirm the row
        row = await session_service.confirm_row(
            session_id,
            row_number,
            request.data
        )

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Row not found"
            )

        # 2. Write to Excel
        excel_service.write_row(
            file_path=session.excel_file.stored_path,
            row_number=row_number,
            data=request.data,
            headers=session.excel_file.headers
        )

        # 3. Mark as written
        await session_service.mark_row_written(session_id, row_number)

        # 4. Advance to next row if auto_advance is enabled
        next_row = None
        if request.auto_advance:
            next_row = await session_service.advance_row(session_id)

        return {
            "success": True,
            "session_id": session_id,
            "row_number": row_number,
            "status": "written",
            "next_row": next_row,
            "message": "Row confirmed and written to Excel"
        }

    except ExcelError as e:
        await session_service.log_error(
            session_id=session_id,
            error_type=ErrorType.EXCEL,
            error_message=str(e),
            context=ErrorContext(row_number=row_number)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write to Excel: {e}"
        )


@router.post("/{session_id}/{row_number}/correct")
async def correct_row_with_voice(
    session_id: str,
    row_number: int,
    request: CorrectionRequest
):
    """
    Correct row data using voice/text instruction.

    Args:
        session_id: Session ID
        row_number: Row number
        request: Correction instruction

    Returns:
        Corrected data
    """
    session_service = get_session_service()
    llm_service = get_llm_service()

    # Get session and row
    session = await session_service.get_session(session_id)
    if not session or not session.excel_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    row = await session_service.get_parsed_row(session_id, row_number)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Row not found"
        )

    try:
        # Use LLM to apply correction
        result = await llm_service.correct_data(
            headers=session.excel_file.headers,
            original_data=row.final_data,
            correction=request.correction_text
        )

        if result.success:
            # Update the row
            updated_row = await session_service.update_parsed_row(
                session_id,
                row_number,
                ParsedRowUpdate(final_data=result.parsed_data)
            )

            return {
                "success": True,
                "session_id": session_id,
                "row_number": row_number,
                "original_data": row.final_data,
                "corrected_data": result.parsed_data,
                "correction_applied": request.correction_text
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Correction failed: {result.error_message}"
            )

    except Exception as e:
        await session_service.log_error(
            session_id=session_id,
            error_type=ErrorType.LLM,
            error_message=str(e),
            context=ErrorContext(
                row_number=row_number,
                additional_info=f"Correction: {request.correction_text}"
            )
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Correction failed: {e}"
        )


@router.delete("/{session_id}/{row_number}")
async def delete_row(session_id: str, row_number: int):
    """
    Delete a parsed row (does not affect Excel).

    Args:
        session_id: Session ID
        row_number: Row number

    Returns:
        Deletion result
    """
    session_service = get_session_service()

    # Just reset the row to draft status with empty data
    row = await session_service.update_parsed_row(
        session_id,
        row_number,
        ParsedRowUpdate(
            final_data={},
            status=RowStatus.DRAFT
        )
    )

    return {
        "success": True,
        "session_id": session_id,
        "row_number": row_number,
        "message": "Row data cleared"
    }


@router.post("/{session_id}/skip")
async def skip_row(session_id: str):
    """
    Skip to the next row without writing data.

    Args:
        session_id: Session ID

    Returns:
        New current row
    """
    session_service = get_session_service()

    next_row = await session_service.advance_row(session_id)

    if next_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or cannot advance"
        )

    return {
        "success": True,
        "session_id": session_id,
        "current_row": next_row,
        "message": f"Skipped to row {next_row}"
    }


@router.post("/{session_id}/goto/{row_number}")
async def goto_row(session_id: str, row_number: int):
    """
    Jump to a specific row.

    Args:
        session_id: Session ID
        row_number: Target row number

    Returns:
        Result
    """
    session_service = get_session_service()
    from app.models.session import SessionUpdate

    session = await session_service.update_session(
        session_id,
        SessionUpdate(current_row=row_number)
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    return {
        "success": True,
        "session_id": session_id,
        "current_row": row_number,
        "message": f"Moved to row {row_number}"
    }
