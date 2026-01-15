"""
Excel file management API routes.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse
from typing import Optional
import os

from app.services.excel_service import get_excel_service, ExcelError
from app.services.session_service import get_session_service
from app.models.error_log import ErrorType, ErrorContext

router = APIRouter(prefix="/excel", tags=["excel"])


@router.post("/upload/{session_id}")
async def upload_excel(
    session_id: str,
    file: UploadFile = File(...)
):
    """
    Upload an Excel file for a session.

    Args:
        session_id: Session ID
        file: Excel file to upload

    Returns:
        File info including headers
    """
    session_service = get_session_service()
    excel_service = get_excel_service()

    # Verify session exists
    session = await session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Validate file type
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )

    valid_extensions = [".xlsx", ".xls"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(valid_extensions)}"
        )

    try:
        # Read file content
        content = await file.read()

        # Save file and extract info
        stored_path, headers, total_rows = excel_service.save_uploaded_file(
            file_content=content,
            original_filename=file.filename,
            session_id=session_id
        )

        # Update session with file info
        await session_service.set_excel_file(
            session_id=session_id,
            original_name=file.filename,
            stored_path=stored_path,
            headers=headers,
            total_rows=total_rows
        )

        return {
            "message": "File uploaded successfully",
            "session_id": session_id,
            "filename": file.filename,
            "headers": headers,
            "total_rows": total_rows
        }

    except ExcelError as e:
        await session_service.log_error(
            session_id=session_id,
            error_type=ErrorType.EXCEL,
            error_message=str(e),
            context=ErrorContext(additional_info=f"Upload: {file.filename}")
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/headers/{session_id}")
async def get_headers(session_id: str):
    """
    Get column headers for a session's Excel file.

    Args:
        session_id: Session ID

    Returns:
        List of column headers
    """
    session_service = get_session_service()

    session = await session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if not session.excel_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Excel file uploaded for this session"
        )

    return {
        "session_id": session_id,
        "headers": session.excel_file.headers,
        "total_rows": session.excel_file.total_rows,
        "current_row": session.excel_file.current_row
    }


@router.get("/download/{session_id}")
async def download_excel(session_id: str):
    """
    Download the Excel file for a session.

    Args:
        session_id: Session ID

    Returns:
        Excel file download
    """
    session_service = get_session_service()

    session = await session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if not session.excel_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Excel file uploaded for this session"
        )

    file_path = session.excel_file.stored_path

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on server"
        )

    return FileResponse(
        path=file_path,
        filename=session.excel_file.original_name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@router.post("/backup/{session_id}")
async def create_backup(session_id: str):
    """
    Create a backup of the Excel file.

    Args:
        session_id: Session ID

    Returns:
        Backup file path
    """
    session_service = get_session_service()
    excel_service = get_excel_service()

    session = await session_service.get_session(session_id)
    if not session or not session.excel_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session or file not found"
        )

    try:
        backup_path = excel_service.create_backup(session.excel_file.stored_path)
        return {
            "message": "Backup created successfully",
            "backup_path": backup_path
        }
    except ExcelError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/preview/{session_id}")
async def preview_data(session_id: str, limit: int = 10):
    """
    Preview data from the Excel file.

    Args:
        session_id: Session ID
        limit: Number of rows to preview

    Returns:
        Headers and sample data rows
    """
    session_service = get_session_service()
    excel_service = get_excel_service()

    session = await session_service.get_session(session_id)
    if not session or not session.excel_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session or file not found"
        )

    try:
        headers, rows = excel_service.get_all_data(session.excel_file.stored_path)
        return {
            "headers": headers,
            "rows": rows[:limit],
            "total_rows": len(rows)
        }
    except ExcelError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
