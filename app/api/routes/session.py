"""
Session management API routes.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Optional

from app.models.session import (
    SessionCreate, SessionUpdate, SessionResponse,
    SessionSettings, SessionStatus
)
from app.services.session_service import get_session_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("/", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_session(request: Optional[SessionCreate] = None):
    """
    Create a new data entry session.

    Returns:
        New session details including session_id
    """
    service = get_session_service()

    settings = request.settings if request else None
    session = await service.create_session(settings)

    return {
        "session_id": session.session_id,
        "status": session.status,
        "created_at": session.created_at.isoformat(),
        "message": "Session created successfully"
    }


@router.get("/{session_id}")
async def get_session(session_id: str):
    """
    Get session details by ID.

    Args:
        session_id: Session ID

    Returns:
        Session details
    """
    service = get_session_service()
    session = await service.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    return {
        "session_id": session.session_id,
        "status": session.status,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "excel_file": session.excel_file.model_dump() if session.excel_file else None,
        "settings": session.settings.model_dump()
    }


@router.patch("/{session_id}")
async def update_session(session_id: str, update: SessionUpdate):
    """
    Update session settings or status.

    Args:
        session_id: Session ID
        update: Update data

    Returns:
        Updated session details
    """
    service = get_session_service()

    session = await service.update_session(session_id, update)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    return {
        "session_id": session.session_id,
        "status": session.status,
        "message": "Session updated successfully"
    }


@router.post("/{session_id}/complete")
async def complete_session(session_id: str):
    """
    Mark a session as completed.

    Args:
        session_id: Session ID

    Returns:
        Completion status
    """
    service = get_session_service()

    session = await service.update_session(
        session_id,
        SessionUpdate(status=SessionStatus.COMPLETED)
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    return {
        "session_id": session_id,
        "status": "completed",
        "message": "Session completed successfully"
    }


@router.get("/{session_id}/stats")
async def get_session_stats(session_id: str):
    """
    Get statistics for a session.

    Args:
        session_id: Session ID

    Returns:
        Session statistics
    """
    service = get_session_service()

    stats = await service.get_session_stats(session_id)

    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    return stats


@router.get("/{session_id}/errors")
async def get_session_errors(session_id: str, unresolved_only: bool = True):
    """
    Get errors for a session.

    Args:
        session_id: Session ID
        unresolved_only: Only return unresolved errors

    Returns:
        List of errors
    """
    service = get_session_service()

    errors = await service.get_session_errors(session_id, unresolved_only)

    return {
        "session_id": session_id,
        "errors": [
            {
                "error_type": e.error_type,
                "error_message": e.error_message,
                "created_at": e.created_at.isoformat(),
                "context": e.context.model_dump() if e.context else None,
                "resolved": e.resolved
            }
            for e in errors
        ]
    }
