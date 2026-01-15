"""
Session Service for managing data entry sessions.
Handles session lifecycle, state management, and data coordination.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from app.db.mongodb import (
    get_sessions_collection,
    get_audio_logs_collection,
    get_parsed_rows_collection,
    get_error_logs_collection
)
from app.models.session import (
    Session, SessionCreate, SessionUpdate, SessionStatus,
    ExcelFileInfo, SessionSettings
)
from app.models.audio_log import AudioLog, AudioLogCreate, AudioStatus, TranscriptionInfo
from app.models.parsed_row import ParsedRow, ParsedRowCreate, ParsedRowUpdate, RowStatus, LLMResponseInfo
from app.models.error_log import ErrorLog, ErrorLogCreate, ErrorType, ErrorContext

logger = logging.getLogger(__name__)


class SessionService:
    """
    Service for managing voice-to-excel sessions.
    Coordinates all session-related operations.
    """

    async def create_session(
        self,
        settings: Optional[SessionSettings] = None
    ) -> Session:
        """
        Create a new session.

        Args:
            settings: Optional session settings

        Returns:
            Created session
        """
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()

        session_data = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "status": SessionStatus.ACTIVE.value,
            "excel_file": None,
            "settings": (settings or SessionSettings()).model_dump()
        }

        collection = get_sessions_collection()
        await collection.insert_one(session_data)

        logger.info(f"Created session: {session_id}")
        return Session(**session_data)

    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session or None if not found
        """
        collection = get_sessions_collection()
        doc = await collection.find_one({"session_id": session_id})

        if doc:
            return Session(**doc)
        return None

    async def update_session(
        self,
        session_id: str,
        update: SessionUpdate
    ) -> Optional[Session]:
        """
        Update a session.

        Args:
            session_id: Session ID
            update: Update data

        Returns:
            Updated session or None
        """
        collection = get_sessions_collection()

        update_data = {"updated_at": datetime.utcnow()}

        if update.status is not None:
            update_data["status"] = update.status.value

        if update.current_row is not None:
            update_data["excel_file.current_row"] = update.current_row

        if update.settings is not None:
            update_data["settings"] = update.settings.model_dump()

        result = await collection.find_one_and_update(
            {"session_id": session_id},
            {"$set": update_data},
            return_document=True
        )

        if result:
            logger.info(f"Updated session: {session_id}")
            return Session(**result)
        return None

    async def set_excel_file(
        self,
        session_id: str,
        original_name: str,
        stored_path: str,
        headers: List[str],
        total_rows: int = 0
    ) -> Optional[Session]:
        """
        Set the Excel file for a session.

        Args:
            session_id: Session ID
            original_name: Original filename
            stored_path: Path where file is stored
            headers: Column headers
            total_rows: Total number of data rows

        Returns:
            Updated session
        """
        collection = get_sessions_collection()

        excel_file = ExcelFileInfo(
            original_name=original_name,
            stored_path=stored_path,
            headers=headers,
            total_rows=total_rows,
            current_row=1
        )

        result = await collection.find_one_and_update(
            {"session_id": session_id},
            {
                "$set": {
                    "excel_file": excel_file.model_dump(),
                    "updated_at": datetime.utcnow()
                }
            },
            return_document=True
        )

        if result:
            logger.info(f"Excel file set for session: {session_id}")
            return Session(**result)
        return None

    async def advance_row(self, session_id: str) -> Optional[int]:
        """
        Advance to the next row in the session.

        Args:
            session_id: Session ID

        Returns:
            New current row number or None
        """
        session = await self.get_session(session_id)
        if not session or not session.excel_file:
            return None

        new_row = session.excel_file.current_row + 1
        await self.update_session(
            session_id,
            SessionUpdate(current_row=new_row)
        )

        logger.info(f"Advanced to row {new_row} in session {session_id}")
        return new_row

    # Audio Log Methods
    async def create_audio_log(
        self,
        session_id: str,
        row_number: int,
        audio_file_path: str,
        duration_seconds: float = 0.0
    ) -> AudioLog:
        """Create an audio log entry."""
        collection = get_audio_logs_collection()

        log_data = {
            "session_id": session_id,
            "row_number": row_number,
            "created_at": datetime.utcnow(),
            "audio_file_path": audio_file_path,
            "duration_seconds": duration_seconds,
            "transcription": None,
            "status": AudioStatus.PENDING.value
        }

        await collection.insert_one(log_data)
        return AudioLog(**log_data)

    async def update_audio_transcription(
        self,
        session_id: str,
        row_number: int,
        text: str,
        confidence: float,
        processing_time_ms: int
    ) -> Optional[AudioLog]:
        """Update audio log with transcription result."""
        collection = get_audio_logs_collection()

        transcription = TranscriptionInfo(
            text=text,
            confidence=confidence,
            processing_time_ms=processing_time_ms
        )

        result = await collection.find_one_and_update(
            {"session_id": session_id, "row_number": row_number},
            {
                "$set": {
                    "transcription": transcription.model_dump(),
                    "status": AudioStatus.TRANSCRIBED.value
                }
            },
            return_document=True
        )

        if result:
            return AudioLog(**result)
        return None

    # Parsed Row Methods
    async def create_parsed_row(
        self,
        session_id: str,
        row_number: int,
        transcription: str,
        llm_response: Optional[LLMResponseInfo] = None,
        final_data: Optional[Dict[str, Any]] = None
    ) -> ParsedRow:
        """Create or update a parsed row entry."""
        collection = get_parsed_rows_collection()
        now = datetime.utcnow()

        row_data = {
            "session_id": session_id,
            "row_number": row_number,
            "created_at": now,
            "updated_at": now,
            "original_transcription": transcription,
            "llm_response": llm_response.model_dump() if llm_response else None,
            "final_data": final_data or {},
            "status": RowStatus.DRAFT.value,
            "written_to_excel": False
        }

        # Upsert to handle re-recording
        await collection.update_one(
            {"session_id": session_id, "row_number": row_number},
            {"$set": row_data},
            upsert=True
        )

        return ParsedRow(**row_data)

    async def get_parsed_row(
        self,
        session_id: str,
        row_number: int
    ) -> Optional[ParsedRow]:
        """Get a parsed row."""
        collection = get_parsed_rows_collection()
        doc = await collection.find_one({
            "session_id": session_id,
            "row_number": row_number
        })

        if doc:
            return ParsedRow(**doc)
        return None

    async def update_parsed_row(
        self,
        session_id: str,
        row_number: int,
        update: ParsedRowUpdate
    ) -> Optional[ParsedRow]:
        """Update a parsed row."""
        collection = get_parsed_rows_collection()

        update_data = {"updated_at": datetime.utcnow()}

        if update.final_data is not None:
            update_data["final_data"] = update.final_data

        if update.status is not None:
            update_data["status"] = update.status.value

        result = await collection.find_one_and_update(
            {"session_id": session_id, "row_number": row_number},
            {"$set": update_data},
            return_document=True
        )

        if result:
            return ParsedRow(**result)
        return None

    async def confirm_row(
        self,
        session_id: str,
        row_number: int,
        final_data: Dict[str, Any]
    ) -> Optional[ParsedRow]:
        """Confirm a row and prepare for writing."""
        return await self.update_parsed_row(
            session_id,
            row_number,
            ParsedRowUpdate(
                final_data=final_data,
                status=RowStatus.CONFIRMED
            )
        )

    async def mark_row_written(
        self,
        session_id: str,
        row_number: int
    ) -> Optional[ParsedRow]:
        """Mark a row as written to Excel."""
        collection = get_parsed_rows_collection()

        result = await collection.find_one_and_update(
            {"session_id": session_id, "row_number": row_number},
            {
                "$set": {
                    "status": RowStatus.WRITTEN.value,
                    "written_to_excel": True,
                    "updated_at": datetime.utcnow()
                }
            },
            return_document=True
        )

        if result:
            return ParsedRow(**result)
        return None

    # Error Logging Methods
    async def log_error(
        self,
        session_id: str,
        error_type: ErrorType,
        error_message: str,
        stack_trace: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ) -> ErrorLog:
        """Log an error."""
        collection = get_error_logs_collection()

        error_data = {
            "session_id": session_id,
            "created_at": datetime.utcnow(),
            "error_type": error_type.value,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "context": context.model_dump() if context else None,
            "resolved": False
        }

        await collection.insert_one(error_data)
        logger.error(f"Error logged for session {session_id}: {error_type.value} - {error_message}")

        return ErrorLog(**error_data)

    async def get_session_errors(
        self,
        session_id: str,
        unresolved_only: bool = True
    ) -> List[ErrorLog]:
        """Get errors for a session."""
        collection = get_error_logs_collection()

        query = {"session_id": session_id}
        if unresolved_only:
            query["resolved"] = False

        cursor = collection.find(query).sort("created_at", -1)
        errors = []
        async for doc in cursor:
            errors.append(ErrorLog(**doc))

        return errors

    # Statistics
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        session = await self.get_session(session_id)
        if not session:
            return {}

        parsed_collection = get_parsed_rows_collection()
        error_collection = get_error_logs_collection()

        # Count rows by status
        draft_count = await parsed_collection.count_documents({
            "session_id": session_id,
            "status": RowStatus.DRAFT.value
        })
        confirmed_count = await parsed_collection.count_documents({
            "session_id": session_id,
            "status": RowStatus.CONFIRMED.value
        })
        written_count = await parsed_collection.count_documents({
            "session_id": session_id,
            "status": RowStatus.WRITTEN.value
        })

        # Count errors
        error_count = await error_collection.count_documents({
            "session_id": session_id,
            "resolved": False
        })

        return {
            "session_id": session_id,
            "status": session.status,
            "current_row": session.excel_file.current_row if session.excel_file else 0,
            "total_rows": session.excel_file.total_rows if session.excel_file else 0,
            "rows_draft": draft_count,
            "rows_confirmed": confirmed_count,
            "rows_written": written_count,
            "unresolved_errors": error_count
        }


# Global service instance
_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    """Get or create the session service instance."""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
