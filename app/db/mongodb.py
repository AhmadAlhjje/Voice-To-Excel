"""
MongoDB database connection and management.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure
from typing import Optional
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Global database client
_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


async def init_db() -> None:
    """Initialize MongoDB connection."""
    global _client, _database

    try:
        _client = AsyncIOMotorClient(
            settings.mongodb_url,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        # Verify connection
        await _client.admin.command("ping")
        _database = _client[settings.mongodb_database]

        # Create indexes
        await _create_indexes()

        logger.info(f"Connected to MongoDB: {settings.mongodb_database}")
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def _create_indexes() -> None:
    """Create necessary indexes for collections."""
    if _database is None:
        return

    # Sessions collection indexes
    await _database.sessions.create_index("session_id", unique=True)
    await _database.sessions.create_index("status")
    await _database.sessions.create_index("created_at")

    # AudioLogs collection indexes
    await _database.audio_logs.create_index("session_id")
    await _database.audio_logs.create_index([("session_id", 1), ("row_number", 1)])
    await _database.audio_logs.create_index("status")

    # ParsedRows collection indexes
    await _database.parsed_rows.create_index("session_id")
    await _database.parsed_rows.create_index([("session_id", 1), ("row_number", 1)], unique=True)
    await _database.parsed_rows.create_index("status")

    # ErrorLogs collection indexes
    await _database.error_logs.create_index("session_id")
    await _database.error_logs.create_index("error_type")
    await _database.error_logs.create_index("created_at")
    await _database.error_logs.create_index("resolved")

    logger.info("MongoDB indexes created successfully")


async def close_db() -> None:
    """Close MongoDB connection."""
    global _client, _database

    if _client is not None:
        _client.close()
        _client = None
        _database = None
        logger.info("MongoDB connection closed")


def get_database() -> AsyncIOMotorDatabase:
    """Get the database instance."""
    if _database is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _database


# Collection accessors
def get_sessions_collection():
    """Get sessions collection."""
    return get_database().sessions


def get_audio_logs_collection():
    """Get audio logs collection."""
    return get_database().audio_logs


def get_parsed_rows_collection():
    """Get parsed rows collection."""
    return get_database().parsed_rows


def get_error_logs_collection():
    """Get error logs collection."""
    return get_database().error_logs
