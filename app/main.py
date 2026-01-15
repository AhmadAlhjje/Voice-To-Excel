"""
Voice To Excel - Arabic Offline Data Entry System
Main FastAPI Application
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

from app.config import settings
from app.db.mongodb import init_db, close_db
from app.api.routes import excel_router, audio_router, session_router, rows_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Voice To Excel application...")

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Create storage directories
    settings.audio_storage_path.mkdir(parents=True, exist_ok=True)
    settings.excel_storage_path.mkdir(parents=True, exist_ok=True)
    logger.info("Storage directories ready")

    yield

    # Shutdown
    logger.info("Shutting down Voice To Excel application...")
    await close_db()


# Create FastAPI application
app = FastAPI(
    title="Voice To Excel",
    description="Arabic Offline Data Entry System - نظام إدخال بيانات صوتي عربي",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(session_router, prefix=settings.api_prefix)
app.include_router(excel_router, prefix=settings.api_prefix)
app.include_router(audio_router, prefix=settings.api_prefix)
app.include_router(rows_router, prefix=settings.api_prefix)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - Application info."""
    return {
        "name": "Voice To Excel",
        "description": "نظام إدخال بيانات صوتي عربي يعمل Offline",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "api_prefix": settings.api_prefix
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.services.whisper_service import get_whisper_service
    from app.services.llm_service import get_llm_service

    # Check services
    whisper_service = get_whisper_service()
    llm_service = get_llm_service()

    whisper_status = whisper_service.is_loaded()
    llm_status = await llm_service.check_health()

    return {
        "status": "healthy" if (whisper_status and llm_status) else "degraded",
        "services": {
            "mongodb": "connected",  # If we got here, DB is connected
            "whisper": "loaded" if whisper_status else "not_loaded",
            "ollama": "available" if llm_status else "unavailable"
        },
        "config": {
            "whisper_model": settings.whisper_model,
            "llm_model": settings.ollama_model
        }
    }


# API info endpoint
@app.get(f"{settings.api_prefix}/info")
async def api_info():
    """API information endpoint."""
    return {
        "api_version": "v1",
        "endpoints": {
            "sessions": f"{settings.api_prefix}/sessions",
            "excel": f"{settings.api_prefix}/excel",
            "audio": f"{settings.api_prefix}/audio",
            "rows": f"{settings.api_prefix}/rows"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
