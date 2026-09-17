"""
FastAPI Application Setup
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)

from ..core import get_config
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting Invoice Parser API...")
    config = get_config()
    logger.info(f"OCR Engine: {config.ocr.engine}")
    logger.info(f"GPU Enabled: {config.ocr.use_gpu}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Invoice Parser API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    config = get_config()
    
    app = FastAPI(
        title="Invoice Parser API",
        description="AI-powered Invoice/Receipt Parser using OCR and Document Understanding",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router, prefix="/api/v1")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": config.version}
    
    @app.get("/")
    async def root():
        return {
            "name": config.name,
            "version": config.version,
            "docs": "/docs",
        }
    
    return app


# Create default app instance
app = create_app()
