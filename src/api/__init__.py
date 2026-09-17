"""
FastAPI Application - REST API for Invoice Parser
"""

from .app import app, create_app
from .routes import router

__all__ = ["app", "create_app", "router"]
