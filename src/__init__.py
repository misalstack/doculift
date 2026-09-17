"""
Invoice Parser - AI-powered Invoice/Receipt Parser

Modules:
    - core: Configuration, logging, exceptions
    - models: Data schemas and Pydantic models
    - ocr: OCR engines (PaddleOCR, EasyOCR)
    - extraction: Field extraction and document understanding
    - processing: Image preprocessing and PDF handling
    - api: FastAPI REST API
    - utils: Helper utilities
    - web: Streamlit web interface
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Convenience imports
from src.core import get_config, get_logger, InvoiceParserError
from src.extraction import InvoiceExtractor

__all__ = [
    "__version__",
    "__author__",
    "get_config",
    "get_logger",
    "InvoiceParserError",
    "InvoiceExtractor",
]
