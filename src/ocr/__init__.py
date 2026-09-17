"""
OCR Module - Text extraction from images using various OCR engines
"""

from .base import BaseOCR, OCRResult
from .paddleocr_engine import PaddleOCREngine
from .easyocr_engine import EasyOCREngine
from .ocr_factory import OCRFactory, get_ocr_engine

__all__ = [
    "BaseOCR",
    "OCRResult", 
    "PaddleOCREngine",
    "EasyOCREngine",
    "OCRFactory",
    "get_ocr_engine",
]
