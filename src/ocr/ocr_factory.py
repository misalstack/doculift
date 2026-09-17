"""
OCR Factory - Create OCR engines based on configuration
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

from .base import BaseOCR
from .paddleocr_engine import PaddleOCREngine
from .easyocr_engine import EasyOCREngine


class OCRFactory:
    """Factory class for creating OCR engines"""
    
    _engines = {
        "paddleocr": PaddleOCREngine,
        "easyocr": EasyOCREngine,
    }
    
    @classmethod
    def register_engine(cls, name: str, engine_class: type):
        """Register a new OCR engine"""
        cls._engines[name.lower()] = engine_class
        
    @classmethod
    def get_available_engines(cls) -> List[str]:
        """Get list of available OCR engines"""
        return list(cls._engines.keys())
    
    @classmethod
    def create(
        cls,
        engine_name: str = "paddleocr",
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        **kwargs
    ) -> BaseOCR:
        """
        Create an OCR engine instance
        
        Args:
            engine_name: Name of the OCR engine
            languages: List of language codes
            use_gpu: Whether to use GPU
            **kwargs: Additional engine-specific arguments
            
        Returns:
            OCR engine instance
        """
        engine_name = engine_name.lower()
        
        if engine_name not in cls._engines:
            available = ", ".join(cls.get_available_engines())
            raise ValueError(
                f"Unknown OCR engine: {engine_name}. "
                f"Available engines: {available}"
            )
        
        engine_class = cls._engines[engine_name]
        logger.info(f"Creating OCR engine: {engine_name}")
        
        return engine_class(
            languages=languages or ["en"],
            use_gpu=use_gpu,
            **kwargs
        )


def get_ocr_engine(
    engine_name: str = "paddleocr",
    languages: Optional[List[str]] = None,
    use_gpu: bool = False,
    **kwargs
) -> BaseOCR:
    """
    Convenience function to get an OCR engine
    
    Args:
        engine_name: Name of the OCR engine
        languages: List of language codes
        use_gpu: Whether to use GPU
        **kwargs: Additional engine-specific arguments
        
    Returns:
        OCR engine instance
    """
    return OCRFactory.create(
        engine_name=engine_name,
        languages=languages,
        use_gpu=use_gpu,
        **kwargs
    )
