"""
EasyOCR Engine - OCR using EasyOCR library
"""

from typing import List, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

from .base import BaseOCR, OCRResult, TextBlock, BoundingBox


class EasyOCREngine(BaseOCR):
    """OCR Engine using EasyOCR"""
    
    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        model_storage_directory: Optional[str] = None,
        download_enabled: bool = True,
    ):
        """
        Initialize EasyOCR engine
        
        Args:
            languages: List of language codes (e.g., ['en', 'vi'])
            use_gpu: Whether to use GPU
            model_storage_directory: Directory to store models
            download_enabled: Whether to allow model downloads
        """
        super().__init__(languages, use_gpu)
        self.model_storage_directory = model_storage_directory
        self.download_enabled = download_enabled
        
    def _initialize_model(self):
        """Initialize EasyOCR model"""
        try:
            import easyocr  # type: ignore
            
            self._model = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                model_storage_directory=self.model_storage_directory,
                download_enabled=self.download_enabled,
            )
            logger.info(f"EasyOCR initialized with languages: {self.languages}")
            
        except ImportError:
            raise ImportError(
                "EasyOCR is not installed. "
                "Install with: pip install easyocr"
            )
    
    def _process_image(self, image: np.ndarray) -> OCRResult:
        """
        Process image using EasyOCR
        
        Args:
            image: Image as numpy array (BGR format)
            
        Returns:
            OCRResult object
        """
        import cv2
        
        # EasyOCR expects RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run OCR
        results: List[Any] = self._model.readtext(image_rgb)
        
        text_blocks: List[TextBlock] = []
        full_text_parts: List[str] = []
        
        for detection in results:
            # detection format: [polygon_points, text, confidence]
            polygon = detection[0]
            text: str = str(detection[1])
            confidence: float = float(detection[2])
            
            # Convert polygon to proper format
            polygon_list = [[float(p[0]), float(p[1])] for p in polygon]
            bbox = BoundingBox.from_points(polygon_list)
            
            text_block = TextBlock(
                text=text,
                bbox=bbox,
                confidence=confidence,
                polygon=polygon_list
            )
            text_blocks.append(text_block)
            full_text_parts.append(text)
        
        return OCRResult(
            text_blocks=text_blocks,
            full_text=" ".join(full_text_parts)
        )
    
    def process_with_detail(
        self,
        image_source,
        detail: int = 1,
        paragraph: bool = False,
    ) -> OCRResult:
        """
        Process image with additional options
        
        Args:
            image_source: Image source
            detail: Level of detail (0 or 1)
            paragraph: Whether to merge text into paragraphs
            
        Returns:
            OCRResult object
        """
        import cv2
        
        if self._model is None:
            self._initialize_model()
        
        image = self.load_image(image_source)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results: List[Any] = self._model.readtext(
            image_rgb,
            detail=detail,
            paragraph=paragraph
        )
        
        text_blocks: List[TextBlock] = []
        full_text_parts: List[str] = []
        
        if detail == 0:
            # Simple mode - just text strings
            full_text_parts = [str(r) for r in results]
        else:
            for detection in results:
                polygon = detection[0]
                text: str = str(detection[1])
                confidence: float = float(detection[2]) if len(detection) > 2 else 1.0
                
                polygon_list = [[float(p[0]), float(p[1])] for p in polygon]
                bbox = BoundingBox.from_points(polygon_list)
                
                text_block = TextBlock(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    polygon=polygon_list
                )
                text_blocks.append(text_block)
                full_text_parts.append(text)
        
        return OCRResult(
            text_blocks=text_blocks,
            full_text=" ".join(full_text_parts),
            image_size=(image.shape[1], image.shape[0])
        )
