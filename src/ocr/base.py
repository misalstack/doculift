"""
Base OCR class and common data structures
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image

import numpy as np
from PIL import Image


@dataclass
class BoundingBox:
    """Represents a bounding box for detected text"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_list(self) -> List[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]
    
    @classmethod
    def from_points(cls, points: List[List[float]]) -> "BoundingBox":
        """Create bounding box from polygon points"""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        return cls(
            x_min=min(x_coords),
            y_min=min(y_coords),
            x_max=max(x_coords),
            y_max=max(y_coords)
        )


@dataclass
class TextBlock:
    """Represents a detected text block"""
    text: str
    bbox: BoundingBox
    confidence: float
    polygon: Optional[List[List[float]]] = None
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "polygon": self.polygon
        }


@dataclass
class OCRResult:
    """Complete OCR result for an image"""
    text_blocks: List[TextBlock] = field(default_factory=list)
    full_text: str = ""
    image_size: Tuple[int, int] = (0, 0)  # (width, height)
    processing_time: float = 0.0
    
    @property
    def num_blocks(self) -> int:
        return len(self.text_blocks)
    
    def get_text_by_confidence(self, min_confidence: float = 0.5) -> str:
        """Get concatenated text from blocks above confidence threshold"""
        filtered_blocks = [b for b in self.text_blocks if b.confidence >= min_confidence]
        return " ".join([b.text for b in filtered_blocks])
    
    def to_dict(self) -> dict:
        return {
            "text_blocks": [b.to_dict() for b in self.text_blocks],
            "full_text": self.full_text,
            "image_size": self.image_size,
            "processing_time": self.processing_time,
            "num_blocks": self.num_blocks
        }


class BaseOCR(ABC):
    """Abstract base class for OCR engines"""
    
    def __init__(self, languages: Optional[List[str]] = None, use_gpu: bool = False):
        """
        Initialize OCR engine
        
        Args:
            languages: List of language codes (e.g., ['en', 'vi'])
            use_gpu: Whether to use GPU acceleration
        """
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self._model = None
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the OCR model"""
        pass
    
    @abstractmethod
    def _process_image(self, image: np.ndarray) -> OCRResult:
        """
        Process a single image and return OCR results
        
        Args:
            image: Image as numpy array (BGR format)
            
        Returns:
            OCRResult object containing detected text
        """
        pass
    
    def load_image(self, image_source: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Load image from various sources
        
        Args:
            image_source: Path to image, numpy array, or PIL Image
            
        Returns:
            Image as numpy array (BGR format)
        """
        import cv2
        
        if isinstance(image_source, (str, Path)):
            image = cv2.imread(str(image_source))
            if image is None:
                raise ValueError(f"Could not load image from: {image_source}")
            return image
        elif isinstance(image_source, Image.Image):
            # Convert PIL Image to numpy array (RGB to BGR)
            return cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
        elif isinstance(image_source, np.ndarray):
            return image_source
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")
    
    def process(self, image_source: Union[str, Path, np.ndarray, Image.Image]) -> OCRResult:
        """
        Main method to process an image
        
        Args:
            image_source: Image path, numpy array, or PIL Image
            
        Returns:
            OCRResult object
        """
        import time
        
        # Initialize model if not already done
        if self._model is None:
            self._initialize_model()
        
        # Load and process image
        image = self.load_image(image_source)
        
        start_time = time.time()
        result = self._process_image(image)
        result.processing_time = time.time() - start_time
        result.image_size = (image.shape[1], image.shape[0])
        
        return result
    
    def process_batch(self, image_sources: List[Union[str, Path, np.ndarray]]) -> List[OCRResult]:
        """
        Process multiple images
        
        Args:
            image_sources: List of image sources
            
        Returns:
            List of OCRResult objects
        """
        return [self.process(img) for img in image_sources]
