"""
PaddleOCR Engine - OCR using PaddlePaddle framework
"""

from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

from .base import BaseOCR, OCRResult, TextBlock, BoundingBox


class PaddleOCREngine(BaseOCR):
    """OCR Engine using PaddleOCR"""
    
    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        use_angle_cls: bool = True,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        cls_model_dir: Optional[str] = None,
    ):
        """
        Initialize PaddleOCR engine
        
        Args:
            languages: List of language codes
            use_gpu: Whether to use GPU
            use_angle_cls: Whether to use angle classification
            det_model_dir: Path to detection model
            rec_model_dir: Path to recognition model
            cls_model_dir: Path to classification model
        """
        super().__init__(languages, use_gpu)
        self.use_angle_cls = use_angle_cls
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.cls_model_dir = cls_model_dir
        
    def _initialize_model(self):
        """Initialize PaddleOCR model"""
        try:
            from paddleocr import PaddleOCR  # type: ignore
            
            # Map language codes
            lang = self._map_language(self.languages[0] if self.languages else "en")
            
            self._model = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=lang,
                use_gpu=self.use_gpu,
                det_model_dir=self.det_model_dir,
                rec_model_dir=self.rec_model_dir,
                cls_model_dir=self.cls_model_dir,
                show_log=False,
            )
            logger.info(f"PaddleOCR initialized with language: {lang}")
            
        except ImportError:
            raise ImportError(
                "PaddleOCR is not installed. "
                "Install with: pip install paddlepaddle paddleocr"
            )
    
    def _map_language(self, lang: str) -> str:
        """Map language code to PaddleOCR format"""
        lang_map = {
            "en": "en",
            "vi": "vi",
            "zh": "ch",
            "ja": "japan",
            "ko": "korean",
            "fr": "fr",
            "de": "german",
        }
        return lang_map.get(lang, "en")
    
    def _process_image(self, image: np.ndarray) -> OCRResult:
        """
        Process image using PaddleOCR
        
        Args:
            image: Image as numpy array (BGR format)
            
        Returns:
            OCRResult object
        """
        result = self._model.ocr(image, cls=self.use_angle_cls)
        
        text_blocks = []
        full_text_parts = []
        
        # PaddleOCR returns list of results per image
        if result and result[0]:
            for line in result[0]:
                # line format: [polygon_points, (text, confidence)]
                polygon = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                bbox = BoundingBox.from_points(polygon)
                
                text_block = TextBlock(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    polygon=polygon
                )
                text_blocks.append(text_block)
                full_text_parts.append(text)
        
        return OCRResult(
            text_blocks=text_blocks,
            full_text=" ".join(full_text_parts)
        )
