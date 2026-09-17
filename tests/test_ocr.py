"""
Tests for OCR module
"""

import pytest
import numpy as np
from pathlib import Path

# Skip tests if dependencies not installed
pytest.importorskip("cv2")


class TestBaseOCR:
    """Tests for base OCR classes"""
    
    def test_bounding_box_creation(self):
        from src.ocr.base import BoundingBox
        
        bbox = BoundingBox(x_min=10, y_min=20, x_max=100, y_max=80)
        
        assert bbox.width == 90
        assert bbox.height == 60
        assert bbox.center == (55, 50)
        assert bbox.area == 5400
    
    def test_bounding_box_from_points(self):
        from src.ocr.base import BoundingBox
        
        points = [[10, 20], [100, 20], [100, 80], [10, 80]]
        bbox = BoundingBox.from_points(points)
        
        assert bbox.x_min == 10
        assert bbox.y_min == 20
        assert bbox.x_max == 100
        assert bbox.y_max == 80
    
    def test_text_block_creation(self):
        from src.ocr.base import TextBlock, BoundingBox
        
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
        block = TextBlock(text="Hello", bbox=bbox, confidence=0.95)
        
        assert block.text == "Hello"
        assert block.confidence == 0.95
        
        d = block.to_dict()
        assert d["text"] == "Hello"
        assert d["confidence"] == 0.95
    
    def test_ocr_result(self):
        from src.ocr.base import OCRResult, TextBlock, BoundingBox
        
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
        blocks = [
            TextBlock(text="Hello", bbox=bbox, confidence=0.9),
            TextBlock(text="World", bbox=bbox, confidence=0.3),
        ]
        
        result = OCRResult(text_blocks=blocks, full_text="Hello World")
        
        assert result.num_blocks == 2
        assert result.get_text_by_confidence(0.5) == "Hello"


class TestOCRFactory:
    """Tests for OCR factory"""
    
    def test_get_available_engines(self):
        from src.ocr import OCRFactory
        
        engines = OCRFactory.get_available_engines()
        
        assert "paddleocr" in engines
        assert "easyocr" in engines
    
    def test_create_invalid_engine(self):
        from src.ocr import OCRFactory
        
        with pytest.raises(ValueError, match="Unknown OCR engine"):
            OCRFactory.create("invalid_engine")
