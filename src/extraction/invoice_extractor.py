"""
Invoice Extractor - Main class for extracting structured data from invoices
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import date
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

from ..ocr import get_ocr_engine, OCRResult
from ..processing import ImagePreprocessor, TextPostprocessor, PDFHandler
from .field_extractor import FieldExtractor


@dataclass
class LineItem:
    """Represents a line item in an invoice"""
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InvoiceData:
    """Structured invoice data"""
    # Vendor information
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    vendor_tax_id: Optional[str] = None
    
    # Invoice details
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    
    # Customer information
    customer_name: Optional[str] = None
    customer_address: Optional[str] = None
    
    # Financial information
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    tax_rate: Optional[float] = None
    discount: Optional[float] = None
    total: Optional[float] = None
    currency: str = "USD"
    
    # Line items
    line_items: List[LineItem] = field(default_factory=list)
    
    # Metadata
    confidence_score: float = 0.0
    raw_text: str = ""
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['line_items'] = [item.to_dict() for item in self.line_items]
        return result
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class InvoiceExtractor:
    """
    Main class for extracting structured data from invoice images
    
    This class orchestrates the entire pipeline:
    1. Preprocessing
    2. OCR
    3. Text postprocessing
    4. Field extraction
    """
    
    def __init__(
        self,
        ocr_engine: str = "paddleocr",
        languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        preprocess: bool = True,
        use_layoutlm: bool = False,
    ):
        """
        Initialize the invoice extractor
        
        Args:
            ocr_engine: OCR engine to use ('paddleocr' or 'easyocr')
            languages: List of language codes
            use_gpu: Whether to use GPU
            preprocess: Whether to preprocess images
            use_layoutlm: Whether to use LayoutLM for extraction
        """
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self.preprocess = preprocess
        self.use_layoutlm = use_layoutlm
        
        # Initialize components
        self.ocr = get_ocr_engine(
            engine_name=ocr_engine,
            languages=self.languages,
            use_gpu=use_gpu
        )
        
        self.preprocessor = ImagePreprocessor() if preprocess else None
        self.postprocessor = TextPostprocessor()
        self.field_extractor = FieldExtractor()
        self.pdf_handler = PDFHandler()
        
        # LayoutLM extractor (lazy initialization)
        self._layoutlm_extractor = None
        
        logger.info(f"InvoiceExtractor initialized with {ocr_engine}")
    
    @property
    def layoutlm_extractor(self):
        """Lazy initialization of LayoutLM extractor"""
        if self._layoutlm_extractor is None and self.use_layoutlm:
            from .layoutlm_extractor import LayoutLMExtractor
            self._layoutlm_extractor = LayoutLMExtractor(use_gpu=self.use_gpu)
        return self._layoutlm_extractor
    
    def extract(
        self,
        image_source: Union[str, Path, np.ndarray],
        return_ocr_result: bool = False,
    ) -> Union[InvoiceData, tuple]:
        """
        Extract structured data from an invoice image
        
        Args:
            image_source: Path to image, numpy array, or PIL Image
            return_ocr_result: Whether to also return raw OCR result
            
        Returns:
            InvoiceData object (and optionally OCRResult)
        """
        image_path = Path(image_source) if isinstance(image_source, str) else None
        
        # Handle PDF
        if image_path and self.pdf_handler.is_pdf(image_path):
            return self._extract_from_pdf(image_path, return_ocr_result)
        
        # Load image
        image = self.ocr.load_image(image_source)
        
        # Preprocess
        if self.preprocessor:
            image = self.preprocessor.process(image)
        
        # Run OCR
        ocr_result = self.ocr.process(image)
        
        # Extract structured data
        invoice_data = self._extract_fields(ocr_result, image)
        
        if return_ocr_result:
            return invoice_data, ocr_result
        return invoice_data
    
    def _extract_from_pdf(
        self,
        pdf_path: Path,
        return_ocr_result: bool = False,
    ) -> Union[InvoiceData, tuple]:
        """Extract data from PDF document"""
        images = self.pdf_handler.pdf_to_images(pdf_path)
        
        if not images:
            logger.warning(f"No images extracted from PDF: {pdf_path}")
            return InvoiceData() if not return_ocr_result else (InvoiceData(), None)
        
        # Process first page (can be extended for multi-page)
        image = images[0]
        
        if self.preprocessor:
            image = self.preprocessor.process(image)
        
        ocr_result = self.ocr.process(image)
        invoice_data = self._extract_fields(ocr_result, image)
        
        if return_ocr_result:
            return invoice_data, ocr_result
        return invoice_data
    
    def _extract_fields(
        self,
        ocr_result: OCRResult,
        image: Optional[np.ndarray] = None,
    ) -> InvoiceData:
        """Extract structured fields from OCR result"""
        # Clean text
        cleaned_text = self.postprocessor.process(ocr_result.full_text)
        
        # Use LayoutLM if enabled
        if self.use_layoutlm and self.layoutlm_extractor and image is not None:
            return self._extract_with_layoutlm(ocr_result, image, cleaned_text)
        
        # Use rule-based extraction
        return self._extract_rule_based(ocr_result, cleaned_text)
    
    def _extract_rule_based(
        self,
        ocr_result: OCRResult,
        cleaned_text: str,
    ) -> InvoiceData:
        """Extract fields using rule-based approach"""
        invoice_data = InvoiceData(raw_text=cleaned_text)
        
        # Extract fields using FieldExtractor
        extracted = self.field_extractor.extract_all(
            cleaned_text,
            ocr_result.text_blocks
        )
        
        # Map extracted fields to InvoiceData
        invoice_data.vendor_name = extracted.get('vendor_name')
        invoice_data.vendor_address = extracted.get('vendor_address')
        invoice_data.invoice_number = extracted.get('invoice_number')
        invoice_data.invoice_date = extracted.get('invoice_date')
        invoice_data.due_date = extracted.get('due_date')
        invoice_data.subtotal = extracted.get('subtotal')
        invoice_data.tax_amount = extracted.get('tax')
        invoice_data.total = extracted.get('total')
        invoice_data.currency = extracted.get('currency', 'USD')
        
        # Extract line items
        line_items = extracted.get('line_items', [])
        invoice_data.line_items = [
            LineItem(**item) for item in line_items
        ]
        
        # Calculate confidence
        invoice_data.confidence_score = self._calculate_confidence(invoice_data)
        
        return invoice_data
    
    def _extract_with_layoutlm(
        self,
        ocr_result: OCRResult,
        image: np.ndarray,
        cleaned_text: str,
    ) -> InvoiceData:
        """Extract fields using LayoutLM model"""
        if not self.layoutlm_extractor:
            return self._extract_rule_based(ocr_result, cleaned_text)

        # Prepare input for LayoutLM
        words = [block.text for block in ocr_result.text_blocks]
        boxes = [block.bbox.to_list() for block in ocr_result.text_blocks]
        
        # Run LayoutLM extraction
        extracted = self.layoutlm_extractor.extract(
            image=image,
            words=words,
            boxes=boxes,
        )
        
        invoice_data = InvoiceData(raw_text=cleaned_text)
        
        # Map extracted fields
        for field_name, value in extracted.items():
            if hasattr(invoice_data, field_name):
                setattr(invoice_data, field_name, value)
        
        invoice_data.confidence_score = extracted.get('confidence', 0.0)
        
        return invoice_data
    
    def _calculate_confidence(self, invoice_data: InvoiceData) -> float:
        """Calculate overall confidence score for extraction"""
        required_fields = [
            invoice_data.invoice_number,
            invoice_data.invoice_date,
            invoice_data.total,
        ]
        
        optional_fields = [
            invoice_data.vendor_name,
            invoice_data.subtotal,
            invoice_data.tax_amount,
        ]
        
        # Count filled fields
        required_filled = sum(1 for f in required_fields if f is not None)
        optional_filled = sum(1 for f in optional_fields if f is not None)
        
        # Weight required fields more
        score = (required_filled * 0.6 / len(required_fields)) + \
                (optional_filled * 0.4 / len(optional_fields))
        
        return round(score, 2)
    
    def extract_batch(
        self,
        image_sources: List[Union[str, Path]],
        show_progress: bool = True,
    ) -> List[InvoiceData]:
        """
        Extract data from multiple invoices
        
        Args:
            image_sources: List of image paths
            show_progress: Whether to show progress bar
            
        Returns:
            List of InvoiceData objects
        """
        from tqdm import tqdm  # type: ignore
        
        results = []
        iterator = tqdm(image_sources) if show_progress else image_sources
        
        for source in iterator:
            try:
                result = self.extract(source)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {source}: {e}")
                results.append(InvoiceData())
        
        return results
