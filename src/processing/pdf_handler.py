"""
PDF Handler - Convert PDF documents to images for OCR
"""

from typing import List, Optional, Union
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PDFHandler:
    """Handle PDF to image conversion"""
    
    def __init__(self, dpi: int = 300, fmt: str = "RGB"):
        """
        Initialize PDF handler
        
        Args:
            dpi: Resolution for conversion
            fmt: Image format (RGB, RGBA, L)
        """
        self.dpi = dpi
        self.fmt = fmt
    
    def pdf_to_images(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Convert PDF to list of images
        
        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to convert (0-indexed)
            first_page: First page to convert (1-indexed)
            last_page: Last page to convert (1-indexed)
            
        Returns:
            List of images as numpy arrays
        """
        try:
            from pdf2image import convert_from_path
            import cv2
        except ImportError:
            raise ImportError(
                "pdf2image is not installed. "
                "Install with: pip install pdf2image. "
                "Also ensure poppler is installed on your system."
            )
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Converting PDF: {pdf_path}")
        
        # Convert pages
        pil_images = convert_from_path(
            pdf_path,
            dpi=self.dpi,
            fmt=self.fmt,
            first_page=first_page,
            last_page=last_page,
        )
        
        # Filter specific pages if requested
        if pages is not None:
            pil_images = [pil_images[i] for i in pages if i < len(pil_images)]
        
        # Convert to numpy arrays (BGR for OpenCV)
        images = []
        for pil_img in pil_images:
            img_array = np.array(pil_img)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            images.append(img_array)
        
        logger.info(f"Converted {len(images)} pages from PDF")
        return images
    
    def get_page_count(self, pdf_path: Union[str, Path]) -> int:
        """Get number of pages in PDF"""
        try:
            from pdf2image import pdfinfo_from_path
        except ImportError:
            raise ImportError("pdf2image is not installed.")
        
        info = pdfinfo_from_path(str(pdf_path))
        return info.get("Pages", 0)
    
    def pdf_to_images_lazy(
        self,
        pdf_path: Union[str, Path],
    ):
        """
        Lazily convert PDF pages to images (generator)
        
        Args:
            pdf_path: Path to PDF file
            
        Yields:
            Images as numpy arrays
        """
        try:
            from pdf2image import convert_from_path
            import cv2
        except ImportError:
            raise ImportError("pdf2image is not installed.")
        
        pdf_path = Path(pdf_path)
        page_count = self.get_page_count(pdf_path)
        
        for page_num in range(1, page_count + 1):
            pil_images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt=self.fmt,
                first_page=page_num,
                last_page=page_num,
            )
            
            if pil_images:
                img_array = np.array(pil_images[0])
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                yield img_array
    
    @staticmethod
    def is_pdf(file_path: Union[str, Path]) -> bool:
        """Check if file is a PDF"""
        return Path(file_path).suffix.lower() == ".pdf"
    
    @staticmethod
    def extract_text_native(pdf_path: Union[str, Path]) -> str:
        """
        Extract text from PDF using native PDF text (not OCR)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("PyPDF2 is not installed.")
        
        reader = PdfReader(str(pdf_path))
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
