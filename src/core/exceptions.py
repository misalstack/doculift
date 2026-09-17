"""
Custom Exceptions for Invoice Parser
"""

from typing import Optional


class InvoiceParserError(Exception):
    """Base exception for all Invoice Parser errors"""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class OCRError(InvoiceParserError):
    """Exception raised when OCR processing fails"""
    pass


class ExtractionError(InvoiceParserError):
    """Exception raised when field extraction fails"""
    pass


class ConfigurationError(InvoiceParserError):
    """Exception raised for configuration errors"""
    pass


class FileProcessingError(InvoiceParserError):
    """Exception raised when file processing fails"""
    pass


class ValidationError(InvoiceParserError):
    """Exception raised when validation fails"""
    pass


class ModelLoadError(InvoiceParserError):
    """Exception raised when model loading fails"""
    pass
