"""
API Routes for Invoice Parser
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import List, Optional, cast

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from ..core import get_config
from ..extraction import InvoiceExtractor
from ..extraction.invoice_extractor import InvoiceData


router = APIRouter(tags=["Invoice Parser"])

# Global extractor instance (lazy initialization)
_extractor: Optional[InvoiceExtractor] = None


def get_extractor() -> InvoiceExtractor:
    """Get or create invoice extractor instance"""
    global _extractor
    if _extractor is None:
        config = get_config()
        _extractor = InvoiceExtractor(
            ocr_engine=config.ocr.engine,
            languages=config.ocr.language,
            use_gpu=config.ocr.use_gpu,
            use_layoutlm=False,  # Can be enabled in config
        )
    return _extractor


# Pydantic models for API responses
class LineItemResponse(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class InvoiceResponse(BaseModel):
    """Response model for parsed invoice data"""
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    vendor_tax_id: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    customer_name: Optional[str] = None
    customer_address: Optional[str] = None
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    tax_rate: Optional[float] = None
    discount: Optional[float] = None
    total: Optional[float] = None
    currency: str = "USD"
    line_items: List[LineItemResponse] = Field(default_factory=list)
    confidence_score: float = 0.0
    raw_text: str = ""


class ParseResponse(BaseModel):
    """Complete API response for invoice parsing"""
    success: bool
    message: str
    data: Optional[InvoiceResponse] = None
    processing_time: Optional[float] = None


class BatchParseResponse(BaseModel):
    """Response for batch parsing"""
    success: bool
    message: str
    total: int
    successful: int
    failed: int
    results: List[ParseResponse] = Field(default_factory=list)


def cleanup_file(file_path: str):
    """Background task to cleanup uploaded files"""
    try:
        os.remove(file_path)
        logger.debug(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {e}")


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    config = get_config()
    
    # Check filename exists
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.api.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {config.api.allowed_extensions}"
        )
    
    # Check file size (approximate check)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset
    
    max_size = config.api.max_file_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {config.api.max_file_size_mb}MB"
        )


async def save_upload_file(file: UploadFile) -> Path:
    """Save uploaded file to temp directory"""
    config = get_config()
    
    # Create temp directory
    temp_dir = Path(config.storage.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_ext = Path(file.filename or "upload").suffix
    unique_name = f"{uuid.uuid4()}{file_ext}"
    file_path = temp_dir / unique_name
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path


@router.post("/parse", response_model=ParseResponse)
async def parse_invoice(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,  # type: ignore
):
    """
    Parse a single invoice/receipt image
    
    Upload an image file (PDF, PNG, JPG, etc.) and receive structured data.
    """
    import time
    
    # Validate file
    validate_file(file)
    
    try:
        start_time = time.time()
        
        # Save file
        file_path = await save_upload_file(file)
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_file, str(file_path))
        
        # Get extractor and process
        extractor = get_extractor()
        invoice_data = cast(InvoiceData, extractor.extract(file_path))
        
        processing_time = time.time() - start_time
        
        # Convert to response model
        response_data = InvoiceResponse(
            vendor_name=invoice_data.vendor_name,
            vendor_address=invoice_data.vendor_address,
            vendor_tax_id=invoice_data.vendor_tax_id,
            invoice_number=invoice_data.invoice_number,
            invoice_date=invoice_data.invoice_date,
            due_date=invoice_data.due_date,
            customer_name=invoice_data.customer_name,
            customer_address=invoice_data.customer_address,
            subtotal=invoice_data.subtotal,
            tax_amount=invoice_data.tax_amount,
            tax_rate=invoice_data.tax_rate,
            discount=invoice_data.discount,
            total=invoice_data.total,
            currency=invoice_data.currency,
            line_items=[
                LineItemResponse(**item.to_dict())
                for item in invoice_data.line_items
            ],
            confidence_score=invoice_data.confidence_score,
            raw_text=invoice_data.raw_text,
        )
        
        return ParseResponse(
            success=True,
            message="Invoice parsed successfully",
            data=response_data,
            processing_time=round(processing_time, 3),
        )
        
    except Exception as e:
        logger.error(f"Error parsing invoice: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing invoice: {str(e)}"
        )


@router.post("/parse/batch", response_model=BatchParseResponse)
async def parse_invoices_batch(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,  # type: ignore
):
    """
    Parse multiple invoice/receipt images
    
    Upload multiple files and receive structured data for each.
    """
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            # Validate and process each file
            validate_file(file)
            file_path = await save_upload_file(file)
            
            if background_tasks:
                background_tasks.add_task(cleanup_file, str(file_path))
            
            extractor = get_extractor()
            invoice_data = cast(InvoiceData, extractor.extract(file_path))
            
            response_data = InvoiceResponse(
                vendor_name=invoice_data.vendor_name,
                invoice_number=invoice_data.invoice_number,
                invoice_date=invoice_data.invoice_date,
                total=invoice_data.total,
                currency=invoice_data.currency,
                confidence_score=invoice_data.confidence_score,
            )
            
            results.append(ParseResponse(
                success=True,
                message=f"Parsed: {file.filename}",
                data=response_data,
            ))
            successful += 1
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append(ParseResponse(
                success=False,
                message=f"Error: {str(e)}",
                data=None,
            ))
            failed += 1
    
    return BatchParseResponse(
        success=failed == 0,
        message=f"Processed {len(files)} files",
        total=len(files),
        successful=successful,
        failed=failed,
        results=results,
    )


@router.get("/ocr-engines")
async def get_available_engines():
    """Get list of available OCR engines"""
    from ..ocr import OCRFactory
    
    return {
        "engines": OCRFactory.get_available_engines(),
        "current": get_config().ocr.engine,
    }


@router.get("/config")
async def get_current_config():
    """Get current configuration (non-sensitive)"""
    config = get_config()
    
    return {
        "ocr_engine": config.ocr.engine,
        "languages": config.ocr.language,
        "use_gpu": config.ocr.use_gpu,
        "max_file_size_mb": config.api.max_file_size_mb,
        "allowed_extensions": config.api.allowed_extensions,
    }
