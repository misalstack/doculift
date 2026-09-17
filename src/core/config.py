"""
Configuration management for Invoice Parser
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OCRConfig(BaseModel):
    """OCR Engine Configuration"""
    engine: str = "paddleocr"
    language: List[str] = ["en", "vi"]
    use_gpu: bool = False


class DocumentAIConfig(BaseModel):
    """Document Understanding Model Configuration"""
    model_name: str = "microsoft/layoutlmv3-base"
    use_gpu: bool = False
    max_length: int = 512


class ExtractionConfig(BaseModel):
    """Field Extraction Configuration"""
    fields: List[str] = [
        "vendor_name", "vendor_address", "invoice_number",
        "invoice_date", "due_date", "subtotal", "tax", "total", "line_items"
    ]
    confidence_threshold: float = 0.7


class APIConfig(BaseModel):
    """API Server Configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_file_size_mb: int = 10
    allowed_extensions: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]


class StorageConfig(BaseModel):
    """Storage Paths Configuration"""
    upload_dir: str = "./data/uploads"
    output_dir: str = "./data/outputs"
    temp_dir: str = "./data/temp"


class AppConfig(BaseModel):
    """Main Application Configuration"""
    name: str = "Invoice Parser"
    version: str = "0.1.0"
    debug: bool = True
    log_level: str = "INFO"
    
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    document_ai: DocumentAIConfig = Field(default_factory=DocumentAIConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file. Defaults to config/config.yaml
        
    Returns:
        AppConfig object
    """
    if config_path is None:
        # Support both old and new locations
        path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    else:
        path = Path(config_path)
    
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
            
        # Flatten nested config
        flat_config = {
            "name": config_dict.get("app", {}).get("name", "Invoice Parser"),
            "version": config_dict.get("app", {}).get("version", "0.1.0"),
            "debug": config_dict.get("app", {}).get("debug", True),
            "log_level": config_dict.get("app", {}).get("log_level", "INFO"),
            "ocr": config_dict.get("ocr", {}),
            "document_ai": config_dict.get("document_ai", {}),
            "extraction": config_dict.get("extraction", {}),
            "api": config_dict.get("api", {}),
            "storage": config_dict.get("storage", {}),
        }
        
        return AppConfig(**flat_config)
    
    return AppConfig()


@lru_cache()
def get_config() -> AppConfig:
    """Get the global configuration instance (cached)"""
    return load_config()
