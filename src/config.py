"""
Configuration management for Invoice Parser

DEPRECATED: This module is kept for backward compatibility.
Please use `from src.core import get_config, load_config, AppConfig` instead.
"""

# Re-export from new location for backward compatibility
from .core.config import (
    OCRConfig,
    DocumentAIConfig,
    ExtractionConfig,
    APIConfig,
    StorageConfig,
    AppConfig,
    load_config,
    get_config,
)

import warnings

warnings.warn(
    "src.config is deprecated. Use src.core instead: "
    "from src.core import get_config, AppConfig",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "OCRConfig",
    "DocumentAIConfig", 
    "ExtractionConfig",
    "APIConfig",
    "StorageConfig",
    "AppConfig",
    "load_config",
    "get_config",
]

