"""
Centralized Logging Configuration for Invoice Parser
"""

import sys
import logging
from pathlib import Path
from typing import Any, Optional, Union

# Try to import loguru, fall back to standard logging
try:
    from loguru import logger as _loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    _loguru_logger = None  # type: ignore


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Setup logging configuration for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        rotation: Log file rotation size (loguru only)
        retention: Log file retention period (loguru only)
    """
    if LOGURU_AVAILABLE and _loguru_logger is not None:
        # Use loguru for rich logging
        _loguru_logger.remove()
        
        # Console logger with colors
        _loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=log_level,
            colorize=True,
        )
        
        # File logger if path provided
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            _loguru_logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression="zip",
            )
    else:
        # Fall back to standard logging
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stderr)]
        )
        
        # Add file handler if path provided
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            logging.getLogger().addHandler(file_handler)


def get_logger(name: str = __name__) -> Union[Any, logging.Logger]:
    """
    Get a logger instance with the given name
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    if LOGURU_AVAILABLE and _loguru_logger is not None:
        return _loguru_logger.bind(name=name)
    else:
        return logging.getLogger(name)
