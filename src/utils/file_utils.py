"""
File Utilities for Invoice Parser
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


def get_file_extension(filename: str) -> str:
    """
    Get file extension in lowercase
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension including the dot (e.g., '.pdf')
    """
    return Path(filename).suffix.lower()


def is_valid_file(
    filename: str,
    allowed_extensions: Optional[List[str]] = None
) -> bool:
    """
    Check if file has a valid extension
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions. Defaults to common image/PDF types
        
    Returns:
        True if file extension is valid
    """
    if allowed_extensions is None:
        allowed_extensions = list(ALLOWED_EXTENSIONS)
    
    ext = get_file_extension(filename)
    return ext in [e.lower() for e in allowed_extensions]


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        path: Directory path
        
    Returns:
        Path object of the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def generate_filename(
    prefix: str = "invoice",
    extension: str = ".json",
    include_timestamp: bool = True,
    include_uuid: bool = True,
) -> str:
    """
    Generate a unique filename
    
    Args:
        prefix: Filename prefix
        extension: File extension
        include_timestamp: Include timestamp in filename
        include_uuid: Include UUID in filename
        
    Returns:
        Generated filename
    """
    parts = [prefix]
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)
    
    if include_uuid:
        short_uuid = str(uuid.uuid4())[:8]
        parts.append(short_uuid)
    
    filename = "_".join(parts)
    
    if not extension.startswith("."):
        extension = f".{extension}"
    
    return f"{filename}{extension}"


def cleanup_temp_files(
    temp_dir: str | Path,
    max_age_hours: int = 24,
    extensions: Optional[List[str]] = None,
) -> int:
    """
    Clean up old temporary files
    
    Args:
        temp_dir: Temporary directory path
        max_age_hours: Maximum age of files in hours
        extensions: Only delete files with these extensions (None = all)
        
    Returns:
        Number of files deleted
    """
    temp_path = Path(temp_dir)
    if not temp_path.exists():
        return 0
    
    deleted_count = 0
    current_time = datetime.now().timestamp()
    max_age_seconds = max_age_hours * 3600
    
    for file_path in temp_path.iterdir():
        if file_path.is_file():
            # Check extension if specified
            if extensions and file_path.suffix.lower() not in extensions:
                continue
            
            # Check age
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except OSError:
                    pass
    
    return deleted_count
