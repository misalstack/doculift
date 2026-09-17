"""
Image Utilities for Invoice Parser
"""

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image


def load_image(
    source: Union[str, Path, bytes, io.BytesIO],
) -> Image.Image:
    """
    Load image from various sources
    
    Args:
        source: File path, bytes, or BytesIO object
        
    Returns:
        PIL Image object
    """
    if isinstance(source, (str, Path)):
        return Image.open(source)
    elif isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    elif isinstance(source, io.BytesIO):
        return Image.open(source)
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")


def resize_image(
    image: Image.Image,
    max_size: Tuple[int, int] = (1920, 1920),
    maintain_aspect: bool = True,
) -> Image.Image:
    """
    Resize image to fit within max dimensions
    
    Args:
        image: PIL Image object
        max_size: Maximum (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if maintain_aspect:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(max_size, Image.Resampling.LANCZOS)


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert image to RGB mode
    
    Args:
        image: PIL Image object
        
    Returns:
        RGB image
    """
    if image.mode == "RGBA":
        # Create white background
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode != "RGB":
        return image.convert("RGB")
    return image


def image_to_bytes(
    image: Image.Image,
    format: str = "PNG",
    quality: int = 95,
) -> bytes:
    """
    Convert PIL Image to bytes
    
    Args:
        image: PIL Image object
        format: Output format (PNG, JPEG, etc.)
        quality: Quality for lossy formats
        
    Returns:
        Image bytes
    """
    buffer = io.BytesIO()
    
    save_kwargs = {}
    if format.upper() in ("JPEG", "JPG"):
        save_kwargs["quality"] = quality
        image = convert_to_rgb(image)
    
    image.save(buffer, format=format, **save_kwargs)
    return buffer.getvalue()


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array
    
    Args:
        image: PIL Image object
        
    Returns:
        Numpy array (H, W, C)
    """
    return np.array(convert_to_rgb(image))


def numpy_to_image(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image
    
    Args:
        array: Numpy array (H, W, C) or (H, W)
        
    Returns:
        PIL Image object
    """
    return Image.fromarray(array)


def get_image_info(image: Image.Image) -> dict:
    """
    Get basic information about an image
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image information
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "size_bytes": len(image_to_bytes(image)),
    }
