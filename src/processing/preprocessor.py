"""
Image Preprocessor - Prepare images for OCR
"""

from typing import Tuple, Optional, List
import numpy as np
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocessing pipeline for document images"""
    
    def __init__(
        self,
        target_dpi: int = 300,
        denoise: bool = True,
        deskew: bool = True,
        binarize: bool = False,
    ):
        """
        Initialize preprocessor
        
        Args:
            target_dpi: Target DPI for resizing
            denoise: Whether to apply denoising
            deskew: Whether to correct skew
            binarize: Whether to binarize image
        """
        self.target_dpi = target_dpi
        self.denoise = denoise
        self.deskew = deskew
        self.binarize = binarize
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image
        """
        result = image.copy()
        
        # Convert to grayscale for some operations
        if len(result.shape) == 3:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = result
        
        # Denoise
        if self.denoise:
            result = self.apply_denoise(result)
        
        # Deskew
        if self.deskew:
            result = self.apply_deskew(result, gray)
        
        # Binarize
        if self.binarize:
            result = self.apply_binarize(result)
        
        return result
    
    def apply_denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising filter"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def apply_deskew(self, image: np.ndarray, gray: np.ndarray = None) -> np.ndarray:
        """Correct skew in document image"""
        if gray is None:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
        
        # Detect skew angle
        angle = self._get_skew_angle(gray)
        
        if abs(angle) > 0.5:  # Only correct if angle is significant
            logger.debug(f"Correcting skew angle: {angle:.2f} degrees")
            return self._rotate_image(image, angle)
        
        return image
    
    def _get_skew_angle(self, gray: np.ndarray) -> float:
        """Detect skew angle using Hough transform"""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 
            threshold=100, minLineLength=100, maxLineGap=10
        )
        
        if lines is None:
            return 0.0
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines
            if abs(angle) < 45:
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        return np.median(angles)
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image bounds
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        return cv2.warpAffine(
            image, rotation_matrix, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
    
    def apply_binarize(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive binarization"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return binary
    
    def resize_to_dpi(
        self, 
        image: np.ndarray, 
        current_dpi: int = 72
    ) -> np.ndarray:
        """Resize image to target DPI"""
        scale = self.target_dpi / current_dpi
        
        if abs(scale - 1.0) < 0.1:
            return image
        
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def remove_borders(self, image: np.ndarray, border_size: int = 10) -> np.ndarray:
        """Remove borders from document image"""
        h, w = image.shape[:2]
        return image[border_size:h-border_size, border_size:w-border_size]
    
    def auto_crop(self, image: np.ndarray) -> np.ndarray:
        """Auto-crop to document boundaries"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return image[y:y+h, x:x+w]
