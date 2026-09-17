"""
Text Utilities for Invoice Parser
"""

import re
from typing import List, Optional, Tuple
from decimal import Decimal


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove control characters
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
    
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text (convert multiple spaces to single)
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    return re.sub(r"\s+", " ", text).strip()


def extract_numbers(text: str, include_decimals: bool = True) -> List[float]:
    """
    Extract all numbers from text
    
    Args:
        text: Input text
        include_decimals: Whether to extract decimal numbers
        
    Returns:
        List of numbers found
    """
    if include_decimals:
        # Match integers and decimals (supports both . and , as decimal separator)
        pattern = r"-?\d+(?:[.,]\d+)?"
    else:
        pattern = r"-?\d+"
    
    matches = re.findall(pattern, text)
    numbers = []
    
    for match in matches:
        # Normalize decimal separator
        normalized = match.replace(",", ".")
        try:
            numbers.append(float(normalized))
        except ValueError:
            pass
    
    return numbers


def extract_currency_amount(text: str) -> Optional[Tuple[float, str]]:
    """
    Extract currency amount and symbol from text
    
    Args:
        text: Input text containing currency amount
        
    Returns:
        Tuple of (amount, currency_symbol) or None if not found
    """
    # Common currency patterns
    patterns = [
        # $1,234.56 or $1234.56
        (r"\$\s*([\d,]+(?:\.\d{2})?)", "USD"),
        # VND 1.234.567 or 1,234,567 VND
        (r"([\d.,]+)\s*(?:VND|đ|₫)", "VND"),
        (r"(?:VND|đ|₫)\s*([\d.,]+)", "VND"),
        # €1.234,56
        (r"€\s*([\d.,]+)", "EUR"),
        # Generic number with common separators
        (r"([\d,]+(?:\.\d{2})?)", ""),
    ]
    
    for pattern, currency in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1)
            # Normalize: remove thousands separators, fix decimal
            # Assume last separator is decimal if followed by 2 digits
            amount_str = amount_str.replace(" ", "")
            
            if re.search(r"[.,]\d{2}$", amount_str):
                # Has decimal portion
                amount_str = re.sub(r"[.,](?=\d{2}$)", ".", amount_str)
                amount_str = amount_str.replace(",", "").replace(".", "", amount_str.count(".") - 1)
            else:
                # No decimal
                amount_str = amount_str.replace(",", "").replace(".", "")
            
            try:
                return (float(amount_str), currency)
            except ValueError:
                continue
    
    return None


def extract_dates(text: str) -> List[str]:
    """
    Extract dates from text
    
    Args:
        text: Input text
        
    Returns:
        List of date strings found
    """
    date_patterns = [
        # DD/MM/YYYY or DD-MM-YYYY
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        # YYYY/MM/DD or YYYY-MM-DD
        r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
        # Month DD, YYYY
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
        # DD Month YYYY
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b",
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    return dates


def format_currency(
    amount: float,
    currency: str = "VND",
    locale: str = "vi_VN",
) -> str:
    """
    Format amount as currency string
    
    Args:
        amount: Numeric amount
        currency: Currency code
        locale: Locale for formatting
        
    Returns:
        Formatted currency string
    """
    if currency == "VND":
        # Vietnamese format: 1.234.567 đ
        formatted = f"{amount:,.0f}".replace(",", ".")
        return f"{formatted} đ"
    elif currency == "USD":
        return f"${amount:,.2f}"
    elif currency == "EUR":
        formatted = f"{amount:,.2f}".replace(",", " ").replace(".", ",")
        return f"€{formatted}"
    else:
        return f"{amount:,.2f} {currency}"


def similarity_score(text1: str, text2: str) -> float:
    """
    Calculate simple similarity score between two texts
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()
    
    if t1 == t2:
        return 1.0
    
    # Simple character-based similarity
    set1 = set(t1)
    set2 = set(t2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0
