"""
Text Postprocessor - Clean and normalize OCR output
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CleaningConfig:
    """Configuration for text cleaning"""
    remove_extra_spaces: bool = True
    fix_common_errors: bool = True
    normalize_numbers: bool = True
    normalize_dates: bool = True
    remove_special_chars: bool = False


class TextPostprocessor:
    """Postprocessing pipeline for OCR text output"""
    
    def __init__(self, config: CleaningConfig = None):
        """
        Initialize postprocessor
        
        Args:
            config: Cleaning configuration
        """
        self.config = config or CleaningConfig()
        
        # Common OCR error patterns
        self.ocr_corrections = {
            "0": ["O", "o", "Q"],
            "1": ["l", "I", "|", "i"],
            "5": ["S", "s"],
            "8": ["B"],
            "$": ["S", "s"],
        }
        
        # Date patterns
        self.date_patterns = [
            r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',
            r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',
        ]
        
        # Currency patterns
        self.currency_patterns = [
            r'[\$€£¥₫]\s*[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*(?:USD|EUR|VND|GBP)',
        ]
    
    def process(self, text: str) -> str:
        """
        Apply full postprocessing pipeline
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        result = text
        
        if self.config.remove_extra_spaces:
            result = self.clean_whitespace(result)
        
        if self.config.fix_common_errors:
            result = self.fix_ocr_errors(result)
        
        if self.config.normalize_numbers:
            result = self.normalize_numbers(result)
        
        if self.config.remove_special_chars:
            result = self.remove_special_characters(result)
        
        return result
    
    def clean_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize line breaks"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Strip each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR misrecognition errors in context"""
        # Fix common word errors
        word_fixes = {
            "lnvoice": "Invoice",
            "Tota1": "Total",
            "0ate": "Date",
            "Arnount": "Amount",
            "Ouantity": "Quantity",
            "Oescription": "Description",
            "1tem": "Item",
            "Subtota1": "Subtotal",
        }
        
        for wrong, correct in word_fixes.items():
            text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_numbers(self, text: str) -> str:
        """Normalize number formatting"""
        # Fix spaces in numbers: "1 234.56" -> "1234.56"
        def fix_number_spaces(match):
            return match.group(0).replace(' ', '')
        
        text = re.sub(r'\d[\d ]*[.,]\d+', fix_number_spaces, text)
        
        return text
    
    def remove_special_characters(self, text: str) -> str:
        """Remove unwanted special characters"""
        # Keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^\w\s\.,\-\/:$€£¥%@#&*()\[\]{}]', '', text)
        return text
    
    def extract_amounts(self, text: str) -> List[Dict[str, any]]:
        """Extract monetary amounts from text"""
        amounts = []
        
        # Pattern for currency amounts
        patterns = [
            (r'[\$]\s*([\d,]+\.?\d*)', 'USD'),
            (r'[€]\s*([\d,]+\.?\d*)', 'EUR'),
            (r'[£]\s*([\d,]+\.?\d*)', 'GBP'),
            (r'[₫]\s*([\d,]+\.?\d*)', 'VND'),
            (r'([\d,]+\.?\d*)\s*(?:VND|đ)', 'VND'),
            (r'([\d,]+\.?\d*)\s*(?:USD|\$)', 'USD'),
        ]
        
        for pattern, currency in patterns:
            for match in re.finditer(pattern, text):
                value_str = match.group(1).replace(',', '')
                try:
                    value = float(value_str)
                    amounts.append({
                        'value': value,
                        'currency': currency,
                        'original': match.group(0),
                        'position': match.span()
                    })
                except ValueError:
                    continue
        
        return amounts
    
    def extract_dates(self, text: str) -> List[Dict[str, any]]:
        """Extract dates from text"""
        dates = []
        
        for pattern in self.date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                dates.append({
                    'original': match.group(0),
                    'position': match.span(),
                    'normalized': self._normalize_date(match.group(0))
                })
        
        return dates
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date to ISO format (YYYY-MM-DD)"""
        from datetime import datetime
        
        date_formats = [
            '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d',
            '%d.%m.%Y', '%m.%d.%Y', '%Y.%m.%d',
            '%d %b %Y', '%d %B %Y',
            '%b %d, %Y', '%B %d, %Y',
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number from text"""
        patterns = [
            r'(?:Invoice|Inv|INV)[\s#:]*([A-Z0-9\-]+)',
            r'(?:Receipt|RCP)[\s#:]*([A-Z0-9\-]+)',
            r'#\s*([A-Z0-9\-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
