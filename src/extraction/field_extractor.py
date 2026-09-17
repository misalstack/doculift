"""
Field Extractor - Rule-based extraction of invoice fields
"""

import re
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass

from ..ocr.base import TextBlock


@dataclass
class ExtractionRule:
    """Rule for extracting a field"""
    name: str
    patterns: List[str]
    post_process: Optional[Any] = None
    priority: int = 0


class FieldExtractor:
    """Rule-based field extraction from OCR text"""
    
    def __init__(self):
        """Initialize field extractor with default rules"""
        self.rules = self._create_default_rules()
    
    def _create_default_rules(self) -> Dict[str, ExtractionRule]:
        """Create default extraction rules"""
        return {
            'invoice_number': ExtractionRule(
                name='invoice_number',
                patterns=[
                    # Invoice No : INV/MNSG8/17/110435
                    r'(?:Invoice\s*No|Invoice\s*Number|Inv\.?\s*No)[\s:]+([A-Z0-9/\-]+)',
                    r'(?:Invoice|Inv|INV)[\s#:]*([A-Z0-9/\-]{5,})',
                    r'(?:Receipt|RCP)[\s#:No.]*([A-Z0-9\-]+)',
                    r'(?:Bill|Order)[\s#:No.]*([A-Z0-9\-]+)',
                    r'#\s*([A-Z0-9]{6,})',
                ],
                priority=10
            ),
            'invoice_date': ExtractionRule(
                name='invoice_date',
                patterns=[
                    # Date : 20-Nov-2017
                    r'\bDate\s*[:\-]?\s*(\d{1,2}\s*[\-/]\s*[A-Za-z]{3,9}\s*[\-/]\s*\d{4})\b',
                    r'(?:Invoice\s*)?Date\s*[:\-]?\s*(\d{1,2}[\-/][A-Za-z]{3}[\-/]\d{4})\b',
                    r'(?:Invoice\s*)?Date\s*[:\-]?\s*(\d{1,2}[\-/][A-Za-z]{3,9}[\-/]\d{4})\b',
                    r'(?:Invoice\s*)?Date\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b',
                    r'(?:Date|Dated)[\s:]*(\d{1,2}\s+\w+\s+\d{4})',
                    r'(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})',
                    # Fallback: date token anywhere (riskier, keep last)
                    r'\b(\d{1,2}[\-/][A-Za-z]{3}[\-/]\d{4})\b',
                ],
                priority=9
            ),
            'due_date': ExtractionRule(
                name='due_date',
                patterns=[
                    r'(?:Due|Payment)\s*(?:Date)?[\s:]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
                    r'(?:Due|Payment)\s*(?:Date)?[\s:]*(\d{1,2}[\-/]\w{3,9}[\-/]\d{4})',
                    r'(?:Pay\s*by|Due\s*by)[\s:]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
                ],
                priority=8
            ),
            'total': ExtractionRule(
                name='total',
                patterns=[
                    # TOTAL 44,780.00
                    r'TOTAL[\s:]*[\$€£¥₫]?\s*([\d,]+\.?\d*)',
                    r'(?:Grand\s*)?Total[\s:]*[\$€£¥₫]?\s*([\d,]+\.?\d*)',
                    r'(?:Amount\s*Due|Balance\s*Due)[\s:]*[\$€£¥₫]?\s*([\d,]+\.?\d*)',
                    r'[\$€£¥₫]\s*([\d,]+\.?\d*)\s*(?:Total|Due)',
                ],
                post_process=lambda x: float(x.replace(',', '')) if x else None,
                priority=10
            ),
            'subtotal': ExtractionRule(
                name='subtotal',
                patterns=[
                    r'(?:SUB\s*TOTAL|Sub\s*total|Subtotal)[\s:]*[\$€£¥₫]?\s*([\d,]+\.?\d*)',
                    r'(?:Net\s*Amount)[\s:]*[\$€£¥₫]?\s*([\d,]+\.?\d*)',
                ],
                post_process=lambda x: float(x.replace(',', '')) if x else None,
                priority=7
            ),
            'tax': ExtractionRule(
                name='tax',
                patterns=[
                    r'(?:GST|Tax|VAT)\s*[\d.]*\s*%?[\s:]*[\$€£¥₫]?\s*([\d,]+\.?\d+)',
                    r'(?:Tax|VAT|GST)[\s:]*[\$€£¥₫]?\s*([\d,]+\.?\d*)',
                ],
                post_process=lambda x: float(x.replace(',', '')) if x else None,
                priority=6
            ),
            'vendor_name': ExtractionRule(
                name='vendor_name',
                patterns=[
                    # Match first line with company name (typically CAPS)
                    r'^([A-Z][A-Z\s\-]+(?:PTE\s*LTD|LTD|INC|LLC|CORP|CO\.?))',
                    r'^([A-Z][A-Za-z\s&.,]+(?:Inc|LLC|Ltd|Corp|Co\.?)?)',
                    r'(?:From|Seller|Vendor)[\s:]*([A-Z][A-Za-z\s&.,]+)',
                ],
                priority=5
            ),
            'vendor_address': ExtractionRule(
                name='vendor_address',
                patterns=[
                    # Multi-line address with road type + (optional) country/city + postal
                    r'(\d{1,6}\s+[A-Za-z][A-Za-z\s]{2,60}?(?:Avenue|Ave|Street|St|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Place|Pl|Highway|Hwy)\s+\d{0,4}\s+(?:[A-Za-z\s]{2,30}\s+)?\d{5,6})',
                    # Road type address (no postal required)
                    r'(\d{1,6}\s+[A-Za-z][A-Za-z\s]{2,80}?(?:Avenue|Ave|Street|St|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Place|Pl|Highway|Hwy)\s+\d{0,4}(?:\s*,\s*[A-Za-z\s]{2,40})?)',
                    # City/Country + postal code (kept last; post-process rejects Tel/Fax lines)
                    r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+\d{5,6})\b',
                ],
                post_process=self._postprocess_address,
                priority=4
            ),
            'vendor_tax_id': ExtractionRule(
                name='vendor_tax_id',
                patterns=[
                    # GST NO: 200510588R
                    r'\bGST\s*(?:NO|No|N0)\.?\s*[:#-]?\s*([A-Z0-9\-]{6,})\b',
                    r'\b(?:GST\s*Reg|Tax\s*ID|TIN)\s*[:#-]?\s*([A-Z0-9\-]{6,})\b',
                    r'\bROC\s*[:#-]?\s*([A-Z0-9\-]{6,})\b',
                    r'\bUEN\s*[:#-]?\s*([A-Z0-9\-]{6,})\b',
                ],
                priority=4
            ),
            'customer_name': ExtractionRule(
                name='customer_name',
                patterns=[
                    # To: RAM VIETNAM MTV CO. LTD
                    r'To[\s:]+([A-Z][A-Z\s\.,]+(?:CO\.?\s*LTD|LTD|INC|LLC|CORP))',
                    r'(?:Bill\s*to|Billed\s*to)[\s:]+([A-Z][A-Za-z\s&.,]+)',
                ],
                priority=5
            ),
            'customer_address': ExtractionRule(
                name='customer_address',
                patterns=[
                    r'(?:ATTN|Attention)[\s:]+([A-Z][A-Za-z\s,]+)',
                ],
                priority=3
            ),
            'currency': ExtractionRule(
                name='currency',
                patterns=[
                    r'\(USD\)|USD',
                    r'(USD|EUR|GBP|VND|JPY|SGD)',
                    r'[\$]',  # Matched -> USD
                    r'[€]',   # Matched -> EUR
                    r'[£]',   # Matched -> GBP
                    r'[₫]',   # Matched -> VND
                ],
                priority=3
            ),
        }

    def _postprocess_address(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None

        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned:
            return None

        # Reject common false positives from invoice headers/footers and contact lines
        forbidden = (
            "tel", "telephone", "phone", "fax", "gst", "roc", "uen",
            "invoice", "total", "subtotal", "payment", "remit", "bank",
            "cheque", "account", "attn", "attention",
        )
        lower = cleaned.lower()
        if any(word in lower for word in forbidden):
            return None

        # Reject strings that look like phone/fax lines (few letters, mostly digits)
        letters = sum(ch.isalpha() for ch in cleaned)
        digits = sum(ch.isdigit() for ch in cleaned)
        if digits >= 6 and letters < 6:
            return None

        return cleaned
    
    def add_rule(self, rule: ExtractionRule):
        """Add a custom extraction rule"""
        self.rules[rule.name] = rule
    
    def extract_field(
        self,
        field_name: str,
        text: str,
    ) -> Optional[str]:
        """
        Extract a specific field from text
        
        Args:
            field_name: Name of the field to extract
            text: Text to search in
            
        Returns:
            Extracted value or None
        """
        if field_name not in self.rules:
            return None
        
        rule = self.rules[field_name]
        
        for pattern in rule.patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1) if match.lastindex else match.group(0)
                
                # Apply post-processing if defined
                if rule.post_process:
                    try:
                        value = rule.post_process(value)
                    except:
                        pass
                
                return value
        
        return None
    
    def extract_all(
        self,
        text: str,
        text_blocks: Optional[List[TextBlock]] = None,
    ) -> Dict[str, Any]:
        """
        Extract all fields from text
        
        Args:
            text: Full text to search
            text_blocks: Optional list of text blocks with positions
            
        Returns:
            Dictionary of extracted fields
        """
        result = {}
        
        # Extract using rules
        for field_name, rule in self.rules.items():
            value = self.extract_field(field_name, text)
            if value is not None:
                result[field_name] = value
        
        # Special handling for currency
        if 'currency' not in result or result['currency'] in ['$', '€', '£', '₫']:
            result['currency'] = self._determine_currency(text)
        
        # Extract line items
        result['line_items'] = self._extract_line_items(text, text_blocks)
        
        return result
    
    def _determine_currency(self, text: str) -> str:
        """Determine currency from text"""
        currency_map = {
            '$': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            '₫': 'VND',
            '¥': 'JPY',
        }
        
        # Check for explicit currency codes
        for code in ['USD', 'EUR', 'GBP', 'VND', 'JPY', 'CNY']:
            if code in text:
                return code
        
        # Check for currency symbols
        for symbol, code in currency_map.items():
            if symbol in text:
                return code
        
        return 'USD'  # Default
    
    def _extract_line_items(
        self,
        text: str,
        text_blocks: Optional[List[TextBlock]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract line items from invoice
        
        Args:
            text: Full text
            text_blocks: Text blocks with positions
            
        Returns:
            List of line item dictionaries
        """
        line_items = []
        
        # Pattern for line items: description, quantity, price, amount
        # This is a simplified pattern - real invoices vary greatly
        patterns = [
            # Description Qty Unit Price Amount
            r'([A-Za-z][A-Za-z\s]+)\s+(\d+)\s+[\$€£]?([\d,.]+)\s+[\$€£]?([\d,.]+)',
            # Description Amount
            r'([A-Za-z][A-Za-z\s]{10,})\s+[\$€£]?([\d,.]+)\s*$',
        ]
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            # Try full pattern
            match = re.match(patterns[0], line)
            if match:
                try:
                    line_items.append({
                        'description': match.group(1).strip(),
                        'quantity': float(match.group(2)),
                        'unit_price': float(match.group(3).replace(',', '')),
                        'amount': float(match.group(4).replace(',', '')),
                    })
                    continue
                except ValueError:
                    pass
            
            # Try simple pattern
            match = re.match(patterns[1], line, re.MULTILINE)
            if match:
                try:
                    line_items.append({
                        'description': match.group(1).strip(),
                        'quantity': None,
                        'unit_price': None,
                        'amount': float(match.group(2).replace(',', '')),
                    })
                except ValueError:
                    pass
        
        return line_items
    
    def extract_with_positions(
        self,
        text_blocks: List[TextBlock],
    ) -> Dict[str, Tuple[Any, TextBlock]]:
        """
        Extract fields with their positions
        
        Args:
            text_blocks: List of text blocks
            
        Returns:
            Dictionary of (value, text_block) tuples
        """
        result = {}
        full_text = " ".join([b.text for b in text_blocks])
        
        for field_name, rule in self.rules.items():
            for pattern in rule.patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    matched_text = match.group(0)
                    
                    # Find the text block containing this match
                    for block in text_blocks:
                        if matched_text in block.text or block.text in matched_text:
                            value = match.group(1) if match.lastindex else match.group(0)
                            if rule.post_process:
                                try:
                                    value = rule.post_process(value)
                                except:
                                    pass
                            result[field_name] = (value, block)
                            break
                    break
        
        return result
