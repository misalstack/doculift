"""
Tests for extraction module
"""

import pytest


class TestFieldExtractor:
    """Tests for field extraction"""
    
    def test_extract_invoice_number(self):
        from src.extraction.field_extractor import FieldExtractor
        
        extractor = FieldExtractor()
        
        # Test various formats
        assert extractor.extract_field("invoice_number", "Invoice #12345") == "12345"
        assert extractor.extract_field("invoice_number", "INV: ABC-123") == "ABC-123"
        assert extractor.extract_field("invoice_number", "Invoice No. 2024-001") == "2024-001"
    
    def test_extract_date(self):
        from src.extraction.field_extractor import FieldExtractor
        
        extractor = FieldExtractor()
        
        text = "Invoice Date: 15/01/2024"
        result = extractor.extract_field("invoice_date", text)
        assert result == "15/01/2024"
    
    def test_extract_total(self):
        from src.extraction.field_extractor import FieldExtractor
        
        extractor = FieldExtractor()
        
        assert extractor.extract_field("total", "Total: $1,234.56") == 1234.56
        assert extractor.extract_field("total", "Grand Total $99.99") == 99.99
    
    def test_determine_currency(self):
        from src.extraction.field_extractor import FieldExtractor
        
        extractor = FieldExtractor()
        
        assert extractor._determine_currency("Total: $100.00") == "USD"
        assert extractor._determine_currency("Amount: €50.00") == "EUR"
        assert extractor._determine_currency("Price: 1,000,000 VND") == "VND"
    
    def test_extract_all_fields(self):
        from src.extraction.field_extractor import FieldExtractor
        
        extractor = FieldExtractor()
        
        text = """
        ACME Corporation
        Invoice #2024-001
        Date: 15/01/2024
        
        Subtotal: $100.00
        Tax: $10.00
        Total: $110.00
        """
        
        result = extractor.extract_all(text)
        
        assert result.get("invoice_number") == "2024-001"
        assert result.get("invoice_date") == "15/01/2024"
        assert result.get("subtotal") == 100.00
        assert result.get("tax") == 10.00
        assert result.get("total") == 110.00
        assert result.get("currency") == "USD"


class TestTextPostprocessor:
    """Tests for text postprocessing"""
    
    def test_clean_whitespace(self):
        from src.processing.postprocessor import TextPostprocessor
        
        processor = TextPostprocessor()
        
        text = "Hello    World\n\n\n\nTest"
        result = processor.clean_whitespace(text)
        
        assert "    " not in result
        assert "\n\n\n" not in result
    
    def test_extract_amounts(self):
        from src.processing.postprocessor import TextPostprocessor
        
        processor = TextPostprocessor()
        
        text = "Price: $100.50 and €50.00"
        amounts = processor.extract_amounts(text)
        
        assert len(amounts) >= 2
        assert any(a["currency"] == "USD" for a in amounts)
        assert any(a["currency"] == "EUR" for a in amounts)
    
    def test_extract_dates(self):
        from src.processing.postprocessor import TextPostprocessor
        
        processor = TextPostprocessor()
        
        text = "Date: 15/01/2024 Due: 15-02-2024"
        dates = processor.extract_dates(text)
        
        assert len(dates) >= 2


class TestInvoiceData:
    """Tests for invoice data model"""
    
    def test_invoice_data_to_dict(self):
        from src.extraction.invoice_extractor import InvoiceData, LineItem
        
        invoice = InvoiceData(
            vendor_name="Test Corp",
            invoice_number="INV-001",
            total=100.00,
            line_items=[
                LineItem(description="Item 1", quantity=2, amount=50.00),
            ]
        )
        
        d = invoice.to_dict()
        
        assert d["vendor_name"] == "Test Corp"
        assert d["invoice_number"] == "INV-001"
        assert d["total"] == 100.00
        assert len(d["line_items"]) == 1
    
    def test_invoice_data_to_json(self):
        from src.extraction.invoice_extractor import InvoiceData
        import json
        
        invoice = InvoiceData(
            vendor_name="Test Corp",
            total=100.00,
        )
        
        json_str = invoice.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["vendor_name"] == "Test Corp"
        assert parsed["total"] == 100.00
