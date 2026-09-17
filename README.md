# Invoice Parser

AI-powered Invoice/Receipt Parser using OCR and Document Understanding.

## 🎯 Features

- **Multiple OCR Engines**: Support for PaddleOCR and EasyOCR
- **Document Understanding**: Optional LayoutLMv3 integration
- **Structured Extraction**: Extract vendor info, dates, amounts, line items
- **REST API**: FastAPI backend for integration
- **Web Interface**: Streamlit app for easy use
- **Multi-language**: Support for English, Vietnamese, and more

## 📁 Project Structure

```
invoice-parser/
├── src/                          # Source code
│   ├── __init__.py               # Package initialization
│   ├── core/                     # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration management
│   │   ├── logger.py             # Centralized logging
│   │   └── exceptions.py         # Custom exceptions
│   ├── models/                   # Data models & schemas
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic models
│   ├── ocr/                      # OCR engines
│   │   ├── __init__.py
│   │   ├── base.py               # Base OCR class
│   │   ├── paddleocr_engine.py   # PaddleOCR implementation
│   │   ├── easyocr_engine.py     # EasyOCR implementation
│   │   └── ocr_factory.py        # OCR factory pattern
│   ├── extraction/               # Field extraction
│   │   ├── __init__.py
│   │   ├── invoice_extractor.py  # Main extractor
│   │   ├── field_extractor.py    # Rule-based extraction
│   │   └── layoutlm_extractor.py # LayoutLM-based extraction
│   ├── processing/               # Image/PDF processing
│   │   ├── __init__.py
│   │   ├── preprocessor.py       # Image preprocessing
│   │   ├── postprocessor.py      # Text postprocessing
│   │   └── pdf_handler.py        # PDF handling
│   ├── utils/                    # Helper utilities
│   │   ├── __init__.py
│   │   ├── file_utils.py         # File operations
│   │   ├── text_utils.py         # Text processing
│   │   └── image_utils.py        # Image operations
│   ├── api/                      # FastAPI REST API
│   │   ├── __init__.py
│   │   ├── app.py                # FastAPI app
│   │   └── routes.py             # API routes
│   └── web/                      # Web interface
│       ├── __init__.py
│       └── streamlit_app.py      # Streamlit application
├── app/                          # (Legacy) Old streamlit location
├── config/
│   └── config.yaml               # Configuration file
├── data/                         # Data directories
│   ├── uploads/                  # Uploaded files
│   ├── outputs/                  # Processed outputs
│   └── temp/                     # Temporary files
├── tests/                        # Unit tests
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_extraction.py
│   └── test_ocr.py
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── run_api.py                    # Run FastAPI server
└── run_app.py                    # Run Streamlit app
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd invoice-parser

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Web Interface (Streamlit)

```bash
python run_app.py
```

Open http://localhost:8501 in your browser.

### 3. Run API Server

```bash
python run_api.py
```

API documentation available at http://localhost:8000/docs

## 📖 Usage

### Python API

```python
from src.extraction import InvoiceExtractor

# Initialize extractor
extractor = InvoiceExtractor(
    ocr_engine="paddleocr",
    languages=["en", "vi"],
    use_gpu=False,
)

# Extract from image
result = extractor.extract("path/to/invoice.jpg")

# Access extracted data
print(f"Invoice #: {result.invoice_number}")
print(f"Date: {result.invoice_date}")
print(f"Total: {result.currency} {result.total}")
print(f"Vendor: {result.vendor_name}")

# Export to JSON
print(result.to_json())
```

### REST API

```bash
# Parse single invoice
curl -X POST "http://localhost:8000/api/v1/parse" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.jpg"
```

### Response Format

```json
{
  "success": true,
  "message": "Invoice parsed successfully",
  "data": {
    "vendor_name": "ACME Corporation",
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "due_date": "2024-02-15",
    "subtotal": 1000.00,
    "tax_amount": 100.00,
    "total": 1100.00,
    "currency": "USD",
    "line_items": [
      {
        "description": "Product A",
        "quantity": 10,
        "unit_price": 100.00,
        "amount": 1000.00
      }
    ],
    "confidence_score": 0.85
  },
  "processing_time": 1.234
}
```

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
ocr:
  engine: "paddleocr"  # or "easyocr"
  language: ["en", "vi"]
  use_gpu: false

document_ai:
  model_name: "microsoft/layoutlmv3-base"
  use_gpu: false

api:
  host: "0.0.0.0"
  port: 8000
  max_file_size_mb: 10
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🛠️ Development

### Adding Custom OCR Engine

```python
from src.ocr.base import BaseOCR, OCRResult
from src.ocr import OCRFactory

class MyOCREngine(BaseOCR):
    def _initialize_model(self):
        # Initialize your model
        pass
    
    def _process_image(self, image):
        # Process image and return OCRResult
        pass

# Register engine
OCRFactory.register_engine("myocr", MyOCREngine)
```

### Adding Custom Extraction Rules

```python
from src.extraction.field_extractor import FieldExtractor, ExtractionRule

extractor = FieldExtractor()
extractor.add_rule(ExtractionRule(
    name="po_number",
    patterns=[
        r"PO[\s#:]*([A-Z0-9\-]+)",
        r"Purchase Order[\s#:]*([A-Z0-9\-]+)",
    ],
))
```

## 📚 Technologies

- **OCR**: PaddleOCR, EasyOCR
- **Document AI**: LayoutLMv3 (HuggingFace Transformers)
- **API**: FastAPI, Pydantic
- **Web UI**: Streamlit
- **Image Processing**: OpenCV, Pillow
- **PDF**: pdf2image, PyPDF2

## 🗺️ Roadmap

- [ ] Fine-tune LayoutLMv3 on custom dataset
- [ ] Add support for more document types (receipts, bills)
- [ ] Multi-page document support
- [ ] Table extraction enhancement
- [ ] Cloud deployment (Docker, Kubernetes)
- [ ] Integration with accounting software

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines first.
