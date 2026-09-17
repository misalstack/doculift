<div id='top' align="center">

# ⚡ DocuLIFT — Intelligent Document Extraction Engine

### Next-Gen AI Platform for Automated Data Extraction

An enterprise-grade document intelligence platform powered by **PaddleOCR**, **EasyOCR**, **LayoutLMv3**, and **FastAPI**. Built to extract key field insights automatically from complex invoices, receipts, and unstructured multi-language documents.

<p>

![Python](https://img.shields.io/badge/-Python-05122A?style=for-the-badge&logo=python)&nbsp;
![FastAPI](https://img.shields.io/badge/-FastAPI-05122A?style=for-the-badge&logo=fastapi)&nbsp;
![Streamlit](https://img.shields.io/badge/-Streamlit-05122A?style=for-the-badge&logo=streamlit)&nbsp;
![PyTorch](https://img.shields.io/badge/-PyTorch-05122A?style=for-the-badge&logo=pytorch)&nbsp;
![OpenCV](https://img.shields.io/badge/-OpenCV-05122A?style=for-the-badge&logo=opencv)

</p>
</div>

---

## 🔗 Live Demo 

* 🚀 **Live App:** [DocuLIFT Railway Deployment](https://doculift-invoice-parser-production.up.railway.app)
---

## 🔧 Technologies

* **Core Engines:** PaddleOCR, EasyOCR
* **Document Understanding Model:** LayoutLMv3 (HuggingFace Transformers)
* **Rule Engine:** Smart Regex & Heuristic Parsing Algorithms
* **API Backend:** FastAPI, Pydantic
* **Web UI Framework:** Streamlit
* **Processing Tools:** OpenCV, Pillow, pdf2image, PyPDF2
* **Containerization:** Docker & Railway Deployment

---

## ✨ Key Features & Capabilities

* 🔍 **Multi-Engine OCR Processing:** Flexible switching between PaddleOCR and EasyOCR core engines for high-precision text recognition across diverse layouts.
* 📐 **Layout-Aware Extraction:** Powered by LayoutLMv3 to recognize spatial relationships between key-value pairs (e.g., matching 'Total' with its corresponding amount).
* ⚙️ **Rule-Based Heuristic Parsing:** Smart Regex algorithms designed to extract dates, invoice numbers, tax amounts, sub-totals, and currencies automatically.
* ⚡ **GPU Acceleration Support:** Built-in toggle for hardware acceleration to execute heavy deep-learning model inferences instantly.
* 📑 **Broad Format & File Support:** Handles large files up to 200MB across PNG, JPG, PDF, TIF, and BMP formats.

---

## 📁 Project Structure

```text
doculift/
├── src/                          # Core source code
│   ├── api/                      # FastAPI REST API implementation & routes
│   ├── core/                     # Configuration, logger, & custom exceptions
│   ├── extraction/               # Invoice, LayoutLMv3 & rule-based extractors
│   ├── models/                   # Pydantic data schemas
│   ├── ocr/                      # OCR engine factories (PaddleOCR & EasyOCR)
│   ├── processing/               # Image preprocessing, postprocessing & PDF handlers
│   ├── utils/                    # Helper scripts (file, text, and image utilities)
│   └── web/                      # Streamlit UI implementation
├── config/                       # System & model configuration settings (config.yaml)
├── data/                         # Temporary upload & processing directories
├── tests/                        # Comprehensive unit & extraction test suites
├── Dockerfile                    # Containerization build setup
├── run_api.py                    # Script to start FastAPI server
├── run_app.py                    # Script to launch Streamlit frontend
└── requirements.txt              # Project dependencies

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

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.
