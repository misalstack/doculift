"""
Streamlit Web Application for DocuLIFT
"""

import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import json
from pathlib import Path
from PIL import Image
from typing import cast
import io
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # Change working directory to project root

from src.extraction import InvoiceExtractor
from src.extraction.invoice_extractor import InvoiceData
from src.core import get_config


# Page configuration
st.set_page_config(
    page_title="DocuLIFT | Intelligent Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS - DocuLIFT Theme
st.markdown("""
<style>
    /* Global App Background */
    .stApp, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1e1b4b 0%, #090d16 40%) !important;
        color: #f9fafb !important;
    }
    
    /* Header Hide */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0d1322 !important;
        border-right: 1px solid #1f2937 !important;
    }

    /* Gradient Brand Title */
    .main-header {
        font-size: 3rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    /* Primary Action Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%) !important;
        color: #ffffff !important;
        border: 1px solid #a78bfa !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.2rem !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6) !important;
    }

    /* Metric Display Cards */
    [data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.7) !important;
        border: 1px solid #374151 !important;
        backdrop-filter: blur(8px) !important;
        padding: 14px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #9ca3af !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    button[aria-selected="true"] {
        background-color: #1f2937 !important;
        color: #a78bfa !important;
        border-bottom: 2px solid #8b5cf6 !important;
    }

    /* Input & File Uploader Box */
    [data-testid="stFileUploadDropzone"] {
        background-color: #111827 !important;
        border: 2px dashed #4b5563 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #8b5cf6 !important;
    }

    /* Table Styling */
    .stTable, [data-testid="stDataFrame"] {
        background-color: #111827 !important;
        border-radius: 10px !important;
        border: 1px solid #1f2937 !important;
    }

    /* Text Colors Fix */
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #f9fafb !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_extractor(ocr_engine: str, languages: list, use_gpu: bool):
    """Load and cache the DocuLIFT extraction engine"""
    return InvoiceExtractor(
        ocr_engine=ocr_engine,
        languages=languages,
        use_gpu=use_gpu,
    )


def main():
    # Brand Header
    st.markdown('<h1 class="main-header">⚡ DocuLIFT</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Next-Gen AI Document Extraction & Vision Intelligence Engine</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Engine Control")
        
        ocr_engine = st.selectbox(
            "OCR Processing Core",
            options=["paddleocr", "easyocr"],
            index=0,
            help="Select underlying OCR neural architecture. On the free cloud deploy, PaddleOCR is installed by default.",
        )
        
        languages = st.multiselect(
            "Target Languages",
            options=["en", "vi", "zh", "ja", "ko", "fr", "de"],
            default=["en"],
            help="Select document text recognition models"
        )
        
        use_gpu = st.checkbox(
            "Enable GPU Acceleration",
            value=False,
            help="Leave off on free hosting (CPU only). Enable only if a GPU is available.",
        )
        
        st.divider()
        
        st.markdown("### 🛠️ Core Tech")
        st.markdown("""
        - **Vision Engines:** PaddleOCR / EasyOCR
        - **Parser:** Rule-based Heuristics & RegEx
        - **Layout Model:** LayoutLMv3 Architecture
        """)
        
        st.divider()
        
        st.markdown("### 🔍 Target Datapoints")
        st.markdown("""
        - Vendor Meta & Entity Details
        - Invoice Meta & Unique IDs
        - Timestamps & Execution Dates
        - Financial Totals & Tax Breakdown
        - Line-Item Itemized Tables
        """)
    
    # Main content layout
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("📤 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Drop invoice, receipt or document image here",
            type=["png", "jpg", "jpeg", "pdf", "tiff", "bmp"],
            help="Supported formats: Images and multi-page PDFs"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            if uploaded_file.type.startswith("image"):
                image = Image.open(uploaded_file)
                st.image(image, caption="Loaded Document Preview", use_container_width=True)
            else:
                st.info(f"📄 Document Loaded: {uploaded_file.name}")
            
            # Action button
            if st.button("🚀 Extract Data with DocuLIFT", type="primary", use_container_width=True):
                with st.spinner("Analyzing document layout and extracting entities..."):
                    try:
                        # Save temp file
                        temp_path = Path(f"./data/temp/{uploaded_file.name}")
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Load extractor and process
                        extractor = load_extractor(ocr_engine, languages, use_gpu)
                        result = cast(InvoiceData, extractor.extract(temp_path))
                        
                        # Store result in session state
                        st.session_state.result = result
                        st.session_state.success = True
                        
                        # Cleanup
                        temp_path.unlink(missing_ok=True)
                        
                    except Exception as e:
                        st.error(f"DocuLIFT Processing Error: {str(e)}")
                        st.session_state.success = False
    
    with col2:
        st.subheader("📊 Extracted Insights")
        
        if 'result' in st.session_state and st.session_state.get('success'):
            result: InvoiceData = st.session_state.result
            
            # Confidence score badge
            confidence = result.confidence_score * 100
            if confidence >= 70:
                st.success(f"🎯 Parsing Precision Score: {confidence:.1f}%")
            elif confidence >= 50:
                st.warning(f"⚠️ Parsing Precision Score: {confidence:.1f}%")
            else:
                st.error(f"❌ Parsing Precision Score: {confidence:.1f}%")
            
            # KPI Metrics Display
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Doc ID", result.invoice_number or "N/A")
            with metrics_col2:
                st.metric("Issued Date", result.invoice_date or "N/A")
            with metrics_col3:
                total_str = f"{result.currency} {result.total:,.2f}" if result.total else "N/A"
                st.metric("Grand Total", total_str)
            
            st.divider()
            
            # Categorized View Tabs
            tab1, tab2, tab3 = st.tabs(["📑 Extracted Schema", "📦 Line Items", "📜 Raw Vision Text"])
            
            with tab1:
                st.markdown("#### 🏢 Merchant & Entity Info")
                vendor_data = {
                    "Attribute": ["Merchant Name", "Address", "Tax Registration"],
                    "Value": [
                        result.vendor_name or "—",
                        result.vendor_address or "—",
                        result.vendor_tax_id or "—"
                    ]
                }
                st.table(pd.DataFrame(vendor_data))
                
                st.markdown("#### 📄 Document Metadata")
                invoice_data = {
                    "Attribute": ["Invoice ID / No.", "Issue Date", "Payment Due"],
                    "Value": [
                        result.invoice_number or "—",
                        result.invoice_date or "—",
                        result.due_date or "—"
                    ]
                }
                st.table(pd.DataFrame(invoice_data))
                
                st.markdown("#### 💰 Financial Breakdown")
                financial_data = {
                    "Attribute": ["Subtotal", "Tax Amount", "Discount", "Grand Total"],
                    "Value": [
                        f"{result.currency} {result.subtotal:,.2f}" if result.subtotal else "—",
                        f"{result.currency} {result.tax_amount:,.2f}" if result.tax_amount else "—",
                        f"{result.currency} {result.discount:,.2f}" if result.discount else "—",
                        f"{result.currency} {result.total:,.2f}" if result.total else "—"
                    ]
                }
                st.table(pd.DataFrame(financial_data))
            
            with tab2:
                if result.line_items:
                    items_data = []
                    for item in result.line_items:
                        items_data.append({
                            "Item Description": item.description,
                            "Qty": item.quantity or "—",
                            "Unit Price": f"{item.unit_price:,.2f}" if item.unit_price else "—",
                            "Line Total": f"{item.amount:,.2f}" if item.amount else "—"
                        })
                    st.dataframe(pd.DataFrame(items_data), use_container_width=True)
                else:
                    st.info("No structured line items were detected in this document.")
            
            with tab3:
                st.text_area("Full OCR Stream Output", result.raw_text, height=280)
            
            st.divider()
            
            # Data Export Hub
            st.markdown("#### 💾 Export Parsed Payload")
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                json_data = result.to_json(indent=2)
                st.download_button(
                    label="📥 Export as JSON",
                    data=json_data,
                    file_name="doculift_extracted.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with export_col2:
                csv_data = pd.DataFrame([{
                    "vendor_name": result.vendor_name,
                    "invoice_number": result.invoice_number,
                    "invoice_date": result.invoice_date,
                    "due_date": result.due_date,
                    "subtotal": result.subtotal,
                    "tax": result.tax_amount,
                    "total": result.total,
                    "currency": result.currency,
                }])
                st.download_button(
                    label="📊 Export as CSV",
                    data=csv_data.to_csv(index=False),
                    file_name="doculift_extracted.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("👈 Upload a document on the left panel and click 'Extract Data with DocuLIFT' to begin.")


if __name__ == "__main__":
    main()