"""
DocuLIFT - Gradio Interface for HuggingFace Spaces
AI-powered Invoice & Receipt Parser
"""

import gradio as gr
import json
import pandas as pd
from pathlib import Path
import sys
import os

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.extraction import InvoiceExtractor
from src.extraction.invoice_extractor import InvoiceData

# Cache the extractor globally (load once)
_extractor = None

def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = InvoiceExtractor(
            ocr_engine="paddleocr",
            languages=["en"],
            use_gpu=False,
        )
    return _extractor


def parse_invoice(image_path):
    """Main processing function called by Gradio."""
    if image_path is None:
        return (
            "Please upload an invoice or receipt image.",
            None,
            None,
            None,
        )

    try:
        extractor = get_extractor()
        result = extractor.extract(Path(image_path))

        # --- Summary text ---
        confidence = result.confidence_score * 100
        summary = f"""## ⚡ DocuLIFT Extraction Results

**Confidence Score:** {confidence:.1f}%

### 🏢 Vendor Info
- **Name:** {result.vendor_name or '—'}
- **Address:** {result.vendor_address or '—'}
- **Tax ID:** {result.vendor_tax_id or '—'}

### 📄 Invoice Details
- **Invoice No.:** {result.invoice_number or '—'}
- **Issue Date:** {result.invoice_date or '—'}
- **Due Date:** {result.due_date or '—'}

### 💰 Financials
- **Subtotal:** {f"{result.currency} {result.subtotal:,.2f}" if result.subtotal else '—'}
- **Tax:** {f"{result.currency} {result.tax_amount:,.2f}" if result.tax_amount else '—'}
- **Discount:** {f"{result.currency} {result.discount:,.2f}" if result.discount else '—'}
- **Total:** {f"{result.currency} {result.total:,.2f}" if result.total else '—'}
"""

        # --- Line items table ---
        if result.line_items:
            items_df = pd.DataFrame([{
                "Description": item.description,
                "Qty": item.quantity or "—",
                "Unit Price": f"{item.unit_price:,.2f}" if item.unit_price else "—",
                "Amount": f"{item.amount:,.2f}" if item.amount else "—",
            } for item in result.line_items])
        else:
            items_df = pd.DataFrame({"Info": ["No line items detected"]})

        # --- Raw OCR text ---
        raw_text = result.raw_text or "No OCR text extracted."

        # --- JSON export ---
        json_output = result.to_json(indent=2)

        return summary, items_df, raw_text, json_output

    except Exception as e:
        error_msg = f"## ❌ Processing Error\n\n```\n{str(e)}\n```"
        return error_msg, None, None, None


# ── Gradio UI ──────────────────────────────────────────────────
with gr.Blocks(
    title="DocuLIFT | Invoice Parser",
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="cyan",
    ),
    css="""
        .gradio-container { max-width: 1200px; margin: auto; }
        h1 { text-align: center; }
    """
) as demo:

    gr.Markdown("""
    # ⚡ DocuLIFT — Intelligent Invoice & Receipt Parser
    ### AI-powered document extraction using PaddleOCR + Rule-based NLP
    Upload an invoice or receipt image and DocuLIFT will extract all structured data automatically.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📤 Upload Invoice / Receipt",
                type="filepath",
                sources=["upload"],
            )
            parse_btn = gr.Button(
                "🚀 Extract Data with DocuLIFT",
                variant="primary",
                size="lg",
            )
            gr.Markdown("""
            **Supported formats:** PNG, JPG, JPEG, TIFF, BMP
            
            **What gets extracted:**
            - Vendor name, address & tax ID
            - Invoice number, dates
            - Subtotal, tax, discount, total
            - Line items table
            """)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📑 Extracted Data"):
                    summary_output = gr.Markdown(
                        value="*Upload an invoice and click Extract to see results.*"
                    )

                with gr.Tab("📦 Line Items"):
                    items_output = gr.Dataframe(
                        label="Line Items",
                        wrap=True,
                    )

                with gr.Tab("📜 Raw OCR Text"):
                    raw_text_output = gr.Textbox(
                        label="Raw OCR Output",
                        lines=12,
                        show_copy_button=True,
                    )

                with gr.Tab("📥 JSON Export"):
                    json_output = gr.Code(
                        label="Extracted JSON",
                        language="json",
                        lines=20,
                    )

    parse_btn.click(
        fn=parse_invoice,
        inputs=[image_input],
        outputs=[summary_output, items_output, raw_text_output, json_output],
        show_progress="full",
    )

    gr.Examples(
        examples=[],
        inputs=image_input,
    )

    gr.Markdown("""
    ---
    Built with ❤️ using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) • 
    [Gradio](https://gradio.app) • 
    [HuggingFace Spaces](https://huggingface.co/spaces)
    """)


if __name__ == "__main__":
    demo.launch()
