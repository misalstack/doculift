"""
LayoutLM Extractor - Document understanding using LayoutLMv3
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LayoutLMExtractor:
    """
    Extract structured information using LayoutLMv3 model
    
    LayoutLMv3 is a multimodal model that combines text, layout (bounding boxes),
    and image features for document understanding tasks.
    """
    
    # Field labels for token classification
    LABELS = [
        "O",  # Outside
        "B-VENDOR_NAME", "I-VENDOR_NAME",
        "B-VENDOR_ADDRESS", "I-VENDOR_ADDRESS",
        "B-INVOICE_NUMBER", "I-INVOICE_NUMBER",
        "B-INVOICE_DATE", "I-INVOICE_DATE",
        "B-DUE_DATE", "I-DUE_DATE",
        "B-TOTAL", "I-TOTAL",
        "B-SUBTOTAL", "I-SUBTOTAL",
        "B-TAX", "I-TAX",
        "B-LINE_ITEM", "I-LINE_ITEM",
    ]
    
    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        use_gpu: bool = False,
        max_length: int = 512,
    ):
        """
        Initialize LayoutLM extractor
        
        Args:
            model_name: HuggingFace model name or path
            use_gpu: Whether to use GPU
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.max_length = max_length
        
        self._model: Any = None
        self._processor: Any = None
        self._device: Any = None
    
    def _initialize(self):
        """Initialize model and processor"""
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                LayoutLMv3Processor,
                LayoutLMv3ForTokenClassification,
            )
        except ImportError:
            raise ImportError(
                "Required packages not installed. "
                "Install with: pip install torch transformers"
            )
        
        logger.info(f"Loading LayoutLMv3 model: {self.model_name}")
        
        # Determine device
        if self.use_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        
        # Load processor and model
        self._processor = LayoutLMv3Processor.from_pretrained(
            self.model_name,
            apply_ocr=False,  # We provide our own OCR results
        )
        
        # Try to load fine-tuned model, fall back to base
        try:
            self._model = LayoutLMv3ForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.LABELS),
            )
        except Exception:
            logger.warning("Loading base model without fine-tuned weights")
            self._model = LayoutLMv3ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base",
                num_labels=len(self.LABELS),
            )
        
        self._model.to(self._device)
        self._model.eval()
        
        logger.info(f"LayoutLMv3 loaded on {self._device}")
    
    def extract(
        self,
        image: np.ndarray,
        words: List[str],
        boxes: List[List[float]],
    ) -> Dict[str, Any]:
        """
        Extract fields from document using LayoutLMv3
        
        Args:
            image: Document image as numpy array
            words: List of words from OCR
            boxes: List of bounding boxes [x_min, y_min, x_max, y_max]
            
        Returns:
            Dictionary of extracted fields
        """
        if self._model is None:
            self._initialize()
        
        import torch  # type: ignore
        from PIL import Image
        import cv2
        
        # Convert image to PIL
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        pil_image = Image.fromarray(image_rgb)
        
        # Normalize bounding boxes to 0-1000 range
        height, width = image.shape[:2]
        normalized_boxes = self._normalize_boxes(boxes, width, height)
        
        # Prepare inputs
        encoding = self._processor(
            pil_image,
            words,
            boxes=normalized_boxes,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Move to device
        encoding = {k: v.to(self._device) for k, v in encoding.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self._model(**encoding)
        
        # Get predictions
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        
        # Extract fields from predictions
        return self._parse_predictions(words, predictions, normalized_boxes)
    
    def _normalize_boxes(
        self,
        boxes: List[List[float]],
        width: int,
        height: int,
    ) -> List[List[int]]:
        """Normalize bounding boxes to 0-1000 range"""
        normalized = []
        for box in boxes:
            normalized.append([
                int(box[0] * 1000 / width),
                int(box[1] * 1000 / height),
                int(box[2] * 1000 / width),
                int(box[3] * 1000 / height),
            ])
        return normalized
    
    def _parse_predictions(
        self,
        words: List[str],
        predictions: List[int],
        boxes: List[List[int]],
    ) -> Dict[str, Any]:
        """Parse model predictions into structured fields"""
        result = {
            'vendor_name': None,
            'vendor_address': None,
            'invoice_number': None,
            'invoice_date': None,
            'due_date': None,
            'total': None,
            'subtotal': None,
            'tax': None,
            'line_items': [],
            'confidence': 0.0,
        }
        
        # Group consecutive tokens by field
        current_field = None
        current_tokens = []
        field_texts = {}
        
        for i, (word, pred_id) in enumerate(zip(words, predictions)):
            if pred_id >= len(self.LABELS):
                continue
                
            label = self.LABELS[pred_id]
            
            if label == "O":
                if current_field and current_tokens:
                    field_name = current_field.lower()
                    if field_name not in field_texts:
                        field_texts[field_name] = []
                    field_texts[field_name].append(" ".join(current_tokens))
                current_field = None
                current_tokens = []
            elif label.startswith("B-"):
                # Save previous field
                if current_field and current_tokens:
                    field_name = current_field.lower()
                    if field_name not in field_texts:
                        field_texts[field_name] = []
                    field_texts[field_name].append(" ".join(current_tokens))
                
                # Start new field
                current_field = label[2:]
                current_tokens = [word]
            elif label.startswith("I-"):
                if current_field == label[2:]:
                    current_tokens.append(word)
        
        # Don't forget last field
        if current_field and current_tokens:
            field_name = current_field.lower()
            if field_name not in field_texts:
                field_texts[field_name] = []
            field_texts[field_name].append(" ".join(current_tokens))
        
        # Map to result
        for field_name, texts in field_texts.items():
            text = " ".join(texts)
            
            if field_name == "vendor_name":
                result['vendor_name'] = text
            elif field_name == "vendor_address":
                result['vendor_address'] = text
            elif field_name == "invoice_number":
                result['invoice_number'] = text
            elif field_name == "invoice_date":
                result['invoice_date'] = text
            elif field_name == "due_date":
                result['due_date'] = text
            elif field_name == "total":
                result['total'] = self._parse_amount(text)
            elif field_name == "subtotal":
                result['subtotal'] = self._parse_amount(text)
            elif field_name == "tax":
                result['tax'] = self._parse_amount(text)
        
        # Calculate confidence based on fields extracted
        filled_fields = sum(1 for v in result.values() if v is not None and v != [])
        result['confidence'] = filled_fields / (len(result) - 1)  # -1 for confidence itself
        
        return result
    
    def _parse_amount(self, text: str) -> Optional[float]:
        """Parse monetary amount from text"""
        import re
        
        # Remove currency symbols and extract number
        cleaned = re.sub(r'[^\d.,]', '', text)
        cleaned = cleaned.replace(',', '')
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def train(
        self,
        train_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        output_dir: str = "./models/layoutlm",
    ):
        """
        Fine-tune LayoutLMv3 on invoice data
        
        Args:
            train_data: List of training examples
            val_data: List of validation examples
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            output_dir: Directory to save model
        """
        # Implementation for training
        # This would require proper dataset preparation
        raise NotImplementedError(
            "Training requires prepared dataset. "
            "See documentation for dataset format."
        )
