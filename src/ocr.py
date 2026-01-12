"""OCR module using PaddleOCR for license plate text recognition."""

import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR


class PlateOCR:
    """PaddleOCR-based license plate text recognizer."""
    
    def __init__(
        self,
        use_gpu: bool = True,
        lang: str = "en",
        ocr_version: str = "PP-OCRv4",
        text_det_thresh: float = 0.05,
        text_det_box_thresh: float = 0.05,
        text_rec_score_thresh: float = 0.0,
        min_score: float = 0.10,
        top_k: int = 4,
    ):
        """
        Initialize PaddleOCR.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            lang: Language for OCR ('en' works well for license plates)
            ocr_version: PaddleOCR version to use (default: best from tests - PP-OCRv4)
        """
        # Set device for PaddlePaddle
        if use_gpu and paddle.device.is_compiled_with_cuda():
            paddle.device.set_device("gpu")
            print(f"PlateOCR: Using GPU | version={ocr_version}")
        else:
            paddle.device.set_device("cpu")
            print(f"PlateOCR: Using CPU | version={ocr_version}")
        
        # PaddleOCR 3.x - optimized for license plates
        self.ocr = PaddleOCR(
            lang=lang,
            ocr_version=ocr_version,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_rec_score_thresh=text_rec_score_thresh,
        )
        
        self.min_score = min_score  # Minimum score for text
        self.top_k = top_k
    
    def recognize(self, image: np.ndarray, preprocess: bool = False) -> str:
        """
        Recognize text in a license plate image.
        
        Args:
            image: BGR image of the license plate crop
            preprocess: Whether to apply preprocessing
            
        Returns:
            Recognized text string
        """
        if image is None or image.size == 0:
            return ""
        
        # Optionally preprocess
        if preprocess:
            image = self._preprocess(image)
        
        # Run OCR using predict (new API)
        result = self.ocr.predict(image)
        
        # Extract text from results
        text = self._extract_text(result)
        
        return text
    
    def recognize_batch(self, images: list[np.ndarray]) -> list[str]:
        """
        Recognize text in multiple license plate images.
        
        Args:
            images: List of BGR images of license plate crops
            
        Returns:
            List of recognized text strings
        """
        results = []
        for image in images:
            text = self.recognize(image)
            results.append(text)
        return results
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: BGR image
            
        Returns:
            Preprocessed image
        """
        # Resize if too small
        h, w = image.shape[:2]
        if h < 50 or w < 100:
            scale = max(50 / h, 100 / w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR for PaddleOCR
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def _extract_text(self, result: list) -> str:
        """
        Extract text from PaddleOCR 3.x result.
        
        Args:
            result: PaddleOCR output
            
        Returns:
            Best recognized text string (highest scoring or longest)
        """
        if not result:
            return ""
        
        # PaddleOCR 3.x returns list of dicts
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            
            # New format: dict with rec_texts and rec_scores
            if isinstance(item, dict):
                texts = item.get("rec_texts", [])
                scores = item.get("rec_scores", [])
                
                if not texts:
                    return ""
                
                # Collect all texts with scores
                text_scores = []
                for text, score in zip(texts, scores):
                    text = text.strip()
                    if text and score >= self.min_score:
                        text_scores.append((text, score))
                
                if not text_scores:
                    # Fallback: get any non-empty text
                    text_scores = [(t.strip(), 0) for t in texts if t.strip()]
                
                if not text_scores:
                    return ""
                
                # Sort by score descending, then by length descending
                text_scores.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
                
                # Get top-k texts and pick the best one (longest plate-like)
                top_k = text_scores[: self.top_k]
                
                # Find the best plate-like text (alphanumeric, reasonable length)
                best_text = ""
                best_score = -1
                
                for text, score in top_k:
                    # License plates are typically 5-8 characters
                    # Prefer texts that look like plates
                    clean = ''.join(c for c in text if c.isalnum())
                    if len(clean) >= 5 and len(clean) <= 10:
                        if score > best_score or (score == best_score and len(clean) > len(best_text)):
                            best_text = text
                            best_score = score
                
                # If no plate-like text found, return the highest scoring one
                if not best_text and top_k:
                    best_text = top_k[0][0]
                
                return best_text
            
            # Old format fallback
            elif isinstance(item, (list, tuple)):
                texts = []
                for line in result[0]:
                    if line and len(line) >= 2:
                        text_data = line[1]
                        if isinstance(text_data, tuple) and len(text_data) >= 2:
                            text, confidence = text_data
                            texts.append(text)
                        elif isinstance(text_data, str):
                            texts.append(text_data)
                return "".join(texts)
        
        return ""


class DirectOCR:
    """
    Direct OCR without preprocessing - for comparison.
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize direct OCR."""
        if use_gpu and paddle.device.is_compiled_with_cuda():
            paddle.device.set_device("gpu")
        else:
            paddle.device.set_device("cpu")
            
        self.ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    
    def recognize(self, image: np.ndarray) -> str:
        """Recognize text directly without preprocessing."""
        if image is None or image.size == 0:
            return ""
        
        result = self.ocr.predict(image)
        
        if not result:
            return ""
        
        item = result[0]
        if isinstance(item, dict):
            texts = item.get("rec_texts", [])
            return "".join(t for t in texts if t.strip())
        
        return ""
