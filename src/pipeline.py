"""Main pipeline combining detection and OCR."""

from pathlib import Path

import cv2
import numpy as np

from .detector import PlateDetector, SimplePlateDetector
from .ocr import PlateOCR


def resize_keep_aspect(image: np.ndarray, target_width: int | None = None) -> tuple[np.ndarray, float, float]:
    """Resize image keeping aspect ratio. Returns resized image and scale factors (sx, sy)."""
    if target_width is None:
        return image, 1.0, 1.0
    h, w = image.shape[:2]
    if w <= target_width:
        return image, 1.0, 1.0
    scale = target_width / w
    new_w = target_width
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale, scale


class PlateRecognitionPipeline:
    """
    Complete pipeline for license plate detection and OCR.
    
    Combines YOLOv8 detection with PaddleOCR text recognition.
    """
    
    def __init__(
        self,
        detector_model: str | None = None,
        use_gpu: bool = True,
        conf_threshold: float = 0.25,
        target_width: int | None = 1600,
        det_thresh: float = 0.05,
        box_thresh: float = 0.05,
        rec_score: float = 0.10,
        min_score: float = 0.10,
        top_k: int = 4,
        padding: float = 0.20,
        imgsz: int | None = None,
        ocr_version: str = "PP-OCRv4",
    ):
        """
        Initialize the pipeline.
        
        Args:
            detector_model: Path to custom YOLO model for detection
            use_gpu: Whether to use GPU acceleration
            conf_threshold: Detection confidence threshold
            target_width: Resize input width before detection (keeps aspect), None to disable
            det_thresh: OCR detection threshold
            box_thresh: OCR box threshold
            rec_score: OCR recognition score threshold
            min_score: Minimum score for candidate filtering
            top_k: Number of top text candidates to consider
            padding: Padding (fraction) around detected box before OCR
            imgsz: Optional YOLO inference size
            ocr_version: PaddleOCR version to use
        """
        self.detector = PlateDetector(
            model_path=detector_model,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
        )
        self.ocr = PlateOCR(
            use_gpu=use_gpu,
            ocr_version=ocr_version,
            text_det_thresh=det_thresh,
            text_det_box_thresh=box_thresh,
            text_rec_score_thresh=rec_score,
            min_score=min_score,
            top_k=top_k,
        )
        self.target_width = target_width
        self.padding = padding
    
    def process_image(
        self,
        image: np.ndarray,
        return_scale: bool = False,
        save: bool = False,
        save_dir: str | None = None,
    ) -> list[dict] | tuple[list[dict], float, float]:
        """
        Process a single image to detect and recognize license plates.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of results with keys: 'bbox', 'text', 'confidence'
        """
        # Optional resize for faster/cleaner detection
        image_resized, sx, sy = resize_keep_aspect(image, self.target_width)

        # Detect plates
        detections = self.detector.detect(image_resized, save_dir=save_dir, save=save)
        
        results = []
        for det in detections:
            # Apply padding to bbox before OCR
            x1, y1, x2, y2 = det["bbox"]
            h, w = image_resized.shape[:2]
            box_w = x2 - x1
            box_h = y2 - y1
            pad_x = int(box_w * self.padding)
            pad_y = int(box_h * self.padding)
            x1_p = max(0, x1 - pad_x)
            y1_p = max(0, y1 - pad_y)
            x2_p = min(w, x2 + pad_x)
            y2_p = min(h, y2 + pad_y)
            crop = image_resized[y1_p:y2_p, x1_p:x2_p]

            # Run OCR on each detected plate
            text = self.ocr.recognize(crop)
            results.append({
                "bbox": (x1_p, y1_p, x2_p, y2_p),
                "text": text,
                "confidence": det["confidence"]
            })
        
        if return_scale:
            return results, sx, sy
        return results
    
    def process_file(self, image_path: str | Path, return_scale: bool = False) -> list[dict] | tuple[list[dict], float, float]:
        """
        Process an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of results with keys: 'bbox', 'text', 'confidence'
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        return self.process_image(image, return_scale=return_scale)


class AnnotationBasedPipeline:
    """
    Pipeline that uses ground truth bounding boxes for detection.
    
    This is useful for:
    - Evaluating OCR independently of detection
    - Using provided annotations from the dataset
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        crop_padding: float = 0.20,
        target_width: int | None = 1600,
        det_thresh: float = 0.05,
        box_thresh: float = 0.05,
        rec_score: float = 0.10,
        min_score: float = 0.10,
        top_k: int = 4,
        ocr_version: str = "PP-OCRv4",
    ):
        """
        Initialize the pipeline.
        
        Args:
            use_gpu: Whether to use GPU acceleration for OCR
            crop_padding: Padding around the crop as fraction of bbox size
            target_width: Resize input width before cropping (keeps aspect), None to disable
            det_thresh: OCR detection threshold
            box_thresh: OCR box threshold
            rec_score: OCR recognition score threshold
            min_score: Minimum score for candidate filtering
            top_k: Number of top text candidates to consider
            ocr_version: PaddleOCR version to use
        """
        self.detector = SimplePlateDetector()
        self.ocr = PlateOCR(
            use_gpu=use_gpu,
            ocr_version=ocr_version,
            text_det_thresh=det_thresh,
            text_det_box_thresh=box_thresh,
            text_rec_score_thresh=rec_score,
            min_score=min_score,
            top_k=top_k,
        )
        self.crop_padding = crop_padding
        self.target_width = target_width
    
    def process_with_annotation(
        self,
        image: np.ndarray,
        annotation: dict,
        padding_percent: float | None = None
    ) -> str:
        """
        Process image using annotation bounding box.
        
        Args:
            image: BGR image as numpy array
            annotation: Dict with 'x_center', 'y_center', 'width', 'height'
            padding_percent: Extra padding around crop (overrides default)
            
        Returns:
            Recognized text
        """
        padding = padding_percent if padding_percent is not None else self.crop_padding

        # Resize image (annotation is normalized, so scale not needed)
        image_resized, _, _ = resize_keep_aspect(image, self.target_width)

        # Crop using annotation
        crop = self.detector.crop_from_annotation(image_resized, annotation, padding)
        
        # Run OCR
        text = self.ocr.recognize(crop)
        
        return text
    
    def process_with_bbox(
        self, 
        image: np.ndarray, 
        bbox: tuple[int, int, int, int],
        padding: float | None = None
    ) -> str:
        """
        Process image using pixel bounding box with padding.
        
        Args:
            image: BGR image as numpy array
            bbox: Bounding box as (x1, y1, x2, y2)
            padding: Padding as fraction of bbox size (overrides default)
            
        Returns:
            Recognized text
        """
        pad = padding if padding is not None else self.crop_padding
        
        image_resized, sx, sy = resize_keep_aspect(image, self.target_width)

        # Apply padding
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * sx)
        x2 = int(x2 * sx)
        y1 = int(y1 * sy)
        y2 = int(y2 * sy)
        h, w = image_resized.shape[:2]
        
        box_w = x2 - x1
        box_h = y2 - y1
        
        pad_x = int(box_w * pad)
        pad_y = int(box_h * pad)
        
        # Expand bbox with padding
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(w, x2 + pad_x)
        y2_padded = min(h, y2 + pad_y)
        
        crop = image_resized[y1_padded:y2_padded, x1_padded:x2_padded]
        
        text = self.ocr.recognize(crop)
        return text
    
    def process_file_with_annotation(
        self,
        image_path: str | Path,
        annotation: dict,
        padding_percent: float | None = None
    ) -> str:
        """
        Process image file using annotation.
        
        Args:
            image_path: Path to image file
            annotation: Dict with annotation data
            padding_percent: Extra padding around crop
            
        Returns:
            Recognized text
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        return self.process_with_annotation(image, annotation, padding_percent)
