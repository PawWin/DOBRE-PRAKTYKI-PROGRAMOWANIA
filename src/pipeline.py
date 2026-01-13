from pathlib import Path

import cv2
import numpy as np

from .detector import PlateDetector, SimplePlateDetector
from .ocr import PlateOCR


def resize_keep_aspect(image: np.ndarray, target_width: int | None = None) -> tuple[np.ndarray, float, float]:
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
    def __init__(
        self,
        detector_model: str | None = None,
        use_gpu: bool = True,
        conf_threshold: float = 0.25,
        target_width: int | None = 1600,
        det_thresh: float = 0.05,
        box_thresh: float = 0.05,
        rec_score: float = 0.10,
        top_k: int = 4,
        padding: float = 0.20,
        imgsz: int | None = None,
        ocr_version: str = "PP-OCRv4",
    ):
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
            top_k=top_k,
        )
        self.target_width = target_width
        self.padding = padding
    
    def process_image(
        self,
        image: np.ndarray,
        return_scale: bool = False,
    ) -> dict | None | tuple[dict | None, float, float]:
        image_resized, sx, sy = resize_keep_aspect(image, self.target_width)

        det = self.detector.detect(image_resized)

        if det:
            x1, y1, x2, y2 = map(int, det)
            h, w = image_resized.shape[:2]
            box_w = x2 - x1
            box_h = y2 - y1
            pad_x = int(box_w * self.padding)
            pad_y = int(box_h * self.padding)
            x1_p = max(0, x1 - pad_x)
            y1_p = max(0, y1 - pad_y)
            x2_p = min(w, x2 + pad_x)
            y2_p = min(h, y2 + pad_y)
            bbox_padded = (x1_p, y1_p, x2_p, y2_p)
            crop = image_resized[y1_p:y2_p, x1_p:x2_p]

            text = self.ocr.recognize(crop)
            det_result = {
                # IoU should use the raw detector box, not the padded crop box
                "bbox": (x1, y1, x2, y2),
                "bbox_padded": bbox_padded,
                "text": text,
            }
        else:
            det_result = None
        
        if return_scale:
            return det_result, sx, sy
        return det_result
    
    def process_file(self, image_path: str | Path, return_scale: bool = False) -> dict | None | tuple[dict | None, float, float]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        return self.process_image(image, return_scale=return_scale)


class AnnotationBasedPipeline:
    def __init__(
        self,
        use_gpu: bool = True,
        crop_padding: float = 0.20,
        target_width: int | None = 1600,
        det_thresh: float = 0.05,
        box_thresh: float = 0.05,
        rec_score: float = 0.10,
        top_k: int = 4,
        ocr_version: str = "PP-OCRv4",
    ):
        self.detector = SimplePlateDetector()
        self.ocr = PlateOCR(
            use_gpu=use_gpu,
            ocr_version=ocr_version,
            text_det_thresh=det_thresh,
            text_det_box_thresh=box_thresh,
            text_rec_score_thresh=rec_score,
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
        padding = padding_percent if padding_percent is not None else self.crop_padding

        image_resized, _, _ = resize_keep_aspect(image, self.target_width)

        crop = self.detector.crop_from_annotation(image_resized, annotation, padding)
        text = self.ocr.recognize(crop)
        
        return text
    
    def process_with_bbox(
        self, 
        image: np.ndarray, 
        bbox: tuple[int, int, int, int],
        padding: float | None = None
    ) -> str:
        pad = padding if padding is not None else self.crop_padding
        
        image_resized, sx, sy = resize_keep_aspect(image, self.target_width)

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
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        return self.process_with_annotation(image, annotation, padding_percent)
