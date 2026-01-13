from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class PlateDetector:
    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.25, imgsz: int | None = None):
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model: {model_path}")
        else:
            # Use YOLOv8n as base - will download if not present
            # For license plates, we'll use a general model and crop generously
            self.model = YOLO("yolov8n.pt")
            print("Using default YOLOv8n.pt (consider training a plate-specific model)")
        
        self.device = "cuda" if self._check_cuda() else "cpu"
        print(f"PlateDetector using device: {self.device}")
    
    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def detect(self, image: np.ndarray) -> dict | None:
        results = self.model(
            image,
            device=self.device,
            verbose=False,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            save=False,
        )
        return results[0].boxes[0].xyxy[0].tolist()
    
    def detect_batch(self, images: list[np.ndarray]) -> list[dict | None]:
        results = self.model(
            images,
            device=self.device,
            verbose=False,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            save=False,
        )
        
        all_detections = []
        for result, image in zip(results, images):
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                all_detections.append({"bbox": (x1, y1, x2, y2)})
            else:
                all_detections.append(None)
        
        return all_detections


class SimplePlateDetector:
    def __init__(self):
        """Initialize simple detector."""
        pass
    
    def crop_from_bbox(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2].copy()
    
    def crop_from_annotation(
        self, 
        image: np.ndarray, 
        annotation: dict,
        padding_percent: float = 0.05
    ) -> np.ndarray:
        h, w = image.shape[:2]
        
        x_center = annotation["x_center"] * w
        y_center = annotation["y_center"] * h
        box_w = annotation["width"] * w
        box_h = annotation["height"] * h
        
        # Add padding
        box_w *= (1 + padding_percent)
        box_h *= (1 + padding_percent)
        
        x1 = max(0, int(x_center - box_w / 2))
        y1 = max(0, int(y_center - box_h / 2))
        x2 = min(w, int(x_center + box_w / 2))
        y2 = min(h, int(y_center + box_h / 2))
        
        return image[y1:y2, x1:x2].copy()
