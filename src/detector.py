"""License plate detection using YOLOv8."""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class PlateDetector:
    """YOLOv8-based license plate detector."""
    
    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.25, imgsz: int | None = None):
        """
        Initialize the plate detector.
        
        Args:
            model_path: Path to custom YOLO model. If None, uses pretrained model.
            conf_threshold: Confidence threshold for detections
            imgsz: Optional inference size override
        """
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
        
        # Move to GPU if available
        self.device = "cuda" if self._check_cuda() else "cpu"
        print(f"PlateDetector using device: {self.device}")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def detect(self, image: np.ndarray, save_dir: str | None = None, save: bool = False) -> list[dict]:
        """
        Detect license plates in an image.
        
        Args:
            image: BGR image as numpy array
            save_dir: Optional directory to save annotated detections
            save: Whether to save annotated image
            
        Returns:
            List of detections with keys: 'bbox', 'confidence', 'crop'
        """
        project = None
        name = None
        if save and save_dir:
            # Ultralytics expects project/name; we use project=save_dir, name="eval"
            project = Path(save_dir).resolve()
            name = "eval"
        results = self.model(
            image,
            device=self.device,
            verbose=False,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            save=save,
            project=project if save else None,
            name=name if save else None,
            exist_ok=True,
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "crop": image[y1:y2, x1:x2],
                })
        
        return detections
    
    def detect_batch(self, images: list[np.ndarray], save_dir: str | None = None, save: bool = False) -> list[list[dict]]:
        """
        Detect license plates in a batch of images.
        
        Args:
            images: List of BGR images as numpy arrays
            save_dir: Optional directory to save annotated detections
            save: Whether to save annotated images
            
        Returns:
            List of detection lists (one per image)
        """
        project = None
        name = None
        if save and save_dir:
            project = Path(save_dir).resolve()
            name = "eval"
        results = self.model(
            images,
            device=self.device,
            verbose=False,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            save=save,
            project=project if save else None,
            name=name if save else None,
            exist_ok=True,
        )
        
        all_detections = []
        for result, image in zip(results, images):
            detections = []
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "crop": image[y1:y2, x1:x2],
                    })
            all_detections.append(detections)
        
        return all_detections


class SimplePlateDetector:
    """
    Simple license plate detector that uses the full image or annotation bounding boxes.
    
    This is useful when:
    - Ground truth bounding boxes are available
    - Testing OCR independently of detection
    """
    
    def __init__(self):
        """Initialize simple detector."""
        pass
    
    def crop_from_bbox(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop image using provided bounding box.
        
        Args:
            image: BGR image as numpy array
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Cropped image region
        """
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2].copy()
    
    def crop_from_annotation(
        self, 
        image: np.ndarray, 
        annotation: dict,
        padding_percent: float = 0.05
    ) -> np.ndarray:
        """
        Crop image using YOLO format annotation.
        
        Args:
            image: BGR image as numpy array
            annotation: Dict with 'x_center', 'y_center', 'width', 'height' (normalized 0-1)
            padding_percent: Extra padding around the crop
            
        Returns:
            Cropped image region
        """
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
