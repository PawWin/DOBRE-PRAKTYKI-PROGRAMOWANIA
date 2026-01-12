"""Evaluation module for license plate recognition."""

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from .pipeline import AnnotationBasedPipeline, PlateRecognitionPipeline
from .utils import calculate_accuracy, calculate_final_grade, normalize_plate_text


@dataclass
class EvaluationResult:
    """Results from evaluation."""
    
    total_images: int
    correct_predictions: int
    accuracy_percent: float
    processing_time_sec: float
    time_per_image_ms: float
    final_grade: float
    predictions: list[dict]
    
    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"EVALUATION RESULTS\n"
            f"{'='*60}\n"
            f"Total images:       {self.total_images}\n"
            f"Correct predictions: {self.correct_predictions}\n"
            f"Accuracy:           {self.accuracy_percent:.2f}%\n"
            f"Processing time:    {self.processing_time_sec:.2f}s (for 100 images)\n"
            f"Time per image:     {self.time_per_image_ms:.2f}ms\n"
            f"{'='*60}\n"
            f"FINAL GRADE:        {self.final_grade}\n"
            f"{'='*60}\n"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_images": self.total_images,
            "correct_predictions": self.correct_predictions,
            "accuracy_percent": self.accuracy_percent,
            "processing_time_sec": self.processing_time_sec,
            "time_per_image_ms": self.time_per_image_ms,
            "final_grade": self.final_grade,
            "predictions": self.predictions,
        }


@dataclass
class ImageAnnotation:
    """Annotation for a single image."""
    image_name: str
    width: int
    height: int
    bbox: tuple[float, float, float, float]  # xtl, ytl, xbr, ybr
    plate_text: str


class CVATDatasetLoader:
    """Load and parse the Poland Vehicle License Plate Dataset in CVAT XML format."""
    
    def __init__(self, dataset_path: Path):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Path to dataset root directory
        """
        self.dataset_path = Path(dataset_path)
        self.photos_dir = self.dataset_path / "photos"
        self.annotations_file = self.dataset_path / "annotations.xml"
        
        # Parse annotations
        self.annotations = self._parse_annotations()
    
    def _parse_annotations(self) -> dict[str, ImageAnnotation]:
        """Parse CVAT XML annotations file."""
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        tree = ET.parse(self.annotations_file)
        root = tree.getroot()
        
        annotations = {}
        
        for image_elem in root.findall(".//image"):
            image_name = image_elem.get("name")
            width = int(image_elem.get("width"))
            height = int(image_elem.get("height"))
            
            # Find the box element
            box_elem = image_elem.find("box")
            if box_elem is None:
                continue
            
            xtl = float(box_elem.get("xtl"))
            ytl = float(box_elem.get("ytl"))
            xbr = float(box_elem.get("xbr"))
            ybr = float(box_elem.get("ybr"))
            
            # Find plate number attribute
            plate_text = ""
            for attr_elem in box_elem.findall("attribute"):
                if attr_elem.get("name") == "plate number":
                    plate_text = attr_elem.text or ""
                    break
            
            annotations[image_name] = ImageAnnotation(
                image_name=image_name,
                width=width,
                height=height,
                bbox=(xtl, ytl, xbr, ybr),
                plate_text=plate_text,
            )
        
        return annotations
    
    def load_samples(self, limit: int | None = None, shuffle: bool = False) -> list[dict]:
        """
        Load samples from the dataset.
        
        Args:
            limit: Maximum number of samples to load
            shuffle: Whether to shuffle samples
            
        Returns:
            List of sample dicts with 'image_path', 'bbox', 'plate_text'
        """
        samples = []
        
        image_names = list(self.annotations.keys())
        
        if shuffle:
            import random
            random.shuffle(image_names)
        
        for image_name in image_names:
            if limit and len(samples) >= limit:
                break
            
            annotation = self.annotations[image_name]
            image_path = self.photos_dir / image_name
            
            if not image_path.exists():
                continue
            
            samples.append({
                "image_path": image_path,
                "bbox": annotation.bbox,
                "plate_text": annotation.plate_text,
                "width": annotation.width,
                "height": annotation.height,
            })
        
        return samples
    
    def get_train_test_split(
        self, 
        test_ratio: float = 0.3,
        seed: int = 42
    ) -> tuple[list[dict], list[dict]]:
        """
        Split dataset into train and test sets.
        
        Args:
            test_ratio: Ratio of samples for testing (default 0.3 = 30%)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_samples, test_samples)
        """
        import random
        
        all_samples = self.load_samples()
        random.seed(seed)
        random.shuffle(all_samples)
        
        split_idx = int(len(all_samples) * (1 - test_ratio))
        train_samples = all_samples[:split_idx]
        test_samples = all_samples[split_idx:]
        
        return train_samples, test_samples


class Evaluator:
    """Evaluator for license plate recognition system."""
    
    def __init__(
        self,
        dataset_path: Path,
        use_gpu: bool = True,
        target_width: int | None = 1600,
        use_yolo: bool = False,
        yolo_conf: float = 0.25,
    ):
        """
        Initialize evaluator.
        
        Args:
            dataset_path: Path to dataset root
            use_gpu: Whether to use GPU acceleration
            target_width: Resize input width before cropping (keeps aspect), None to disable
            use_yolo: Whether to use YOLO detection instead of GT boxes
            yolo_conf: Confidence threshold for YOLO detection
        """
        self.loader = CVATDatasetLoader(dataset_path)
        self.pipeline_gt = AnnotationBasedPipeline(
            use_gpu=use_gpu,
            target_width=target_width,
        )
        self.pipeline_yolo = PlateRecognitionPipeline(
            use_gpu=use_gpu,
            target_width=target_width,
            conf_threshold=yolo_conf,
        )
        self.use_yolo = use_yolo
    
    def evaluate(
        self,
        num_samples: int = 100,
        verbose: bool = True,
        use_test_split: bool = False,
        use_yolo: bool | None = None,
    ) -> EvaluationResult:
        """
        Run evaluation on the dataset.
        
        Args:
            num_samples: Number of samples to evaluate
            verbose: Whether to show progress bar
            use_test_split: Whether to use only test split (30% of data)
            use_yolo: Override to enable YOLO detection (default: initializer setting)
            
        Returns:
            EvaluationResult with all metrics
        """
        effective_use_yolo = self.use_yolo if use_yolo is None else use_yolo

        # Load samples
        if use_test_split:
            _, samples = self.loader.get_train_test_split(test_ratio=0.3)
            if len(samples) > num_samples:
                samples = samples[:num_samples]
        else:
            samples = self.loader.load_samples(limit=num_samples)
        
        if len(samples) < num_samples:
            print(f"Warning: Only {len(samples)} samples available (requested {num_samples})")
        
        predictions_list = []
        ground_truths = []
        predictions = []
        
        # Start timing
        start_time = time.time()
        
        # Process each sample
        iterator = tqdm(samples, desc="Processing") if verbose else samples
        for sample in iterator:
            image = cv2.imread(str(sample["image_path"]))
            if image is None:
                continue
            
            if effective_use_yolo:
                # YOLO detection + OCR
                detections = self.pipeline_yolo.process_image(image)
                if detections:
                    best = max(detections, key=lambda d: d.get("confidence", 0))
                    predicted_text = best.get("text", "")
                else:
                    predicted_text = ""
            else:
                # Get bbox (already in pixel coordinates)
                bbox = sample["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Run OCR with bbox
                predicted_text = self.pipeline_gt.process_with_bbox(image, (x1, y1, x2, y2))
            
            ground_truth = sample["plate_text"]
            
            predictions.append(predicted_text)
            ground_truths.append(ground_truth)
            
            # Store detailed prediction info
            predictions_list.append({
                "image": str(sample["image_path"].name),
                "ground_truth": ground_truth,
                "prediction": predicted_text,
                "normalized_gt": normalize_plate_text(ground_truth),
                "normalized_pred": normalize_plate_text(predicted_text),
                "correct": normalize_plate_text(predicted_text) == normalize_plate_text(ground_truth),
            })
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Scale time to 100 images for grading
        actual_samples = len(predictions)
        time_per_100 = (total_time / actual_samples) * 100 if actual_samples > 0 else 0
        
        # Calculate accuracy
        accuracy = calculate_accuracy(predictions, ground_truths)
        
        # Calculate grade
        grade = calculate_final_grade(accuracy, time_per_100)
        
        # Count correct
        correct = sum(1 for p in predictions_list if p["correct"])
        
        return EvaluationResult(
            total_images=actual_samples,
            correct_predictions=correct,
            accuracy_percent=accuracy,
            processing_time_sec=time_per_100,
            time_per_image_ms=(total_time / actual_samples * 1000) if actual_samples > 0 else 0,
            final_grade=grade,
            predictions=predictions_list,
        )
    
    def save_results(self, result: EvaluationResult, output_path: Path) -> None:
        """Save evaluation results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to: {output_path}")
