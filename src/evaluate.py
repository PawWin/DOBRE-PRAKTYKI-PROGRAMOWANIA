import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from .pipeline import AnnotationBasedPipeline, PlateRecognitionPipeline, resize_keep_aspect
from .utils import calculate_accuracy, calculate_final_grade, normalize_plate_text, iou


@dataclass
class EvaluationResult:
    total_images: int
    correct_predictions: int
    accuracy_percent: float
    processing_time_sec: float
    time_per_image_ms: float
    final_grade: float
    predictions: list[dict]
    mean_iou: float | None = None
    
    def __str__(self) -> str:
        parts = [
            f"\n{'='*60}\n",
            "EVALUATION RESULTS\n",
            f"{'='*60}\n",
            f"Total images:       {self.total_images}\n",
            f"Correct predictions: {self.correct_predictions}\n",
            f"Accuracy:           {self.accuracy_percent:.2f}%\n",
            f"Processing time:    {self.processing_time_sec:.2f}s (for 100 images)\n",
            f"Time per image:     {self.time_per_image_ms:.2f}ms\n",
        ]
        if self.mean_iou is not None:
            parts.append(f"IoU (mean):        {self.mean_iou:.4f}\n")
        parts.extend([
            f"{'='*60}\n",
            f"FINAL GRADE:        {self.final_grade}\n",
            f"{'='*60}\n",
        ])
        return "".join(parts)
    
    def to_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "correct_predictions": self.correct_predictions,
            "accuracy_percent": self.accuracy_percent,
            "processing_time_sec": self.processing_time_sec,
            "time_per_image_ms": self.time_per_image_ms,
            "final_grade": self.final_grade,
            "predictions": self.predictions,
            "mean_iou": self.mean_iou,
        }


@dataclass
class ImageAnnotation:
    image_name: str
    width: int
    height: int
    bbox: tuple[float, float, float, float]  # xtl, ytl, xbr, ybr
    plate_text: str


class CVATDatasetLoader:
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.photos_dir = self.dataset_path / "photos"
        self.annotations_file = self.dataset_path / "annotations.xml"
        
        self.annotations = self._parse_annotations()
    
    def _parse_annotations(self) -> dict[str, ImageAnnotation]:
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        tree = ET.parse(self.annotations_file)
        root = tree.getroot()
        
        annotations = {}
        
        for image_elem in root.findall(".//image"):
            image_name = image_elem.get("name")
            width = int(image_elem.get("width"))
            height = int(image_elem.get("height"))
            
            box_elem = image_elem.find("box")
            if box_elem is None:
                continue
            
            xtl = float(box_elem.get("xtl"))
            ytl = float(box_elem.get("ytl"))
            xbr = float(box_elem.get("xbr"))
            ybr = float(box_elem.get("ybr"))
            
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
        import random
        
        all_samples = self.load_samples()
        random.seed(seed)
        random.shuffle(all_samples)
        
        split_idx = int(len(all_samples) * (1 - test_ratio))
        train_samples = all_samples[:split_idx]
        test_samples = all_samples[split_idx:]
        
        return train_samples, test_samples


class Evaluator:
    def __init__(
        self,
        dataset_path: Path,
        use_gpu: bool = True,
        target_width: int | None = 1600,
        use_yolo: bool = False,
        yolo_conf: float = 0.25,
        yolo_model: Path | None = None,
        yolo_imgsz: int | None = None,
        padding: float = 0.20,
        det_thresh: float = 0.05,
        box_thresh: float = 0.05,
        rec_score: float = 0.10,
        top_k: int = 4,
        ocr_version: str = "PP-OCRv4",
        batch_size: int = 1,
        ocr_batch_size: int = 1,
    ):
        """
        Initialize evaluator.
        
        Args:
            dataset_path: Path to dataset root
            use_gpu: Whether to use GPU acceleration
            target_width: Resize input width before cropping (keeps aspect), None to disable
            use_yolo: Whether to use YOLO detection instead of GT boxes
            yolo_conf: Confidence threshold for YOLO detection
            yolo_model: Path to YOLO model (.pt)
            yolo_imgsz: Optional YOLO inference size
            det_thresh: OCR detection threshold
            box_thresh: OCR box threshold
            rec_score: OCR recognition score threshold
            top_k: Number of top text candidates to consider
            ocr_version: PaddleOCR version to use
            batch_size: Batch size for YOLO inference (use_yolo mode)
            ocr_batch_size: Batch size for OCR (use_yolo mode)
        """
        self.loader = CVATDatasetLoader(dataset_path)
        self.pipeline_gt = AnnotationBasedPipeline(
            use_gpu=use_gpu,
            target_width=target_width,
            crop_padding=padding,
            det_thresh=det_thresh,
            box_thresh=box_thresh,
            rec_score=rec_score,
            top_k=top_k,
            ocr_version=ocr_version,
        )
        self.pipeline_yolo = PlateRecognitionPipeline(
            use_gpu=use_gpu,
            target_width=target_width,
            conf_threshold=yolo_conf,
            detector_model=str(yolo_model) if yolo_model else None,
            imgsz=yolo_imgsz,
            padding=padding,
            det_thresh=det_thresh,
            box_thresh=box_thresh,
            rec_score=rec_score,
            top_k=top_k,
            ocr_version=ocr_version,
        )
        self.use_yolo = use_yolo
        self.batch_size = max(1, batch_size)
        self.ocr_batch_size = max(1, ocr_batch_size)
    
    def evaluate(
        self,
        num_samples: int = 100,
        verbose: bool = True,
        use_test_split: bool = False,
        use_yolo: bool | None = None,
    ) -> EvaluationResult:
        effective_use_yolo = self.use_yolo if use_yolo is None else use_yolo

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
        iou_values = []
        
        start_time = time.time()
        
        preloaded = self._preload_samples(samples)

        if effective_use_yolo and self.batch_size > 1:
            chunk = []
            chunk_samples = []
            iterator = tqdm(preloaded, desc="Processing (batch)") if verbose else preloaded
            for sample, image in iterator:
                img_resized, sx, sy = resize_keep_aspect(image, self.pipeline_yolo.target_width)
                chunk.append((img_resized, sx, sy))
                chunk_samples.append(sample)
                if len(chunk) == self.batch_size:
                    self._process_yolo_chunk(chunk, chunk_samples, predictions, ground_truths, predictions_list, iou_values)
                    chunk, chunk_samples = [], []
            if chunk:
                self._process_yolo_chunk(chunk, chunk_samples, predictions, ground_truths, predictions_list, iou_values)
        else:
            iterator = tqdm(preloaded, desc="Processing") if verbose else preloaded
            for sample, image in iterator:
                if effective_use_yolo:
                    # YOLO detection + OCR (single)
                    detection, sx, sy = self.pipeline_yolo.process_image(image, return_scale=True)
                    if detection:
                        predicted_text = detection["text"]
                        pred_bbox = detection["bbox"]
                        if pred_bbox:
                            gt_bbox = sample["bbox"]
                            x1g, y1g, x2g, y2g = gt_bbox
                            gt_scaled = (x1g * sx, y1g * sy, x2g * sx, y2g * sy)
                            iou_val = iou(pred_bbox, gt_scaled)
                            iou_values.append(iou_val)
                    else:
                        predicted_text = ""
                        iou_values.append(0.0)
                else:
                    # GT bbox path
                    bbox = sample["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)
                    predicted_text = self.pipeline_gt.process_with_bbox(image, (x1, y1, x2, y2))
                
                ground_truth = sample["plate_text"]
                predictions.append(predicted_text)
                ground_truths.append(ground_truth)
                predictions_list.append({
                    "image": str(sample["image_path"].name),
                    "ground_truth": ground_truth,
                    "prediction": predicted_text,
                    "normalized_gt": normalize_plate_text(ground_truth),
                    "normalized_pred": normalize_plate_text(predicted_text),
                    "correct": normalize_plate_text(predicted_text) == normalize_plate_text(ground_truth),
                })
        
        total_time = time.time() - start_time
        
        actual_samples = len(predictions)
        time_per_100 = (total_time / actual_samples) * 100 if actual_samples > 0 else 0
        
        accuracy = calculate_accuracy(predictions, ground_truths)
        
        grade = calculate_final_grade(accuracy, time_per_100)
        
        correct = sum(1 for p in predictions_list if p["correct"])
        
        mean_iou = (sum(iou_values) / len(iou_values)) if iou_values else None

        return EvaluationResult(
            total_images=actual_samples,
            correct_predictions=correct,
            accuracy_percent=accuracy,
            processing_time_sec=time_per_100,
            time_per_image_ms=(total_time / actual_samples * 1000) if actual_samples > 0 else 0,
            final_grade=grade,
            predictions=predictions_list,
            mean_iou=mean_iou,
        )

    def _preload_samples(self, samples: list[dict]) -> list[tuple[dict, np.ndarray]]:
        loaded: list[tuple[dict, np.ndarray]] = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(cv2.imread, str(sample["image_path"])) for sample in samples]
            for sample, future in zip(samples, futures):
                image = future.result()
                if image is not None:
                    loaded.append((sample, image))
        return loaded

    def _process_yolo_chunk(
        self,
        chunk: list[tuple[np.ndarray, float, float]],
        chunk_samples: list[dict],
        predictions: list[str],
        ground_truths: list[str],
        predictions_list: list[dict],
        iou_values: list[float],
    ) -> None:
        images = [item[0] for item in chunk]
        scales = [(item[1], item[2]) for item in chunk]
        detections_batch = self.pipeline_yolo.detector.detect_batch(images)
        crops = []
        crop_info = []

        for det, (sx, sy), sample, img_resized in zip(detections_batch, scales, chunk_samples, images):
            if det:
                x1, y1, x2, y2 = det["bbox"]
                h, w = img_resized.shape[:2]
                box_w = x2 - x1
                box_h = y2 - y1
                pad_x = int(box_w * self.pipeline_yolo.padding)
                pad_y = int(box_h * self.pipeline_yolo.padding)
                x1_p = max(0, x1 - pad_x)
                y1_p = max(0, y1 - pad_y)
                x2_p = min(w, x2 + pad_x)
                y2_p = min(h, y2 + pad_y)
                crop = img_resized[y1_p:y2_p, x1_p:x2_p]

                crops.append(crop)
                crop_info.append((sample, sx, sy, (x1_p, y1_p, x2_p, y2_p)))
            else:
                # no detection for this sample
                crops.append(None)
                crop_info.append((sample, sx, sy, None))

        # OCR in batch if enabled
        texts = []
        if self.ocr_batch_size > 1:
            batch_crops = [c for c in crops if c is not None]
            if batch_crops:
                texts_batch = []
                for i in range(0, len(batch_crops), self.ocr_batch_size):
                    sub = batch_crops[i : i + self.ocr_batch_size]
                    texts_batch.extend(self.pipeline_yolo.ocr.recognize_list(sub))
                # Map back
                idx = 0
                for c in crops:
                    if c is None:
                        texts.append("")
                    else:
                        texts.append(texts_batch[idx] if idx < len(texts_batch) else "")
                        idx += 1
            else:
                texts = ["" for _ in crops]
        else:
            for c in crops:
                texts.append(self.pipeline_yolo.ocr.recognize(c) if c is not None else "")

        # Collect results and IoU
        for text, (sample, sx, sy, pred_bbox) in zip(texts, crop_info):
            if pred_bbox:
                gt_bbox = sample["bbox"]
                x1g, y1g, x2g, y2g = gt_bbox
                gt_scaled = (x1g * sx, y1g * sy, x2g * sx, y2g * sy)
                iou_val = iou(pred_bbox, gt_scaled)
                iou_values.append(iou_val)
            else:
                iou_values.append(0.0)

            ground_truth = sample["plate_text"]
            predictions.append(text)
            ground_truths.append(ground_truth)
            predictions_list.append({
                "image": str(sample["image_path"].name),
                "ground_truth": ground_truth,
                "prediction": text,
                "normalized_gt": normalize_plate_text(ground_truth),
                "normalized_pred": normalize_plate_text(text),
                "correct": normalize_plate_text(text) == normalize_plate_text(ground_truth),
            })
    
    def save_results(self, result: EvaluationResult, output_path: Path) -> None:
        """Save evaluation results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to: {output_path}")
