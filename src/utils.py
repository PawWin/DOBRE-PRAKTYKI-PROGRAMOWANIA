import re
from pathlib import Path


def normalize_plate_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^A-Za-z0-9]", "", text)
    return text.upper()


def calculate_accuracy(predictions: list[str], ground_truths: list[str]) -> float:
    if not ground_truths:
        return 0.0
    
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_normalized = normalize_plate_text(pred)
        gt_normalized = normalize_plate_text(gt)
        if pred_normalized == gt_normalized:
            correct += 1
    
    return (correct / len(ground_truths)) * 100


def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    
    accuracy_norm = (accuracy_percent - 60) / 40
    
    time_norm = (60 - processing_time_sec) / 50
    
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    
    grade = 2.0 + 3.0 * score
    
    return round(grade * 2) / 2


def get_image_files(directory: Path, extensions: tuple = (".jpg", ".jpeg", ".png")) -> list[Path]:
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(image_files)


def iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

#print(calculate_final_grade(94, 6))
import torch
print(torch.cuda.is_available())