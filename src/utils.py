"""Utility functions for license plate recognition."""

import re
from pathlib import Path


def normalize_plate_text(text: str) -> str:
    """
    Normalize license plate text for comparison.
    
    - Removes all whitespace
    - Converts to uppercase
    - Removes special characters except alphanumeric
    """
    # Remove all whitespace
    text = re.sub(r"\s+", "", text)
    # Keep only alphanumeric characters
    text = re.sub(r"[^A-Za-z0-9]", "", text)
    # Convert to uppercase
    return text.upper()


def calculate_accuracy(predictions: list[str], ground_truths: list[str]) -> float:
    """
    Calculate accuracy as the ratio of correctly recognized plates.
    
    Args:
        predictions: List of predicted plate texts
        ground_truths: List of ground truth plate texts
        
    Returns:
        Accuracy as percentage (0-100)
    """
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
    """
    Calculates the final grade based on license plate OCR accuracy and processing time.
    
    Parameters:
        accuracy_percent: OCR accuracy as a percentage (0–100)
        processing_time_sec: total time to process 100 images in seconds
        
    Returns:
        Grade on a scale from 2.0 to 5.0 (rounded to the nearest 0.5)
    """
    # Check minimum requirements
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    
    # Normalize accuracy: 60% → 0.0, 100% → 1.0
    accuracy_norm = (accuracy_percent - 60) / 40
    
    # Normalize time: 60s → 0.0, 10s → 1.0
    time_norm = (60 - processing_time_sec) / 50
    
    # Compute weighted score
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    
    grade = 2.0 + 3.0 * score
    
    # Round to the nearest 0.5
    return round(grade * 2) / 2


def get_image_files(directory: Path, extensions: tuple = (".jpg", ".jpeg", ".png")) -> list[Path]:
    """Get all image files from a directory."""
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(image_files)

#print(calculate_final_grade(93, 4.2))