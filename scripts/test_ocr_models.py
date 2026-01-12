#!/usr/bin/env python3
"""Test different PaddleOCR models to find the best one for license plates."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import CVATDatasetLoader
from src.utils import normalize_plate_text

# Hyperparameters (defaults from best grid search run)
DEFAULT_PADDING = 0.20
DEFAULT_DET_THRESH = 0.05
DEFAULT_BOX_THRESH = 0.05
DEFAULT_REC_SCORE = 0.10
DEFAULT_MIN_SCORE = 0.10
DEFAULT_TOP_K = 4
DEFAULT_TARGET_WIDTH = 1600


# Resize helper
def resize_keep_aspect(image: np.ndarray, target_width: int | None = None):
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


# Available OCR versions in PaddleOCR
OCR_VERSIONS = [
    #"PP-OCRv5",
    "PP-OCRv4", 
]

# Detection models
DET_MODELS = [
    "PP-OCRv5_server_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv4_server_det",
    "PP-OCRv4_mobile_det",
]

# Recognition models  
REC_MODELS = [
    "en_PP-OCRv5_server_rec",
    "en_PP-OCRv5_mobile_rec",
    "en_PP-OCRv4_rec",
    "en_PP-OCRv3_rec",
]


def crop_with_padding(image: np.ndarray, bbox: tuple, padding: float = 0.12) -> np.ndarray:
    """Crop image with padding around bbox."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)
    
    x1_padded = max(0, x1 - pad_x)
    y1_padded = max(0, y1 - pad_y)
    x2_padded = min(w, x2 + pad_x)
    y2_padded = min(h, y2 + pad_y)
    
    return image[y1_padded:y2_padded, x1_padded:x2_padded].copy()


def extract_text_from_result(result: list, min_score: float = DEFAULT_MIN_SCORE, top_k: int = DEFAULT_TOP_K) -> tuple[str, list]:
    """Extract all texts and scores from PaddleOCR result."""
    if not result:
        return "", []
    
    item = result[0]
    if isinstance(item, dict):
        texts = item.get("rec_texts", [])
        scores = item.get("rec_scores", [])
        
        # Return best text and all text-score pairs
        all_results = list(zip(texts, scores))

        # Filter by min_score
        filtered = [(t, s) for t, s in all_results if t.strip() and s >= min_score]
        if not filtered:
            filtered = [(t, s) for t, s in all_results if t.strip()]
        if not filtered:
            return "", all_results

        # Sort and take top_k
        filtered.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        candidates = filtered[:top_k]

        # Prefer plate-like lengths
        best_text = ""
        best_score = -1
        for text, score in candidates:
            clean = ''.join(c for c in text if c.isalnum())
            if 5 <= len(clean) <= 10 and score > best_score:
                best_text = text
                best_score = score

        if not best_text:
            best_text = candidates[0][0]
        
        return best_text, all_results
    
    return "", []


def test_single_image(
    image_path: Path,
    bbox: tuple,
    ground_truth: str,
    padding: float = DEFAULT_PADDING,
    ocr_version: str = "PP-OCRv4",
    target_width: int | None = DEFAULT_TARGET_WIDTH,
    det_thresh: float = DEFAULT_DET_THRESH,
    box_thresh: float = DEFAULT_BOX_THRESH,
    rec_score: float = DEFAULT_REC_SCORE,
    min_score: float = DEFAULT_MIN_SCORE,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Test OCR on a single image with different settings."""
    
    image = cv2.imread(str(image_path))
    if image is None:
        return {"error": f"Could not read {image_path}"}

    image, sx, sy = resize_keep_aspect(image, target_width=target_width)
    x1, y1, x2, y2 = bbox
    bbox_resized = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
    crop = crop_with_padding(image, bbox_resized, padding)
    
    # Initialize OCR
    ocr = PaddleOCR(
        lang="en",
        ocr_version=ocr_version,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_thresh=det_thresh,
        text_det_box_thresh=box_thresh,
        text_rec_score_thresh=rec_score,
    )
    
    # Run OCR
    result = ocr.predict(crop)
    best_text, all_results = extract_text_from_result(result, min_score=min_score, top_k=top_k)
    
    gt_norm = normalize_plate_text(ground_truth)
    pred_norm = normalize_plate_text(best_text)
    
    return {
        "image": image_path.name,
        "ground_truth": ground_truth,
        "prediction": best_text,
        "gt_normalized": gt_norm,
        "pred_normalized": pred_norm,
        "correct": gt_norm == pred_norm,
        "all_texts": all_results,
        "ocr_version": ocr_version,
    }


def test_ocr_versions(dataset_path: Path, num_samples: int = 20, target_width: int | None = DEFAULT_TARGET_WIDTH):
    """Compare different OCR versions."""
    
    print("=" * 70)
    print("TESTING DIFFERENT PADDLEOCR VERSIONS")
    print("=" * 70)
    
    # Set GPU
    paddle.device.set_device("gpu")
    
    # Load dataset
    loader = CVATDatasetLoader(dataset_path)
    samples = loader.load_samples(limit=num_samples)
    
    print(f"\nLoaded {len(samples)} samples")
    
    results_by_version = {}
    
    for version in OCR_VERSIONS:
        print(f"\n{'='*50}")
        print(f"Testing {version}")
        print("=" * 50)
        
        correct = 0
        total = 0
        errors = []
        
        start_time = time.time()
        
        for sample in samples:
            result = test_single_image(
                sample["image_path"],
                sample["bbox"],
                sample["plate_text"],
                padding=DEFAULT_PADDING,
                ocr_version=version,
                target_width=target_width,
                det_thresh=DEFAULT_DET_THRESH,
                box_thresh=DEFAULT_BOX_THRESH,
                rec_score=DEFAULT_REC_SCORE,
                min_score=DEFAULT_MIN_SCORE,
                top_k=DEFAULT_TOP_K,
            )
            
            if "error" in result:
                continue
            
            total += 1
            if result["correct"]:
                correct += 1
            else:
                errors.append(result)
        
        elapsed = time.time() - start_time
        accuracy = (correct / total * 100) if total > 0 else 0
        
        results_by_version[version] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "time": elapsed,
            "errors": errors[:5],  # Keep first 5 errors
        }
        
        print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
        print(f"Time: {elapsed:.2f}s ({elapsed/total*1000:.1f}ms/image)")
        
        # Show some errors
        if errors:
            print("\nSample errors:")
            for err in errors[:3]:
                print(f"  {err['image']}: GT='{err['ground_truth']}' Pred='{err['prediction']}'")
                print(f"    All detected: {err['all_texts'][:5]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PODSUMOWANIE - PORÓWNANIE MODELI PADDLEOCR")
    print("=" * 70)
    
    # Sort by accuracy
    sorted_results = sorted(results_by_version.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    print(f"\n{'Model':<20} {'Accuracy':<15} {'Correct':<12} {'Time':<15} {'ms/img':<10}")
    print("-" * 70)
    
    for version, data in sorted_results:
        ms_per_img = data['time'] / data['total'] * 1000 if data['total'] > 0 else 0
        print(f"{version:<20} {data['accuracy']:>6.1f}%        {data['correct']:>3}/{data['total']:<6}   {data['time']:>6.2f}s        {ms_per_img:>6.1f}ms")
    
    # Best model
    best_version = sorted_results[0][0]
    best_data = sorted_results[0][1]
    
    print("\n" + "=" * 70)
    print(f"NAJLEPSZY MODEL: {best_version}")
    print(f"  Accuracy: {best_data['accuracy']:.1f}%")
    print(f"  Czas: {best_data['time']:.2f}s ({best_data['time']/best_data['total']*1000:.1f}ms/obraz)")
    print("=" * 70)
    
    # Recommendation
    print("\nREKOMENDACJA:")
    if best_data['accuracy'] >= 60:
        print(f"  Użyj modelu {best_version} - spełnia wymaganie 60% accuracy")
    else:
        print(f"  Żaden model nie osiągnął 60% accuracy.")
        print("  Rozważ:")
        print("    - Zmniejszenie paddingu (crop_padding)")
        print("    - Użycie tylko rozpoznawania (bez detekcji) na wyciętych tablicach")
        print("    - Preprocessing obrazu (binaryzacja, kontrast)")
    
    return results_by_version


def test_single_file_detailed(image_path: Path, bbox: tuple, ground_truth: str, target_width: int | None = DEFAULT_TARGET_WIDTH):
    """Test a single file with all versions and show detailed results."""
    
    print(f"\n{'='*70}")
    print(f"DETAILED TEST: {image_path.name}")
    print(f"Ground truth: {ground_truth}")
    print("=" * 70)
    
    paddle.device.set_device("gpu")
    
    image = cv2.imread(str(image_path))
    if image is None:
        print("Error: Could not read image")
        return
    image, sx, sy = resize_keep_aspect(image, target_width=target_width)
    x1, y1, x2, y2 = bbox
    bbox_resized = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
    
    # Test with different paddings
    paddings = [DEFAULT_PADDING]
    
    for version in OCR_VERSIONS:
        print(f"\n--- {version} ---")
        
        for padding in paddings:
            crop = crop_with_padding(image, bbox_resized, padding)
            
            ocr = PaddleOCR(
                lang="en",
                ocr_version=version,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_thresh=DEFAULT_DET_THRESH,
                text_det_box_thresh=DEFAULT_BOX_THRESH,
                text_rec_score_thresh=DEFAULT_REC_SCORE,
            )
            
            result = ocr.predict(crop)
            best_text, all_results = extract_text_from_result(result, min_score=DEFAULT_MIN_SCORE, top_k=DEFAULT_TOP_K)
            
            gt_norm = normalize_plate_text(ground_truth)
            pred_norm = normalize_plate_text(best_text)
            status = "✓" if gt_norm == pred_norm else "✗"
            
            print(f"  padding={padding:.2f}: {status} '{best_text}' (GT: '{ground_truth}')")
            if all_results:
                for text, score in all_results[:5]:
                    print(f"      - '{text}' (score: {score:.3f})")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PaddleOCR models")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--single-image", type=Path, default=None, help="Test single image")
    parser.add_argument("--bbox", type=str, default=None, help="Bbox as x1,y1,x2,y2")
    parser.add_argument("--gt", type=str, default=None, help="Ground truth text")
    parser.add_argument("--target-width", type=int, default=DEFAULT_TARGET_WIDTH, help="Resize input width before crop (keeps aspect)")
    
    args = parser.parse_args()
    
    # Find dataset
    if args.dataset_path is None:
        import kagglehub
        args.dataset_path = Path(kagglehub.dataset_download(
            "piotrstefaskiue/poland-vehicle-license-plate-dataset"
        ))
    
    print(f"Dataset: {args.dataset_path}")
    
    if args.single_image:
        # Test single image
        if args.bbox:
            bbox = tuple(map(float, args.bbox.split(",")))
        else:
            # Try to get from dataset annotations
            loader = CVATDatasetLoader(args.dataset_path)
            image_name = args.single_image.name
            if image_name in loader.annotations:
                ann = loader.annotations[image_name]
                bbox = ann.bbox
                gt = args.gt or ann.plate_text
            else:
                print(f"Error: Could not find annotation for {image_name}")
                return 1
        
        gt = args.gt or "UNKNOWN"
        test_single_file_detailed(args.single_image, bbox, gt, target_width=args.target_width)
    else:
        # Test all versions
        test_ocr_versions(args.dataset_path, args.num_samples, target_width=args.target_width)
    
    return 0


if __name__ == "__main__":
    exit(main())
