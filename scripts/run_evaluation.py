#!/usr/bin/env python3
"""Run evaluation on the license plate dataset."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import Evaluator


def find_dataset_path() -> Path | None:
    """Try to find the dataset path."""
    try:
        import kagglehub
        
        # This returns the cached path if already downloaded
        path = kagglehub.dataset_download(
            "piotrstefaskiue/poland-vehicle-license-plate-dataset"
        )
        return Path(path)
    except Exception:
        pass
    
    # Check local data directory
    local_path = Path(__file__).parent.parent / "data"
    if local_path.exists():
        return local_path
    
    return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate license plate recognition system"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to dataset (default: auto-detect)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=1600,
        help="Resize input width before processing (keeps aspect)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--test-split",
        action="store_true",
        help="Use only test split (30%% of data)"
    )
    parser.add_argument(
        "--use-yolo",
        action="store_true",
        help="Use YOLO detection instead of GT boxes"
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default 0.25)"
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=None,
        help="Path to YOLO plate detector model (.pt)"
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=None,
        help="Optional YOLO inference size (imgsz)"
    )
    parser.add_argument("--padding", type=float, default=0.20, help="Padding fraction around bbox for OCR (default 0.20)")
    parser.add_argument("--det-thresh", type=float, default=0.05, help="OCR det threshold")
    parser.add_argument("--box-thresh", type=float, default=0.05, help="OCR box threshold")
    parser.add_argument("--rec-score", type=float, default=0.10, help="OCR recognition score threshold")
    parser.add_argument("--min-score", type=float, default=0.10, help="OCR min score filter")
    parser.add_argument("--top-k", type=int, default=4, help="OCR top-K candidates")
    parser.add_argument("--ocr-version", type=str, default="PP-OCRv4", help="PaddleOCR version (default PP-OCRv4)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for YOLO inference (default 1)")
    parser.add_argument("--ocr-batch-size", type=int, default=1, help="Batch size for OCR (default 1 - per crop)")
    parser.add_argument("--polish-fix", action="store_true", help="Apply heuristic corrections for Polish plates")
    
    args = parser.parse_args()
    
    # Find dataset path
    dataset_path = args.dataset_path
    if dataset_path is None:
        print("Looking for dataset...")
        dataset_path = find_dataset_path()
        if dataset_path is None:
            print("Error: Dataset not found. Run download_data.py first.")
            return 1
    
    print(f"Using dataset: {dataset_path}")
    
    # Check if dataset structure is valid
    photos_dir = dataset_path / "photos"
    annotations_file = dataset_path / "annotations.xml"
    
    if not photos_dir.exists() or not annotations_file.exists():
        print(f"Error: Invalid dataset structure at {dataset_path}")
        print("Expected 'photos' directory and 'annotations.xml' file.")
        return 1
    
    # Initialize evaluator
    print(f"\nInitializing evaluator (GPU: {not args.no_gpu})...")
    evaluator = Evaluator(
        dataset_path=dataset_path,
        use_gpu=not args.no_gpu,
        target_width=args.target_width,
        use_yolo=args.use_yolo,
        yolo_conf=args.yolo_conf,
        yolo_model=args.yolo_model,
        yolo_imgsz=args.yolo_imgsz,
        padding=args.padding,
        det_thresh=args.det_thresh,
        box_thresh=args.box_thresh,
        rec_score=args.rec_score,
        min_score=args.min_score,
        top_k=args.top_k,
        ocr_version=args.ocr_version,
        batch_size=args.batch_size,
        ocr_batch_size=args.ocr_batch_size,
        correct_polish=args.polish_fix,
    )
    
    # Run evaluation
    print(f"\nRunning evaluation on {args.num_samples} samples...\n")
    result = evaluator.evaluate(
        num_samples=args.num_samples,
        verbose=not args.quiet,
        use_test_split=args.test_split,
        use_yolo=args.use_yolo,
    )
    
    # Print results
    print(result)
    
    # Show some example predictions
    print("\nSample predictions (first 10):")
    print("-" * 60)
    for pred in result.predictions[:10]:
        status = "✓" if pred["correct"] else "✗"
        print(f"{status} {pred['image']}")
        print(f"   GT:   {pred['ground_truth']} -> {pred['normalized_gt']}")
        print(f"   Pred: {pred['prediction']} -> {pred['normalized_pred']}")
    
    # Save results if requested
    if args.output:
        evaluator.save_results(result, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
