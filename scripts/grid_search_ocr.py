#!/usr/bin/env python3
"""Grid search for PaddleOCR configurations on the PL license plate dataset."""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import CVATDatasetLoader  # noqa: E402
from src.utils import normalize_plate_text  # noqa: E402


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


def crop_with_padding(image: np.ndarray, bbox: Tuple[float, float, float, float], padding: float) -> np.ndarray:
    """Crop image around bbox with padding fraction."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    x1_p = max(0, x1 - pad_x)
    y1_p = max(0, y1 - pad_y)
    x2_p = min(w, x2 + pad_x)
    y2_p = min(h, y2 + pad_y)

    return image[y1_p:y2_p, x1_p:x2_p].copy()


def extract_best_text(result: list, min_score: float, top_k: int) -> str:
    """Extract best text candidate from PaddleOCR result (v3 API)."""
    if not result:
        return ""
    item = result[0]
    if isinstance(item, dict):
        texts = item.get("rec_texts", [])
        scores = item.get("rec_scores", [])
        if not texts:
            return ""
        pairs = []
        for t, s in zip(texts, scores):
            t = t.strip()
            if t and s >= min_score:
                pairs.append((t, s))
        if not pairs:
            pairs = [(t.strip(), 0) for t in texts if t.strip()]
        if not pairs:
            return ""
        # sort by score then length
        pairs.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        candidates = pairs[:top_k]
        # prefer plate-like length 5-10
        best_text = ""
        best_score = -1
        for text, score in candidates:
            clean = "".join(c for c in text if c.isalnum())
            if 5 <= len(clean) <= 10 and score > best_score:
                best_text = text
                best_score = score
        if not best_text:
            best_text = candidates[0][0]
        return best_text
    return ""


def run_config(
    samples: List[dict],
    ocr_version: str,
    padding: float,
    det_thresh: float,
    box_thresh: float,
    rec_score_thresh: float,
    min_score: float,
    top_k: int,
    target_width: int | None,
) -> dict:
    """Evaluate one OCR configuration."""
    paddle.device.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")

    ocr = PaddleOCR(
        lang="en",
        ocr_version=ocr_version,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_thresh=det_thresh,
        text_det_box_thresh=box_thresh,
        text_rec_score_thresh=rec_score_thresh,
    )

    correct = 0
    total = 0
    start = time.time()

    for sample in samples:
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            continue
        image, sx, sy = resize_keep_aspect(image, target_width=target_width)
        x1, y1, x2, y2 = sample["bbox"]
        bbox_resized = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        crop = crop_with_padding(image, bbox_resized, padding)
        result = ocr.predict(crop)
        pred_text = extract_best_text(result, min_score=min_score, top_k=top_k)

        gt = sample["plate_text"]
        if normalize_plate_text(pred_text) == normalize_plate_text(gt):
            correct += 1
        total += 1

    elapsed = time.time() - start
    acc = (correct / total * 100) if total else 0

    return {
        "ocr_version": ocr_version,
        "padding": padding,
        "det_thresh": det_thresh,
        "box_thresh": box_thresh,
        "rec_score_thresh": rec_score_thresh,
        "min_score": min_score,
        "top_k": top_k,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "time": elapsed,
        "ms_per_image": (elapsed / total * 1000) if total else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Grid search PaddleOCR configs on PL dataset")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Path to dataset (default: kagglehub cache)")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples (default: 100)")
    parser.add_argument(
        "--versions",
        type=str,
        nargs="*",
        default=["PP-OCRv4" ], #, "PP-OCRv5",],
        help="OCR versions",
    )
    parser.add_argument(
        "--paddings",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        help="Crop paddings",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        nargs="*",
        default=[0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
        help="Detection thresholds",
    )
    parser.add_argument(
        "--box-thresh",
        type=float,
        nargs="*",
        default=[0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
        help="Box thresholds",
    )
    parser.add_argument(
        "--rec-score",
        type=float,
        nargs="*",
        default=[0.0, 0.03, 0.05, 0.10],
        help="Recognition score thresholds",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        nargs="*",
        default=[0.03, 0.05, 0.08, 0.10, 0.12, 0.15],
        help="Min score for candidate filter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5],
        help="Top-K texts to consider",
    )
    parser.add_argument("--target-width", type=int, default=1600, help="Resize input width before crop (keeps aspect)")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    # Dataset path
    if args.dataset_path is None:
        import kagglehub
        args.dataset_path = Path(kagglehub.dataset_download("piotrstefaskiue/poland-vehicle-license-plate-dataset"))

    loader = CVATDatasetLoader(args.dataset_path)
    samples = loader.load_samples(limit=args.num_samples)

    if args.no_gpu:
        paddle.device.set_device("cpu")
    else:
        paddle.device.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")

    configs = []
    for v in args.versions:
        for p in args.paddings:
            for dt in args.det_thresh:
                for bt in args.box_thresh:
                    for rs in args.rec_score:
                        for ms in args.min_score:
                            for tk in args.top_k:
                                configs.append((v, p, dt, bt, rs, ms, tk))

    print(f"Testing {len(configs)} configurations on {len(samples)} samples...")
    results = []

    for cfg in configs:
        v, p, dt, bt, rs, ms, tk = cfg
        print(f"\nRunning: version={v}, padding={p}, det={dt}, box={bt}, rec={rs}, min={ms}, top_k={tk}")
        res = run_config(samples, v, p, dt, bt, rs, ms, tk, args.target_width)
        print(f"  -> Acc: {res['accuracy']:.2f}% ({res['correct']}/{res['total']}), time: {res['time']:.2f}s ({res['ms_per_image']:.1f} ms/img)")
        results.append(res)

    # Summary
    print("\n" + "=" * 80)
    print("PODSUMOWANIE GRID SEARCH")
    print("=" * 80)
    print(f"{'Model':<10} {'pad':>4} {'det':>5} {'box':>5} {'min':>5} {'topK':>5} {'acc%':>8} {'corr':>7} {'time(s)':>9} {'ms/img':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(
            f"{r['ocr_version']:<10} {r['padding']:>4.2f} {r['det_thresh']:>5.2f} {r['box_thresh']:>5.2f} "
            f"{r['min_score']:>5.2f} {r['top_k']:>5d} {r['accuracy']:>8.2f} {r['correct']:>4}/{r['total']:<3} "
            f"{r['time']:>9.2f} {r['ms_per_image']:>8.1f}"
        )

    best = max(results, key=lambda x: (x["accuracy"], -x["time"]))
    print("\n" + "=" * 80)
    print("NAJLEPSZA KONFIGURACJA")
    print("=" * 80)
    print(
        f"{best['ocr_version']} | pad={best['padding']} det={best['det_thresh']} box={best['box_thresh']} "
        f"min={best['min_score']} topK={best['top_k']} -> acc={best['accuracy']:.2f}% "
        f"time={best['time']:.2f}s ({best['ms_per_image']:.1f} ms/img)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
