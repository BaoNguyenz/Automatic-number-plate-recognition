"""
Benchmark script for PaddleOCR on benchmark_crops/.
Evaluates Exact Match % and Character Accuracy % on 26 ground-truth crops,
and outputs a comparative report against EasyOCR and Qwen2-VL.

Usage:
    python scripts/benchmark_paddleocr.py
"""

import sys
import time
import csv
from pathlib import Path
from typing import List, Dict, Any

# Ensure UTF-8 output on Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recognition.paddle_engine import PaddleOCREngine



def compute_char_acc(pred: str, gt: str) -> float:
    """Computes Levenshtein character-level similarity ratio."""
    import difflib
    matcher = difflib.SequenceMatcher(None, pred, gt)
    return matcher.ratio()


def run_benchmark():
    print("=" * 80)
    print("🔬 PADDLEOCR BENCHMARK EVALUATION (26 GROUND-TRUTH CROPS)")
    print("=" * 80)

    benchmark_dir = ROOT / "benchmark_crops"
    crop_files = sorted(list(benchmark_dir.glob("car_*_*.jpg")))

    if not crop_files:
        print(f"[-] Không tìm thấy ảnh benchmark nào tại: {benchmark_dir}")
        return

    print(f"[+] Tìm thấy {len(crop_files)} ảnh crop biển số benchmark.")

    # Initialize PaddleOCR Engine
    engine = PaddleOCREngine(lang="en", use_angle_cls=False, show_log=False)

    results: List[Dict[str, Any]] = []
    exact_matches = 0
    total_char_acc = 0.0
    latencies: List[float] = []

    print("\n" + "-" * 80)
    print(f"{'Ảnh Crop':<25} | {'Ground Truth':<12} | {'PaddleOCR Raw':<15} | {'Cleaned':<10} | {'Status':<6} | {'Time (ms)'}")
    print("-" * 80)

    for crop_path in crop_files:
        # File pattern: car_<id>_<PLATE>_<score>.jpg
        parts = crop_path.stem.split("_")
        gt_plate = parts[2].upper()

        t0 = time.time()
        rec_result = engine.recognize_plate(crop_path)
        latency_ms = (time.time() - t0) * 1000.0
        latencies.append(latency_ms)

        pred_plate = rec_result["plate_number"]
        raw_text = rec_result["raw_text"]
        conf = rec_result["confidence_score"]

        is_exact = (pred_plate == gt_plate)
        if is_exact:
            exact_matches += 1
            status = "✓ MATCH"
        else:
            status = "✗ DIFF"

        char_acc = compute_char_acc(pred_plate, gt_plate)
        total_char_acc += char_acc

        print(f"{crop_path.name[:25]:<25} | {gt_plate:<12} | {raw_text[:15]:<15} | {pred_plate:<10} | {status:<6} | {latency_ms:6.1f} ms")

        results.append({
            "filename": crop_path.name,
            "ground_truth": gt_plate,
            "raw_text": raw_text,
            "pred_plate": pred_plate,
            "confidence": conf,
            "latency_ms": round(latency_ms, 2),
            "exact_match": is_exact,
            "char_acc": round(char_acc, 4)
        })

    # Summary Metrics
    total = len(crop_files)
    exact_match_pct = (exact_matches / total) * 100.0
    avg_char_acc_pct = (total_char_acc / total) * 100.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    print("-" * 80)
    print("\n" + "=" * 80)
    print("📊 KẾT QUẢ ĐÁNH GIÁ PADDLEOCR TRÊN 26 CROPS:")
    print("=" * 80)
    print(f"  • Độ chính xác Khớp Tuyệt Đối (Exact Match):  {exact_match_pct:5.1f}% ({exact_matches}/{total})")
    print(f"  • Độ chính xác Cấp Ký Tự (Character Acc):    {avg_char_acc_pct:5.1f}%")
    print(f"  • Tốc độ suy luận trung bình:               {avg_latency:5.1f} ms/ảnh")
    print("=" * 80)

    # Save CSV
    csv_out = benchmark_dir / "paddleocr_benchmark.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "ground_truth", "raw_text", "pred_plate", "confidence", "latency_ms", "exact_match", "char_acc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[✓] Kết quả chi tiết đã được lưu tại: {csv_out}")

    # Comparative Summary Table
    print("\n" + "=" * 80)
    print("🏆 BẢNG TỔNG KẾT SO SÁNH 4 MÔ HÌNH NHẬN DIỆN:")
    print("=" * 80)
    print(f"{'Mô hình':<30} | {'Exact Match':<15} | {'Char Acc':<12} | {'Tốc độ (ms)'}")
    print("-" * 80)
    print(f"{'EasyOCR (Baseline cũ)':<30} | {'15.4% (4/26)':<15} | {'~60.0%':<12} | {'~25.0 ms'}")
    print(f"{'PaddleOCR (Mới thêm)':<30} | {f'{exact_match_pct:.1f}% ({exact_matches}/{total})':<15} | {f'{avg_char_acc_pct:.1f}%':<12} | {f'{avg_latency:.1f} ms'}")
    print(f"{'Qwen2-VL-2B (Zero-Shot)':<30} | {'57.7% (15/26)':<15} | {'90.7%':<12} | {'257.6 ms'}")
    print(f"{'Qwen2-VL-2B (Fine-Tuned QLoRA)':<30} | {'80.8% (21/26)':<15} | {'96.7%':<12} | {'261.2 ms'}")
    print("=" * 80)


if __name__ == "__main__":
    import numpy as np
    run_benchmark()
