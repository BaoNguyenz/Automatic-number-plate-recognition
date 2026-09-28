"""
Dataset Preparation Script for Fine-tuning Qwen2-VL-2B on License Plates.
Fast sequential frame processor that extracts plate crops and builds
conversational JSON annotations for QLoRA training.
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Any
import cv2
import numpy as np
import pandas as pd

# UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIDEO_PATH = ROOT / "2103099-uhd_3840_2160_30fps.mp4"
CSV_PATH = ROOT / "data" / "test_interpolated.csv"
OUTPUT_DIR = ROOT / "data" / "finetune_dataset"
IMAGES_DIR = OUTPUT_DIR / "images"
BENCHMARK_DIR = ROOT / "benchmark_crops"

PROMPT = (
    "Read the vehicle license plate number shown in this cropped image. "
    "Output ONLY the alphanumeric plate characters with no extra spaces, punctuation, or explanation."
)

def augment_crop(crop: np.ndarray) -> np.ndarray:
    """Applies realistic augmentations (brightness, contrast, slight blur)."""
    aug = crop.copy()
    op = random.choice(["bright", "dark", "contrast", "blur"])
    if op == "bright":
        val = random.randint(15, 35)
        aug = cv2.add(aug, np.array([val, val, val], dtype=np.uint8))
    elif op == "dark":
        val = random.randint(15, 30)
        aug = cv2.subtract(aug, np.array([val, val, val], dtype=np.uint8))
    elif op == "contrast":
        alpha = random.uniform(1.1, 1.3)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=0)
    elif op == "blur":
        aug = cv2.GaussianBlur(aug, (3, 3), 0)
    return aug

def prepare_dataset(max_video_frames: int = 500):
    print("=" * 80, flush=True)
    print("🛠️ PREPARING FINETUNE DATASET FOR QWEN2-VL-2B (QLoRA)", flush=True)
    print("=" * 80, flush=True)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    sample_idx = 0

    # 1. Incorporate verified benchmark crops first
    benchmark_crops = list(BENCHMARK_DIR.glob("*.jpg"))
    print(f"[+] Incorporating {len(benchmark_crops)} verified benchmark crops...", flush=True)
    for b_crop in benchmark_crops:
        parts = b_crop.stem.split("_")
        if len(parts) >= 3:
            plate_text = parts[2].upper()
            b_name = f"bench_{sample_idx:04d}_{plate_text}.jpg"
            b_dest = IMAGES_DIR / b_name
            import shutil
            shutil.copy(str(b_crop), str(b_dest))

            samples.append({
                "id": f"sample_{sample_idx}",
                "image": f"images/{b_name}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{PROMPT}"},
                    {"from": "gpt", "value": plate_text}
                ]
            })
            sample_idx += 1

            # Add 2 augmented variations of each benchmark crop
            for aug_i in range(2):
                img_cv = cv2.imread(str(b_crop))
                if img_cv is not None:
                    aug_cv = augment_crop(img_cv)
                    aug_name = f"bench_{sample_idx:04d}_{plate_text}_aug{aug_i}.jpg"
                    cv2.imwrite(str(IMAGES_DIR / aug_name), aug_cv)
                    samples.append({
                        "id": f"sample_{sample_idx}",
                        "image": f"images/{aug_name}",
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{PROMPT}"},
                            {"from": "gpt", "value": plate_text}
                        ]
                    })
                    sample_idx += 1

    # 2. Extract crops sequentially from video if available
    if VIDEO_PATH.exists() and CSV_PATH.exists():
        print(f"[+] Loading {CSV_PATH} and scanning video frames sequentially...", flush=True)
        df = pd.read_csv(CSV_PATH)
        valid_df = df[(df['license_number'] != '0') & (df['license_number_score'] > 0.2)].copy()

        # Build lookup table: frame_nmr -> list of rows
        frame_lookup = {}
        for _, row in valid_df.iterrows():
            f_num = int(row['frame_nmr'])
            if f_num not in frame_lookup:
                frame_lookup[f_num] = []
            frame_lookup[f_num].append(row)

        cap = cv2.VideoCapture(str(VIDEO_PATH))
        frame_counter = 0

        # Read sequentially to avoid slow random seeks
        while cap.isOpened() and frame_counter < max_video_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_counter in frame_lookup and frame_counter % 5 == 0:
                H, W = frame.shape[:2]
                for r in frame_lookup[frame_counter]:
                    plate_text = str(r['license_number']).strip().upper()
                    if len(plate_text) < 4 or len(plate_text) > 10:
                        continue
                    try:
                        raw_bbox = str(r['license_plate_bbox']).strip()
                        coords = [float(v) for v in raw_bbox.replace('[', '').replace(']', '').split()]
                        if len(coords) != 4:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in coords]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W, x2), min(H, y2)

                        if (x2 - x1) >= 30 and (y2 - y1) >= 15:
                            crop = frame[y1:y2, x1:x2]
                            img_name = f"crop_{sample_idx:04d}_{plate_text}.jpg"
                            cv2.imwrite(str(IMAGES_DIR / img_name), crop)

                            samples.append({
                                "id": f"sample_{sample_idx}",
                                "image": f"images/{img_name}",
                                "conversations": [
                                    {"from": "human", "value": f"<image>\n{PROMPT}"},
                                    {"from": "gpt", "value": plate_text}
                                ]
                            })
                            sample_idx += 1
                    except Exception:
                        pass

            frame_counter += 1

        cap.release()
        print(f"[+] Scanned {frame_counter} video frames.", flush=True)

    random.seed(42)
    random.shuffle(samples)

    split_idx = int(len(samples) * 0.85)
    train_data = samples[:split_idx]
    val_data = samples[split_idx:]

    train_json = OUTPUT_DIR / "train.json"
    val_json = OUTPUT_DIR / "val.json"

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Dataset creation complete!", flush=True)
    print(f"[✓] Total plate crops: {len(samples)}", flush=True)
    print(f"[✓] Training samples saved to: {train_json} ({len(train_data)} samples)", flush=True)
    print(f"[✓] Validation samples saved to: {val_json} ({len(val_data)} samples)", flush=True)

if __name__ == "__main__":
    prepare_dataset(max_video_frames=400)
