"""
Master ANPR Pipeline Execution Script.
Integrates TensorRT Vehicle/Plate Detectors + ByteTrack + Fine-Tuned Qwen2-VL-2B (QLoRA)
+ SQLite Database + WebP Storage + Trajectory Interpolator + Video Rendering.

Usage:
    python scripts/run_pipeline.py [--max-frames 300] [--output out.mp4]
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import yaml
import numpy as np

# Set UTF-8 encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.connection import DatabaseManager
from src.vision import VehiclePlateDetector, BestFrameTracker, TrackedVehicle
from src.recognition import Qwen2VLEngine, PlatePostProcessor
from src.utils import PlateStorageManager, TrajectoryInterpolator, ANPRVisualizer


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    video_path: Optional[str] = None,
    output_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    skip_render: bool = False
):
    print("=" * 85)
    print("🚦 AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) - MASTER PIPELINE")
    print("   Architecture: TensorRT (YOLOv8) -> ByteTrack -> Qwen2-VL-2B QLoRA -> SQLite -> WebP")
    print("=" * 85)

    config = load_config(ROOT / "config" / "config.yaml")

    input_video = ROOT / (video_path or config["paths"]["input_video"])
    output_video = ROOT / (output_path or config["paths"]["output_video"])

    if not input_video.exists():
        print(f"[-] Error: Input video file not found at '{input_video}'")
        return

    # 1. Initialize Database
    print("\n[+] [1/6] Khởi tạo Database Layer (SQLite via SQLAlchemy ORM)...")
    db = DatabaseManager(db_url=config["database"]["url"])

    # 2. Initialize Detectors (TensorRT FP16)
    print("\n[+] [2/6] Khởi tạo Lõi Thị Giác TensorRT...")
    detector = VehiclePlateDetector(
        vehicle_model_path=str(ROOT / config["paths"]["coco_engine"]),
        vehicle_pt_path=str(ROOT / config["paths"]["coco_model"]),
        plate_model_path=str(ROOT / config["paths"]["plate_engine"]),
        plate_pt_path=str(ROOT / config["paths"]["plate_model"]),
        vehicle_conf=config["vision"]["vehicle_conf"],
        plate_conf=config["vision"]["plate_conf"],
        vehicle_classes=config["vision"]["vehicle_classes"]
    )

    # 3. Initialize Tracker with Best-Frame Selector
    print("\n[+] [3/6] Khởi tạo ByteTrack & Best-Frame Selector...")
    tracker = BestFrameTracker(
        detector=detector,
        max_missed_frames=12,
        tracker_type=config["vision"]["tracker"]
    )

    # 4. Initialize Qwen2-VL Engine with LoRA
    print("\n[+] [4/6] Khởi tạo Lõi Nhận Diện Qwen2-VL-2B (QLoRA Adapter)...")
    lora_dir = ROOT / config["paths"]["vlm_lora_dir"]
    vlm_engine = Qwen2VLEngine(
        model_id=config["paths"]["vlm_model_id"],
        processor_id="Qwen/Qwen2-VL-2B-Instruct",
        lora_dir=str(lora_dir) if lora_dir.exists() else None,
        prompt=config["vlm"]["prompt"],
        max_new_tokens=config["vlm"]["max_new_tokens"]
    )

    # 5. Initialize Storage & Visualizer
    storage = PlateStorageManager(base_dir=str(ROOT / config["paths"]["media_dir"]))
    visualizer = ANPRVisualizer()

    # Open Video
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    limit_frames = min(max_frames, total_video_frames) if max_frames else total_video_frames
    print(f"\n[+] Video: '{input_video.name}' ({width}x{height}, {fps:.1f} FPS, {limit_frames}/{total_video_frames} frames)")

    # ---------------------------------------------------------
    # PHASE 1: Detection, Tracking & Asynchronous Recognition
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("⏩ GIAI ĐOẠN 1: THEO DÕI XE & TRÍCH XUẤT BIỂN SỐ VLM")
    print("=" * 85)

    raw_frame_results: Dict[int, Dict[int, Dict[str, Any]]] = {}
    vehicle_records: Dict[int, Dict[str, Any]] = {}
    frame_idx = 0
    start_time = time.time()

    def process_and_record_vehicle(v: TrackedVehicle):
        if v.car_id in vehicle_records:
            return
        if v.best_crop is None:
            return

        # 1. VLM Recognition
        rec_res = vlm_engine.recognize_plate(v.best_crop, vehicle_type=v.vehicle_type)
        plate_str = rec_res["plate_number"]
        v.plate_text = plate_str

        # 2. Save WebP crop to disk
        crop_path = storage.save_crop(v.best_crop, car_id=v.car_id, plate_number=plate_str)

        # 3. Check Watchlist & Save to Database
        det_record = db.add_detection(
            car_id=v.car_id,
            plate_number=plate_str,
            frame_number=v.best_crop_frame,
            raw_vlm_text=rec_res["raw_text"],
            confidence_score=rec_res["confidence_score"],
            vehicle_type=v.vehicle_type,
            plate_crop_path=crop_path
        )

        vehicle_records[v.car_id] = {
            "plate_number": plate_str,
            "vehicle_type": v.vehicle_type,
            "is_watchlist": det_record.is_watchlist_match,
            "alert_level": "CRITICAL" if "mất cắp" in str(det_record.watchlist_reason) else "WARNING",
            "watchlist_reason": det_record.watchlist_reason,
            "best_crop_frame": v.best_crop_frame,
            "crop_path": crop_path
        }

    while cap.isOpened() and frame_idx < limit_frames:
        ret, frame = cap.read()
        if not ret:
            break

        raw_frame_results[frame_idx] = {}

        # Tracking + Plate detection
        active_vehicles, finished = tracker.track_and_associate(frame, frame_idx)

        # Record active boxes for this frame
        for v in active_vehicles:
            raw_frame_results[frame_idx][v.car_id] = {
                "car_bbox": list(v.vehicle_bbox),
                "plate_bbox": list(v.best_plate_bbox) if v.best_plate_bbox else None
            }

        # Recognize finished departed vehicles
        for v in finished:
            process_and_record_vehicle(v)

        frame_idx += 1
        if frame_idx % 60 == 0 or frame_idx == limit_frames:
            elapsed = time.time() - start_time
            cur_fps = frame_idx / max(0.001, elapsed)
            print(f"  Frame {frame_idx:04d}/{limit_frames:04d} ({frame_idx/limit_frames*100:.1f}%) | Speed: {cur_fps:.1f} FPS | Active Tracks: {len(active_vehicles)} | Recognized: {len(vehicle_records)}")

    # Finalize remaining cars still on screen at end of video
    print("\n[+] Hoàn tất các xe còn lại trong khung hình cuối...")
    remaining_tracks = tracker.finalize_remaining_tracks()
    for v in remaining_tracks:
        process_and_record_vehicle(v)

    # Attach recognized metadata to frame results
    for f_idx in raw_frame_results:
        for c_id in raw_frame_results[f_idx]:
            if c_id in vehicle_records:
                raw_frame_results[f_idx][c_id].update(vehicle_records[c_id])

    p1_time = time.time() - start_time
    print(f"\n[✓] Giai đoạn 1 hoàn thành trong {p1_time:.1f}s ({frame_idx / p1_time:.1f} FPS trung bình)")
    print(f"[✓] Tổng số phương tiện được theo dõi: {len(tracker.tracks)}")
    print(f"[✓] Tổng số biển số đã nhận diện qua Qwen2-VL: {len(vehicle_records)}")

    # Interpolate trajectories to eliminate missing frames and flickering
    print("\n[+] [5/6] Đang chạy Trajectory Interpolator (Nội suy chuyển động mượt mà)...")
    smooth_results = TrajectoryInterpolator.interpolate_frame_results(raw_frame_results, max_gap=25)

    cap.release()

    # ---------------------------------------------------------
    # PHASE 2: Video Rendering with Modern UI Overlays
    # ---------------------------------------------------------
    if skip_render:
        print("\n[i] Bỏ qua bước Render Video (--skip-render).")
        return

    print("\n" + "=" * 85)
    print(f"🎬 GIAI ĐOẠN 2: RENDER VIDEO THÀNH PHẨM ({output_video.name})")
    print("=" * 85)

    cap = cv2.VideoCapture(str(input_video))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    render_start = time.time()
    r_frame_idx = 0

    while cap.isOpened() and r_frame_idx < limit_frames:
        ret, frame = cap.read()
        if not ret:
            break

        active_alerts_this_frame = []

        if r_frame_idx in smooth_results:
            cars_in_frame = smooth_results[r_frame_idx]

            for car_id, data in cars_in_frame.items():
                c_bbox = data.get("car_bbox")
                p_bbox = data.get("plate_bbox")
                plate_num = data.get("plate_number")
                v_type = data.get("vehicle_type", "car")
                is_watch = data.get("is_watchlist", False)
                a_level = data.get("alert_level", "NORMAL")
                reason = data.get("watchlist_reason")

                if is_watch:
                    active_alerts_this_frame.append({
                        "plate_number": plate_num,
                        "alert_level": a_level,
                        "reason": reason
                    })

                # Draw sophisticated overlay
                visualizer.draw_vehicle_overlay(
                    frame=frame,
                    car_id=car_id,
                    car_bbox=c_bbox,
                    plate_bbox=p_bbox,
                    plate_number=plate_num,
                    vehicle_type=v_type,
                    is_watchlist=is_watch,
                    alert_level=a_level,
                    watchlist_reason=reason
                )

        # Draw Top Command-Center HUD
        visualizer.draw_top_hud(
            frame=frame,
            active_alerts=active_alerts_this_frame,
            fps=fps,
            frame_idx=r_frame_idx
        )

        out.write(frame)
        r_frame_idx += 1

        if r_frame_idx % 60 == 0 or r_frame_idx == limit_frames:
            r_elapsed = time.time() - render_start
            print(f"  Rendering Frame {r_frame_idx:04d}/{limit_frames:04d} ({r_frame_idx/limit_frames*100:.1f}%) | Speed: {r_frame_idx/max(0.001, r_elapsed):.1f} FPS")

    cap.release()
    out.release()

    r_time = time.time() - render_start
    total_time = p1_time + r_time

    print("\n" + "=" * 85)
    print("🎉 HOÀN THÀNH TOÀN BỘ PIPELINE ANPR!")
    print("=" * 85)
    print(f"  • Video đầu ra:        {output_video} (Dung lượng: {output_video.stat().st_size / (1024**2):.1f} MB)")
    print(f"  • Cơ sở dữ liệu:       data/anpr.db (Xem qua: python scripts/view_db.py)")
    print(f"  • Kho ảnh WebP:        media/plates/")
    print(f"  • Tổng thời gian chạy: {total_time:.1f}s ({total_time/60:.1f} phút)")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full ANPR Pipeline")
    parser.add_argument("--video", type=str, default=None, help="Path to input video")
    parser.add_argument("--output", type=str, default="out.mp4", help="Path to output video")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--skip-render", action="store_true", help="Skip video rendering")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames,
        skip_render=args.skip_render
    )
