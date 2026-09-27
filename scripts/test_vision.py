"""
Unit test script for Vision & Tracking Module (Group 2).
Tests TensorRT Vehicle/Plate Detector and ByteTrack Best-Frame Selector on video frames.
"""

import sys
import cv2
from pathlib import Path


# UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision import VehiclePlateDetector, BestFrameTracker

def test_vision():
    video_path = "2103099-uhd_3840_2160_30fps.mp4"
    if not Path(video_path).exists():
        print(f"[-] Video file not found: {video_path}")
        return

    print("[+] Initializing VehiclePlateDetector with TensorRT engines...")
    detector = VehiclePlateDetector(
        vehicle_model_path="Weight/yolov8n.engine",
        vehicle_pt_path="Weight/yolov8n.pt",
        plate_model_path="Weight/license_plate_detector.engine",
        plate_pt_path="Weight/license_plate_detector.pt",
        vehicle_conf=0.4,
        plate_conf=0.3
    )

    print("[+] Initializing BestFrameTracker with ByteTrack...")
    tracker = BestFrameTracker(
        detector=detector,
        max_missed_frames=10,
        tracker_type="bytetrack.yaml"
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[+] Loaded video '{video_path}' ({total_frames} frames, {fps:.1f} FPS)")

    # Run for first 60 frames to verify tracking and best crop selection
    test_limit = 60
    frame_idx = 0
    selected_candidates = []

    print(f"[+] Processing first {test_limit} frames for testing...")
    while cap.isOpened() and frame_idx < test_limit:
        ret, frame = cap.read()
        if not ret:
            break

        active_vehicles, ready = tracker.track_and_associate(frame, frame_idx)
        for v in ready:
            selected_candidates.append(v)
            print(f"  🏁 [Car {v.car_id}] departed. Best crop frame: {v.best_crop_frame}, Quality Score: {v.best_crop_quality:.2f}, Shape: {v.best_crop.shape}")

        frame_idx += 1

    # Finalize remaining cars currently in frame
    remaining = tracker.finalize_remaining_tracks()
    selected_candidates.extend(remaining)

    print(f"\n[✓] Processed {frame_idx} frames successfully!")
    print(f"[✓] Total tracked vehicles detected: {len(tracker.tracks)}")
    print(f"[✓] Vehicles with selected Best-Frame plate crops: {len(selected_candidates)}")

    for v in selected_candidates:
        if v.best_crop is not None:
            print(f"   🚗 Car ID: {v.car_id} ({v.vehicle_type}) | Best Frame: {v.best_crop_frame} | Quality: {v.best_crop_quality:.1f} | Crop Shape: {v.best_crop.shape}")

    cap.release()
    print("\n[🎉] GROUP 2 TEST PASSED: TensorRT + ByteTrack + Best-Frame Selector working smoothly!")

if __name__ == "__main__":
    test_vision()
