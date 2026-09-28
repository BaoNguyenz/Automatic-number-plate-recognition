"""
Unit test script for Storage, Interpolator, and Visualizer (Group 4).
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import PlateStorageManager, TrajectoryInterpolator, ANPRVisualizer

def test_utils():
    print("=" * 80)
    print("🎨 UNIT TEST: NHÓM 4 - STORAGE (WEBP), INTERPOLATOR & VISUALIZER")
    print("=" * 80)

    # 1. Test PlateStorageManager (WebP)
    print("\n[+] 1. Kiểm thử PlateStorageManager (WebP):")
    storage = PlateStorageManager(base_dir="media/plates", webp_quality=85)
    dummy_crop = np.zeros((60, 180, 3), dtype=np.uint8)
    # Draw some text so it's not a blank image
    cv2.putText(dummy_crop, "SC56DYP", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    saved_path = storage.save_crop(dummy_crop, car_id=1701, plate_number="SC56DYP")
    print(f"   [✓] Đã lưu crop WebP tại: {saved_path}")
    assert Path(saved_path).exists(), "WebP file does not exist!"
    assert saved_path.endswith(".webp"), "File must have .webp extension!"

    loaded_crop = storage.load_crop(saved_path)
    assert loaded_crop is not None and loaded_crop.shape == dummy_crop.shape, "Loaded WebP crop mismatch!"
    print(f"   [✓] Đã load lại thành công crop WebP: Shape {loaded_crop.shape}, Dung lượng: {Path(saved_path).stat().st_size} bytes")

    # 2. Test TrajectoryInterpolator
    print("\n[+] 2. Kiểm thử TrajectoryInterpolator:")
    frames = [10, 15]
    bboxes = [[100.0, 100.0, 300.0, 300.0], [200.0, 200.0, 400.0, 400.0]]
    interpolated = TrajectoryInterpolator.interpolate_trajectory(frames, bboxes)
    print(f"   [✓] Bounding box tại Frame 10: {interpolated[10]}")
    print(f"   [✓] Bounding box nội suy Frame 12: {interpolated[12]}")
    print(f"   [✓] Bounding box tại Frame 15: {interpolated[15]}")
    assert len(interpolated) == 6, f"Expected 6 frames (10 to 15), got {len(interpolated)}"
    assert np.isclose(interpolated[12][0], 140.0), f"Expected x1=140.0, got {interpolated[12][0]}"

    # 3. Test ANPRVisualizer
    print("\n[+] 3. Kiểm thử ANPRVisualizer:")
    test_canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    viz = ANPRVisualizer()

    # Draw regular car
    viz.draw_vehicle_overlay(
        frame=test_canvas,
        car_id=3,
        car_bbox=(200, 300, 500, 600),
        plate_bbox=(300, 520, 420, 560),
        plate_number="NA13NRU",
        vehicle_type="car",
        is_watchlist=False
    )

    # Draw watchlist critical car (Stolen Vehicle)
    viz.draw_vehicle_overlay(
        frame=test_canvas,
        car_id=1701,
        car_bbox=(900, 300, 1300, 650),
        plate_bbox=(1050, 550, 1200, 600),
        plate_number="SC56DYP",
        vehicle_type="sedan",
        is_watchlist=True,
        alert_level="CRITICAL",
        watchlist_reason="Phương tiện bị báo mất cắp"
    )

    # Draw Top HUD
    viz.draw_top_hud(
        frame=test_canvas,
        active_alerts=[{"plate_number": "SC56DYP", "alert_level": "CRITICAL", "reason": "Báo mất cắp"}],
        fps=30.0,
        frame_idx=450
    )

    preview_output = Path("media/test_preview.jpg")
    cv2.imwrite(str(preview_output), test_canvas)
    print(f"   [✓] Đã xuất ảnh demo visualizer tại: {preview_output}")
    assert preview_output.exists() and preview_output.stat().st_size > 0

    print("\n" + "=" * 80)
    print("🎉 TẤT CẢ CÁC BÀI KIỂM THỬ NHÓM 4 ĐÃ VƯỢT QUA XUẤT SẮC!")
    print("=" * 80)

if __name__ == "__main__":
    test_utils()
