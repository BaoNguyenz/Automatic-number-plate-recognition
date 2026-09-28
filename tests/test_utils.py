"""Unit tests for Utilities: TrajectoryInterpolator, PlateStorageManager, and ANPRVisualizer."""

from pathlib import Path
import numpy as np
import pytest
from src.utils.interpolator import TrajectoryInterpolator
from src.utils.storage import PlateStorageManager
from src.utils.visualizer import ANPRVisualizer


def test_trajectory_interpolation_math():
    """Tests 1D linear bounding box interpolation between two distant frames."""
    frames = [10, 15]
    bboxes = [[100.0, 100.0, 300.0, 300.0], [200.0, 200.0, 400.0, 400.0]]
    interpolated = TrajectoryInterpolator.interpolate_trajectory(frames, bboxes)

    assert len(interpolated) == 6  # Frames 10, 11, 12, 13, 14, 15
    assert np.isclose(interpolated[10][0], 100.0)
    assert np.isclose(interpolated[15][0], 200.0)
    # Midpoint frame 12 should be 100 + (2/5)*100 = 140
    assert np.isclose(interpolated[12][0], 140.0)
    assert np.isclose(interpolated[12][1], 140.0)


def test_frame_results_interpolation():
    """Tests multi-object trajectory interpolation across frame gaps."""
    raw_results = {
        0: {1: {"car_bbox": [10.0, 10.0, 50.0, 50.0], "plate_bbox": [20.0, 40.0, 35.0, 45.0]}},
        3: {1: {"car_bbox": [40.0, 40.0, 80.0, 80.0], "plate_bbox": [50.0, 70.0, 65.0, 75.0]}}
    }
    smooth = TrajectoryInterpolator.interpolate_frame_results(raw_results, max_gap=5)

    assert 1 in smooth  # Interpolated frame 1
    assert 2 in smooth  # Interpolated frame 2
    assert 1 in smooth[1]  # Car ID 1 is tracked in frame 1
    assert np.isclose(smooth[1][1]["car_bbox"][0], 20.0)


def test_storage_manager_webp(tmp_path, dummy_crop):
    """Tests lossless WebP crop compression and loading."""
    storage = PlateStorageManager(base_dir=str(tmp_path), webp_quality=85)
    saved_rel_path = storage.save_crop(dummy_crop, car_id=99, plate_number="TEST123")

    full_path = tmp_path / saved_rel_path
    assert full_path.exists()
    assert str(full_path).endswith(".webp")

    # Verify loading back
    loaded = storage.load_crop(str(full_path))
    assert loaded is not None
    assert loaded.shape == dummy_crop.shape


def test_visualizer_drawing(dummy_frame):
    """Tests ANPRVisualizer renders overlays without throwing exceptions."""
    viz = ANPRVisualizer()

    # Draw regular vehicle
    viz.draw_vehicle_overlay(
        frame=dummy_frame,
        car_id=1,
        car_bbox=(100, 100, 400, 400),
        plate_bbox=(200, 300, 300, 350),
        plate_number="SC56DYP",
        vehicle_type="car",
        is_watchlist=False
    )

    # Draw watchlist hit
    viz.draw_vehicle_overlay(
        frame=dummy_frame,
        car_id=2,
        car_bbox=(500, 100, 800, 400),
        plate_bbox=(600, 300, 700, 350),
        plate_number="EY61NBG",
        vehicle_type="car",
        is_watchlist=True,
        alert_level="CRITICAL",
        watchlist_reason="Stolen Vehicle"
    )

    # Draw top HUD
    viz.draw_top_hud(
        frame=dummy_frame,
        active_alerts=[{"plate_number": "EY61NBG", "alert_level": "CRITICAL", "reason": "Stolen"}],
        fps=60.0,
        frame_idx=150
    )

    # Frame should have non-zero pixels
    assert np.count_nonzero(dummy_frame) > 0
