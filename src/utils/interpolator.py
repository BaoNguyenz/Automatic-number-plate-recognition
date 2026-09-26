"""
Trajectory and Bounding Box Interpolator Module.
Fills missing detection gaps for smoothly moving vehicles and plate coordinates.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class TrajectoryInterpolator:
    """
    Interpolates missing bounding boxes and smooths tracking trajectories.
    Eliminates flickering and missing detections when vehicles pass under obstacles or low-confidence angles.
    """

    @staticmethod
    def interpolate_trajectory(
        frames: List[int],
        bboxes: List[List[float]],
        max_gap: int = 30
    ) -> Dict[int, List[float]]:
        """
        Interpolates 4D bounding boxes (x1, y1, x2, y2) across missing frame numbers.
        Only gaps <= max_gap are interpolated to prevent joining disconnected tracks.
        """
        if len(frames) == 0:
            return {}
        if len(frames) == 1:
            return {frames[0]: bboxes[0]}

        sorted_pairs = sorted(zip(frames, bboxes), key=lambda x: x[0])
        sorted_frames = [p[0] for p in sorted_pairs]
        sorted_boxes = np.array([p[1] for p in sorted_pairs], dtype=np.float32)

        result_dict: Dict[int, List[float]] = {}
        for i in range(len(sorted_frames) - 1):
            f_curr = sorted_frames[i]
            f_next = sorted_frames[i + 1]
            box_curr = sorted_boxes[i]
            box_next = sorted_boxes[i + 1]

            result_dict[f_curr] = box_curr.tolist()
            gap = f_next - f_curr

            if 1 < gap <= max_gap:
                x_pts = np.array([f_curr, f_next])
                x_new = np.arange(f_curr + 1, f_next)
                interp = interp1d(x_pts, np.vstack([box_curr, box_next]), axis=0, kind="linear")
                interp_boxes = interp(x_new)
                for f_interp, b_interp in zip(x_new, interp_boxes):
                    result_dict[int(f_interp)] = b_interp.tolist()

        result_dict[sorted_frames[-1]] = sorted_boxes[-1].tolist()
        return result_dict

    @classmethod
    def interpolate_frame_results(
        cls,
        results: Dict[int, Dict[int, Dict[str, Any]]],
        max_gap: int = 30
    ) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Takes frame-indexed tracking results:
            results[frame_nmr][car_id] = {
                'car_bbox': [x1, y1, x2, y2],
                'plate_bbox': [x1, y1, x2, y2],
                'plate_number': 'SC56DYP',
                ...
            }
        Interpolates missing frames per car_id and returns a fully continuous results dictionary.
        """
        # Group by car_id
        car_trajectories: Dict[int, Dict[str, Any]] = {}
        for frame_nmr, cars in results.items():
            for car_id, data in cars.items():
                if car_id not in car_trajectories:
                    car_trajectories[car_id] = {
                        "frames": [],
                        "car_bboxes": [],
                        "plate_frames": [],
                        "plate_bboxes": [],
                        "meta": {}
                    }
                car_trajectories[car_id]["frames"].append(frame_nmr)
                car_trajectories[car_id]["car_bboxes"].append(data.get("car_bbox", [0, 0, 0, 0]))

                if "plate_bbox" in data and data["plate_bbox"] is not None:
                    car_trajectories[car_id]["plate_frames"].append(frame_nmr)
                    car_trajectories[car_id]["plate_bboxes"].append(data["plate_bbox"])

                # Preserve metadata (plate number, type, alerts)
                for k, v in data.items():
                    if k not in ["car_bbox", "plate_bbox"]:
                        car_trajectories[car_id]["meta"][k] = v

        # Interpolate per car
        interpolated_results: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for car_id, traj in car_trajectories.items():
            interp_car_boxes = cls.interpolate_trajectory(traj["frames"], traj["car_bboxes"], max_gap=max_gap)
            interp_plate_boxes = {}
            if len(traj["plate_frames"]) > 0:
                interp_plate_boxes = cls.interpolate_trajectory(traj["plate_frames"], traj["plate_bboxes"], max_gap=max_gap)

            for f_nmr, c_box in interp_car_boxes.items():
                if f_nmr not in interpolated_results:
                    interpolated_results[f_nmr] = {}

                entry: Dict[str, Any] = {
                    "car_bbox": c_box,
                    "plate_bbox": interp_plate_boxes.get(f_nmr, None)
                }
                entry.update(traj["meta"])
                interpolated_results[f_nmr][car_id] = entry

        return interpolated_results
