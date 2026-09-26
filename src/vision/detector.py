"""
Unified Vehicle and License Plate Detector Module.
Supports TensorRT engine acceleration with PyTorch (.pt) fallback.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
from ultralytics import YOLO


@dataclass
class DetectionBox:
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


@dataclass
class MatchedVehiclePlate:
    vehicle_bbox: Tuple[float, float, float, float]
    vehicle_conf: float
    vehicle_type: str
    plate_bbox: Tuple[float, float, float, float]
    plate_conf: float
    plate_crop: np.ndarray  # Raw RGB crop of license plate


class VehiclePlateDetector:
    """
    High-performance Detector for vehicles and license plates.
    Prioritizes TensorRT .engine models for <3ms inference latency.
    """

    def __init__(
        self,
        vehicle_model_path: str = "Weight/yolov8n.engine",
        vehicle_pt_path: str = "Weight/yolov8n.pt",
        plate_model_path: str = "Weight/license_plate_detector.engine",
        plate_pt_path: str = "Weight/license_plate_detector.pt",
        vehicle_conf: float = 0.4,
        plate_conf: float = 0.3,
        vehicle_classes: Optional[List[int]] = None,
    ):
        self.vehicle_conf = vehicle_conf
        self.plate_conf = plate_conf
        # COCO class IDs: 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = vehicle_classes or [2, 3, 5, 7]

        # Load Vehicle Model
        self.vehicle_model = self._load_model(vehicle_model_path, vehicle_pt_path, "Vehicle Detector")

        # Load Plate Model
        self.plate_model = self._load_model(plate_model_path, plate_pt_path, "License Plate Detector")

    def _load_model(self, engine_path: str, pt_path: str, model_name: str) -> YOLO:
        """Load TensorRT engine if available, fallback to PyTorch weights."""
        engine_file = Path(engine_path)
        pt_file = Path(pt_path)

        if engine_file.exists():
            print(f"[✓] Loading {model_name} from TensorRT Engine: {engine_file}")
            return YOLO(str(engine_file), task="detect")
        elif pt_file.exists():
            print(f"[!] Warning: TensorRT engine not found. Falling back to PyTorch: {pt_file}")
            return YOLO(str(pt_file))
        else:
            raise FileNotFoundError(f"Neither {engine_file} nor {pt_file} exists for {model_name}!")

    def detect_vehicles(self, frame: np.ndarray) -> List[DetectionBox]:
        """
        Detect vehicles (cars, buses, trucks, motorcycles) in the given image/frame.
        """
        results = self.vehicle_model(
            frame,
            conf=self.vehicle_conf,
            classes=self.vehicle_classes,
            verbose=False
        )[0]

        detections = []
        names = results.names
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = names.get(cls_id, "vehicle")
            detections.append(DetectionBox(
                bbox=(coords[0], coords[1], coords[2], coords[3]),
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name
            ))
        return detections

    def detect_plates(self, frame: np.ndarray) -> List[DetectionBox]:
        """
        Detect license plates in the given image/frame.
        """
        results = self.plate_model(
            frame,
            conf=self.plate_conf,
            verbose=False
        )[0]

        detections = []
        names = results.names
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = names.get(cls_id, "license_plate")
            detections.append(DetectionBox(
                bbox=(coords[0], coords[1], coords[2], coords[3]),
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name
            ))
        return detections

    @staticmethod
    def match_plate_to_vehicle(
        plate_box: DetectionBox,
        vehicle_boxes: List[Tuple[float, float, float, float, int]]  # (x1, y1, x2, y2, track_id)
    ) -> Optional[Tuple[float, float, float, float, int]]:
        """
        Assign a license plate to the corresponding vehicle track.
        A plate is assigned if its center point falls within the vehicle bounding box.
        """
        px1, py1, px2, py2 = plate_box.bbox
        p_cx = (px1 + px2) / 2.0
        p_cy = (py1 + py2) / 2.0

        for vx1, vy1, vx2, vy2, track_id in vehicle_boxes:
            if vx1 <= p_cx <= vx2 and vy1 <= p_cy <= vy2:
                return (vx1, vy1, vx2, vy2, track_id)

        # Fallback: Check bounding box overlap if center is slightly on the edge
        for vx1, vy1, vx2, vy2, track_id in vehicle_boxes:
            # Overlap coordinates
            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                plate_area = (px2 - px1) * (py2 - py1)
                if plate_area > 0 and (inter_area / plate_area) > 0.6:
                    return (vx1, vy1, vx2, vy2, track_id)

        return None
