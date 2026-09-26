"""
Vehicle Tracker and Best-Frame Selector Module.
Maintains persistent car_id trajectories and selects the optimal license plate crop for VLM recognition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from ultralytics import YOLO

from src.vision.detector import DetectionBox, VehiclePlateDetector


@dataclass
class TrackedVehicle:
    car_id: int
    vehicle_type: str
    vehicle_bbox: Tuple[float, float, float, float]
    confidence: float
    first_frame: int
    last_frame: int
    missed_frames: int = 0
    # Best-Frame Crop details
    best_crop: Optional[np.ndarray] = None
    best_crop_quality: float = 0.0
    best_crop_frame: int = -1
    best_plate_bbox: Optional[Tuple[float, float, float, float]] = None
    best_plate_conf: float = 0.0
    # Recognition Status
    is_processed: bool = False
    plate_text: Optional[str] = None


class BestFrameTracker:
    """
    Integrates ByteTrack multi-object tracking with a Sharpness/Scale-aware
    Best-Frame Selector to minimize redundant VLM inference.
    """

    def __init__(
        self,
        detector: VehiclePlateDetector,
        max_missed_frames: int = 15,
        min_crop_area: int = 400,
        tracker_type: str = "bytetrack.yaml",
    ):
        self.detector = detector
        self.max_missed_frames = max_missed_frames
        self.min_crop_area = min_crop_area
        self.tracker_type = tracker_type
        self.tracks: Dict[int, TrackedVehicle] = {}

    @staticmethod
    def calculate_quality_score(crop: np.ndarray, plate_conf: float) -> float:
        """
        Calculate an objective quality score for a license plate crop.
        Combines:
        1. Bounding box area (scale)
        2. Laplacian variance (image sharpness/clarity)
        3. YOLO detection confidence
        4. Aspect ratio compliance (UK/EU plates typically 2.0 ~ 5.5)
        """
        h, w = crop.shape[:2]
        area = w * h
        if area < 100 or h < 10 or w < 20:
            return 0.0

        # Compute image sharpness via Laplacian variance
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Aspect ratio check
        aspect_ratio = w / float(h)
        ar_weight = 1.0 if (2.0 <= aspect_ratio <= 5.5) else 0.7

        # Objective Quality Metric
        score = plate_conf * np.sqrt(area) * np.log1p(sharpness) * ar_weight
        return float(score)

    def track_and_associate(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> Tuple[List[TrackedVehicle], List[TrackedVehicle]]:
        """
        Processes a video frame:
        1. Runs ByteTrack on vehicles.
        2. Detects license plates with TensorRT plate detector.
        3. Associates plates to tracked vehicles.
        4. Updates the Best-Frame candidate for each vehicle.
        5. Returns: (active_vehicles_in_frame, finished_vehicles_ready_for_recognition)
        """
        # Step 1: Track vehicles with ByteTrack
        results = self.detector.vehicle_model.track(
            frame,
            persist=True,
            tracker=self.tracker_type,
            conf=self.detector.vehicle_conf,
            classes=self.detector.vehicle_classes,
            verbose=False
        )[0]

        active_track_ids = set()
        active_vehicles: List[TrackedVehicle] = []
        vehicle_boxes_for_matching: List[Tuple[float, float, float, float, int]] = []

        if results.boxes is not None and len(results.boxes) > 0:
            names = results.names
            for box in results.boxes:
                if box.id is None:
                    continue  # Unassigned track
                car_id = int(box.id[0].cpu().numpy())
                coords = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                v_type = names.get(cls_id, "car")

                active_track_ids.add(car_id)
                vehicle_boxes_for_matching.append((coords[0], coords[1], coords[2], coords[3], car_id))

                if car_id not in self.tracks:
                    self.tracks[car_id] = TrackedVehicle(
                        car_id=car_id,
                        vehicle_type=v_type,
                        vehicle_bbox=(coords[0], coords[1], coords[2], coords[3]),
                        confidence=conf,
                        first_frame=frame_number,
                        last_frame=frame_number,
                    )
                else:
                    track = self.tracks[car_id]
                    track.vehicle_bbox = (coords[0], coords[1], coords[2], coords[3])
                    track.confidence = conf
                    track.last_frame = frame_number
                    track.missed_frames = 0

                active_vehicles.append(self.tracks[car_id])

        # Step 2: Detect License Plates
        plates = self.detector.detect_plates(frame)

        # Step 3: Match plates to vehicles and update best candidate
        for plate in plates:
            match = self.detector.match_plate_to_vehicle(plate, vehicle_boxes_for_matching)
            if match is not None:
                _, _, _, _, car_id = match
                px1, py1, px2, py2 = [int(v) for v in plate.bbox]
                # Boundary clipping
                H, W = frame.shape[:2]
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(W, px2), min(H, py2)

                if (px2 - px1) > 0 and (py2 - py1) > 0:
                    crop = frame[py1:py2, px1:px2].copy()
                    quality = self.calculate_quality_score(crop, plate.confidence)

                    track = self.tracks[car_id]
                    # Update if new crop has strictly better quality
                    if quality > track.best_crop_quality:
                        track.best_crop = crop
                        track.best_crop_quality = quality
                        track.best_crop_frame = frame_number
                        track.best_plate_bbox = plate.bbox
                        track.best_plate_conf = plate.confidence

        # Step 4: Check for completed / departed tracks
        ready_for_recognition: List[TrackedVehicle] = []
        for car_id, track in list(self.tracks.items()):
            if car_id not in active_track_ids:
                track.missed_frames += 1
                if track.missed_frames >= self.max_missed_frames:
                    if not track.is_processed and track.best_crop is not None:
                        ready_for_recognition.append(track)
                        track.is_processed = True

        return active_vehicles, ready_for_recognition

    def finalize_remaining_tracks(self) -> List[TrackedVehicle]:
        """
        Called at end of video to retrieve all remaining tracks that have not been recognized yet.
        """
        remaining = []
        for track in self.tracks.values():
            if not track.is_processed and track.best_crop is not None:
                remaining.append(track)
                track.is_processed = True
        return remaining
