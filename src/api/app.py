"""
FastAPI Backend Application for ANPR Sentry Tactical UI.
Provides RESTful APIs for real-time and batch vehicle plate recognition,
integrating TensorRT, PaddleOCR / Qwen2-VL, SQLite database, and tactical HUD visualizations.
"""

import os
import sys
import time
import uuid
import base64
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import cv2
import yaml
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.connection import DatabaseManager
from src.database.models import VehicleDetection, Watchlist
from src.vision.detector import VehiclePlateDetector, DetectionBox
from src.recognition import PlatePostProcessor, Qwen2VLEngine, PaddleOCREngine
from src.utils.storage import PlateStorageManager


def load_config() -> Dict[str, Any]:
    cfg_path = ROOT / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


config = load_config()

app = FastAPI(
    title="ANPR Sentry Tactical API",
    description="High-Assurance Military & Security Vehicle Plate Recognition API",
    version="4.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Template directories
STATIC_DIR = ROOT / "src" / "api" / "static"
TEMPLATES_DIR = ROOT / "src" / "api" / "templates"
MEDIA_DIR = ROOT / "media" / "plates"
BENCHMARK_DIR = ROOT / "benchmark_crops"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/media", StaticFiles(directory=str(ROOT / "media")), name="media")
app.mount("/benchmark_crops", StaticFiles(directory=str(BENCHMARK_DIR)), name="benchmark_crops")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------
# Singleton System State & Engines
# ---------------------------------------------------------
class EngineManager:
    def __init__(self):
        self.db = DatabaseManager(db_url=config.get("database", {}).get("url", "sqlite:///data/anpr.db"))
        self.storage = PlateStorageManager(base_dir=str(ROOT / "media" / "plates"))
        self.detector: Optional[VehiclePlateDetector] = None
        self.paddle_engine: Optional[PaddleOCREngine] = None
        self.qwen_engine: Optional[Qwen2VLEngine] = None
        self.active_engine_name: str = "paddleocr"
        self._lock = threading.Lock()
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def get_detector(self) -> VehiclePlateDetector:
        if self.detector is None:
            with self._lock:
                if self.detector is None:
                    self.detector = VehiclePlateDetector(
                        vehicle_model_path=str(ROOT / config.get("paths", {}).get("coco_engine", "Weight/yolov8n.engine")),
                        vehicle_pt_path=str(ROOT / config.get("paths", {}).get("coco_model", "Weight/yolov8n.pt")),
                        plate_model_path=str(ROOT / config.get("paths", {}).get("plate_engine", "Weight/license_plate_detector.engine")),
                        plate_pt_path=str(ROOT / config.get("paths", {}).get("plate_model", "Weight/license_plate_detector.pt")),
                        vehicle_conf=config.get("vision", {}).get("vehicle_conf", 0.35),
                        plate_conf=config.get("vision", {}).get("plate_conf", 0.25)
                    )
        return self.detector

    def get_paddle_engine(self) -> PaddleOCREngine:
        if self.paddle_engine is None:
            with self._lock:
                if self.paddle_engine is None:
                    self.paddle_engine = PaddleOCREngine(
                        lang=config.get("recognition", {}).get("paddle_lang", "en"),
                        use_angle_cls=config.get("recognition", {}).get("paddle_use_angle_cls", False),
                        show_log=False
                    )
        return self.paddle_engine

    def get_qwen_engine(self) -> Qwen2VLEngine:
        if self.qwen_engine is None:
            with self._lock:
                if self.qwen_engine is None:
                    lora_dir = ROOT / config.get("paths", {}).get("vlm_lora_dir", "Weight/qwen2_vl_lora_plate")
                    self.qwen_engine = Qwen2VLEngine(
                        model_id=config.get("paths", {}).get("vlm_model_id", "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"),
                        processor_id="Qwen/Qwen2-VL-2B-Instruct",
                        lora_dir=str(lora_dir) if lora_dir.exists() else None,
                        prompt=config.get("vlm", {}).get("prompt", None),
                        max_new_tokens=config.get("vlm", {}).get("max_new_tokens", 15)
                    )
        return self.qwen_engine

    def get_recognition_engine(self, engine_name: Optional[str] = None):
        target = (engine_name or self.active_engine_name).lower()
        if "qwen" in target:
            return self.get_qwen_engine(), "qwen2_vl"
        return self.get_paddle_engine(), "paddleocr"


engine_mgr = EngineManager()


# ---------------------------------------------------------
# Helper Functions for Tactical HUD Visualization
# ---------------------------------------------------------
def draw_tactical_corner_bracket(img: np.ndarray, bbox: List[int], color: Tuple[int, int, int], thickness: int = 2, length: int = 14):
    """Draws 4 corner brackets instead of solid bounding box for military HUD look."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    l = min(length, w // 4, h // 4)

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + l, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - l, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + l, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - l, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, thickness)


# ---------------------------------------------------------
# Web & API Endpoints
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Renders the Tactical Video & File Analysis Workspace dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/system/status")
async def get_system_status():
    """Returns GPU metrics, loaded models, database detection counts, and watchlist size."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Fallback)"
    gpu_mem_used_mb = 0
    gpu_mem_total_mb = 0

    if cuda_available:
        gpu_mem_used_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_mem_total_mb = round(total_bytes / (1024 * 1024), 1)

    total_detections = 0
    total_watchlist = 0
    try:
        with engine_mgr.db.get_session() as s:
            total_detections = s.query(VehicleDetection).count()
            total_watchlist = s.query(Watchlist).count()
    except Exception as e:
        print(f"Error querying db stats: {e}")

    return {
        "status": "online",
        "device": device_name,
        "cuda_available": cuda_available,
        "gpu_memory_used_mb": gpu_mem_used_mb,
        "gpu_memory_total_mb": gpu_mem_total_mb,
        "active_engine": engine_mgr.active_engine_name,
        "total_detections": total_detections,
        "total_watchlist": total_watchlist
    }


@app.post("/api/engine/switch")
async def switch_engine(engine: str = Form(...)):
    """Switches the active default recognition engine."""
    clean_engine = engine.strip().lower()
    if clean_engine not in ["paddleocr", "qwen2_vl"]:
        raise HTTPException(status_code=400, detail="Invalid engine. Options: paddleocr, qwen2_vl")
    engine_mgr.active_engine_name = clean_engine
    return {"message": f"Active recognition engine switched to {clean_engine}", "active_engine": clean_engine}


@app.get("/api/sample_images")
async def list_sample_images():
    """Lists available benchmark crops for fast 1-click testing."""
    samples = []
    if BENCHMARK_DIR.exists():
        for f in BENCHMARK_DIR.glob("*.jpg"):
            samples.append({
                "filename": f.name,
                "url": f"/benchmark_crops/{f.name}",
                "size_kb": round(f.stat().st_size / 1024, 1)
            })
    return {"samples": samples[:15]}


@app.get("/api/detections")
async def get_detections(
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None,
    filter_type: str = "all"
):
    """Fetches recent detections from SQLite database."""
    with engine_mgr.db.get_session() as s:
        query = s.query(VehicleDetection)

        if search:
            query = query.filter(VehicleDetection.plate_number.ilike(f"%{search.strip()}%"))

        if filter_type == "watchlist":
            query = query.filter(VehicleDetection.is_watchlist_match == True)
        elif filter_type == "high_conf":
            query = query.filter(VehicleDetection.confidence_score >= 0.90)

        query = query.order_by(VehicleDetection.detected_at.desc())
        total = query.count()
        results = query.offset(offset).limit(limit).all()

        data = []
        for r in results:
            data.append({
                "id": r.id,
                "car_id": r.car_id,
                "plate_number": r.plate_number,
                "raw_vlm_text": r.raw_vlm_text,
                "confidence_score": round(r.confidence_score or 0.0, 3),
                "vehicle_type": r.vehicle_type,
                "vehicle_color": r.vehicle_color,
                "frame_number": r.frame_number,
                "plate_crop_path": r.plate_crop_path,
                "detected_at": r.detected_at.strftime("%Y-%m-%d %H:%M:%S") if r.detected_at else "",
                "is_watchlist_match": bool(r.is_watchlist_match),
                "watchlist_reason": r.watchlist_reason
            })

    return {"total": total, "detections": data}


@app.get("/api/watchlist")
async def get_watchlist():
    """Lists all vehicle plates in the watchlist."""
    with engine_mgr.db.get_session() as s:
        items = s.query(Watchlist).all()
        return [
            {
                "id": item.id,
                "plate_number": item.plate_number,
                "vehicle_owner": item.vehicle_owner,
                "reason": item.reason,
                "alert_level": item.alert_level,
                "created_at": item.created_at.strftime("%Y-%m-%d %H:%M:%S") if item.created_at else ""
            }
            for item in items
        ]


class WatchlistCreate(BaseModel):
    plate_number: str
    vehicle_owner: Optional[str] = None
    reason: str
    alert_level: str = "CRITICAL"


@app.post("/api/watchlist")
async def add_watchlist(item: WatchlistCreate):
    """Adds a plate number to the watchlist."""
    clean_plate = PlatePostProcessor.clean_plate_text(item.plate_number)[0]
    with engine_mgr.db.get_session() as s:
        existing = s.query(Watchlist).filter(Watchlist.plate_number == clean_plate).first()
        if existing:
            raise HTTPException(status_code=400, detail="Plate already exists in watchlist")

        new_entry = Watchlist(
            plate_number=clean_plate,
            vehicle_owner=item.vehicle_owner,
            reason=item.reason,
            alert_level=item.alert_level
        )
        s.add(new_entry)
    return {"message": f"Plate '{clean_plate}' added to watchlist", "plate": clean_plate}


@app.delete("/api/watchlist/{plate_number}")
async def delete_watchlist(plate_number: str):
    """Removes a plate from the watchlist."""
    with engine_mgr.db.get_session() as s:
        item = s.query(Watchlist).filter(Watchlist.plate_number == plate_number.upper()).first()
        if not item:
            raise HTTPException(status_code=404, detail="Plate not found in watchlist")
        s.delete(item)
    return {"message": f"Plate '{plate_number}' deleted from watchlist"}


# ---------------------------------------------------------
# Image Analysis Endpoint (Single Image / Crop Quick Test)
# ---------------------------------------------------------
@app.post("/api/analyze/image")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None),
    engine: Optional[str] = Form(None)
):
    """
    Performs full ANPR analysis on an uploaded image or sample benchmark crop:
    1. Detects vehicle & license plate bounding boxes via TensorRT.
    2. Performs OCR / VLM inference (PaddleOCR or Qwen2-VL).
    3. Normalizes plate text and checks watchlist.
    4. Renders tactical HUD overlay annotations.
    """
    t_start = time.perf_counter()

    # Read image
    if file and file.filename:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif sample_filename:
        sample_path = BENCHMARK_DIR / sample_filename
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample file {sample_filename} not found")
        img = cv2.imread(str(sample_path))
    else:
        raise HTTPException(status_code=400, detail="Either file upload or sample_filename must be provided")

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    h, w = img.shape[:2]
    annotated = img.copy()

    # Check if image itself is already a cropped plate (e.g. from benchmark_crops)
    # A crop usually has smaller dimension, e.g. width < 400 and aspect ratio > 1.8
    is_direct_plate_crop = (w < 500 and h < 250)

    detector = engine_mgr.get_detector()
    rec_engine, engine_used = engine_mgr.get_recognition_engine(engine)

    detections_output = []
    has_hotlist = False
    hotlist_info = None

    t_det_start = time.perf_counter()

    if is_direct_plate_crop:
        # Direct Plate Crop Test
        det_time_ms = round((time.perf_counter() - t_det_start) * 1000, 2)
        plate_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t_ocr = time.perf_counter()
        pred = rec_engine.recognize_plate(plate_rgb, vehicle_type="car")
        ocr_lat_ms = round((time.perf_counter() - t_ocr) * 1000, 2)

        clean_text = pred.get("plate_number", "")
        raw_text = pred.get("raw_text", "")
        conf = float(pred.get("confidence_score", 0.95))
        is_valid = bool(pred.get("is_valid_format", False))

        # Check watchlist
        with engine_mgr.db.get_session() as s:
            w_item = s.query(Watchlist).filter(Watchlist.plate_number == clean_text).first()
            is_match = w_item is not None
            reason = w_item.reason if w_item else None
            alert_lvl = w_item.alert_level if w_item else "INFO"

        # Save crop WebP
        crop_rel_path = engine_mgr.storage.save_crop(plate_rgb, car_id=1, plate_number=clean_text)

        # Save to database
        with engine_mgr.db.get_session() as s:
            record = VehicleDetection(
                car_id=1,
                plate_number=clean_text if clean_text else "UNKNOWN",
                raw_vlm_text=raw_text,
                confidence_score=conf,
                vehicle_type="car",
                frame_number=0,
                plate_crop_path=crop_rel_path,
                is_watchlist_match=is_match,
                watchlist_reason=reason
            )
            s.add(record)

        if is_match:
            has_hotlist = True
            hotlist_info = {
                "plate": clean_text,
                "reason": reason,
                "level": alert_lvl
            }

        # Tactical Draw on crop
        bracket_color = (68, 68, 239) if is_match else (212, 182, 6) # BGR
        draw_tactical_corner_bracket(annotated, [4, 4, w - 4, h - 4], bracket_color, thickness=2)

        detections_output.append({
            "car_id": 1,
            "vehicle_bbox": [0, 0, w, h],
            "vehicle_type": "car",
            "vehicle_conf": 1.0,
            "plate_bbox": [0, 0, w, h],
            "plate_conf": conf,
            "plate_text": clean_text if clean_text else raw_text,
            "raw_text": raw_text,
            "is_valid_format": is_valid,
            "ocr_latency_ms": ocr_lat_ms,
            "is_watchlist_match": is_match,
            "watchlist_reason": reason,
            "crop_url": f"/{crop_rel_path}" if crop_rel_path else None
        })

    else:
        # Full Image Surveillance Pipeline
        vehicles = detector.detect_vehicles(img)
        plates = detector.detect_plates(img)
        det_time_ms = round((time.perf_counter() - t_det_start) * 1000, 2)

        # Vehicle bbox tuples: (x1, y1, x2, y2, index)
        v_tuples = [(int(v.bbox[0]), int(v.bbox[1]), int(v.bbox[2]), int(v.bbox[3]), idx + 1) for idx, v in enumerate(vehicles)]

        for p_idx, plate in enumerate(plates):
            px1, py1, px2, py2 = [int(c) for c in plate.bbox]
            # Clip coordinates
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)

            if px2 <= px1 or py2 <= py1:
                continue

            plate_crop_bgr = img[py1:py2, px1:px2]
            plate_rgb = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2RGB)

            matched_v = detector.match_plate_to_vehicle(plate, v_tuples)
            car_id = matched_v[4] if matched_v else (p_idx + 1)
            v_type = vehicles[car_id - 1].class_name if (matched_v and car_id - 1 < len(vehicles)) else "vehicle"

            # OCR Inference
            t_ocr = time.perf_counter()
            pred = rec_engine.recognize_plate(plate_rgb, vehicle_type=v_type)
            ocr_lat_ms = round((time.perf_counter() - t_ocr) * 1000, 2)

            clean_text = pred.get("plate_number", "")
            raw_text = pred.get("raw_text", "")
            conf = float(pred.get("confidence_score", 0.95))
            is_valid = bool(pred.get("is_valid_format", False))

            # Check Watchlist
            with engine_mgr.db.get_session() as s:
                w_item = s.query(Watchlist).filter(Watchlist.plate_number == clean_text).first()
                is_match = w_item is not None
                reason = w_item.reason if w_item else None
                alert_lvl = w_item.alert_level if w_item else "INFO"

            # Save Crop
            crop_rel_path = engine_mgr.storage.save_crop(plate_rgb, car_id=car_id, plate_number=clean_text)

            # Record in SQLite
            with engine_mgr.db.get_session() as s:
                rec = VehicleDetection(
                    car_id=car_id,
                    plate_number=clean_text if clean_text else "UNKNOWN",
                    raw_vlm_text=raw_text,
                    confidence_score=conf,
                    vehicle_type=v_type,
                    frame_number=0,
                    plate_crop_path=crop_rel_path,
                    is_watchlist_match=is_match,
                    watchlist_reason=reason
                )
                s.add(rec)

            if is_match:
                has_hotlist = True
                hotlist_info = {
                    "plate": clean_text,
                    "reason": reason,
                    "level": alert_lvl
                }

            # Draw Annotations on Full Image
            bracket_color = (68, 68, 239) if is_match else (212, 182, 6) # BGR
            # Draw on Vehicle
            if matched_v:
                vx1, vy1, vx2, vy2, _ = matched_v
                draw_tactical_corner_bracket(annotated, [vx1, vy1, vx2, vy2], bracket_color, thickness=2, length=20)
                # Label tag
                tag_y = max(18, vy1 - 6)
                cv2.rectangle(annotated, (vx1, tag_y - 18), (vx1 + 130, tag_y), (15, 19, 29), -1)
                cv2.putText(annotated, f"{v_type.upper()} #{car_id}", (vx1 + 4, tag_y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (241, 226, 223), 1, cv2.LINE_AA)

            # Draw on Plate
            cv2.rectangle(annotated, (px1, py1), (px2, py2), bracket_color, 2)
            plate_tag_y = max(14, py1 - 6)
            tag_text = f"{clean_text} ({int(conf * 100)}%)" if clean_text else "PLATE"
            cv2.rectangle(annotated, (px1, plate_tag_y - 16), (px1 + len(tag_text) * 9, plate_tag_y + 2), (15, 19, 29), -1)
            cv2.putText(annotated, tag_text, (px1 + 3, plate_tag_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (76, 215, 246) if not is_match else (68, 68, 239), 1, cv2.LINE_AA)

            detections_output.append({
                "car_id": car_id,
                "vehicle_bbox": [matched_v[0], matched_v[1], matched_v[2], matched_v[3]] if matched_v else None,
                "vehicle_type": v_type,
                "vehicle_conf": float(vehicles[car_id - 1].confidence) if (matched_v and car_id - 1 < len(vehicles)) else 1.0,
                "plate_bbox": [px1, py1, px2, py2],
                "plate_conf": conf,
                "plate_text": clean_text if clean_text else raw_text,
                "raw_text": raw_text,
                "is_valid_format": is_valid,
                "ocr_latency_ms": ocr_lat_ms,
                "is_watchlist_match": is_match,
                "watchlist_reason": reason,
                "crop_url": f"/{crop_rel_path}" if crop_rel_path else None
            })

    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # Encode annotated image to JPEG Base64
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    return {
        "status": "success",
        "engine_used": engine_used,
        "detection_latency_ms": det_time_ms,
        "total_latency_ms": total_time_ms,
        "total_vehicles_detected": len(detections_output),
        "has_hotlist_hit": has_hotlist,
        "hotlist_hit": hotlist_info,
        "detections": detections_output,
        "annotated_image": annotated_b64
    }


# ---------------------------------------------------------
# Background Video Pipeline Job Runner
# ---------------------------------------------------------
def _run_video_job(job_id: str, video_path: str, max_frames: int, engine_name: str):
    """Background worker executing the video pipeline and reporting frame-by-frame status."""
    from scripts.run_pipeline import run_pipeline

    job = engine_mgr.jobs[job_id]
    job["status"] = "processing"
    job["start_time"] = time.time()

    try:
        run_pipeline(
            video_path=video_path,
            max_frames=max_frames,
            engine=engine_name,
            skip_render=True
        )
        job["status"] = "completed"
        job["progress_percent"] = 100
        job["duration_s"] = round(time.time() - job["start_time"], 2)
        job["video_url"] = f"/videos/{Path(video_path).name}"
        if (ROOT / "out.mp4").exists():
            job["annotated_video_url"] = "/videos/out.mp4"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


@app.get("/videos/{video_name}")
async def serve_video(video_name: str):
    """Streams video file supporting HTML5 range seeking."""
    p1 = ROOT / video_name
    p2 = ROOT / "media" / video_name
    if p1.exists() and p1.is_file():
        target = p1
    elif p2.exists() and p2.is_file():
        target = p2
    else:
        raise HTTPException(status_code=404, detail=f"Video file '{video_name}' not found")

    return FileResponse(path=str(target), media_type="video/mp4")


@app.post("/api/analyze/video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    max_frames: int = Form(150),
    engine: Optional[str] = Form("paddleocr")
):
    """Uploads and launches video pipeline analysis in background."""
    job_id = str(uuid.uuid4())[:8]

    if file and file.filename:
        video_filename = f"upload_{job_id}_{file.filename}"
        video_path = ROOT / "media" / video_filename
        with open(video_path, "wb") as f:
            f.write(await file.read())
        rel_video_path = str(video_path.relative_to(ROOT))
    else:
        # Default sample video in repo
        rel_video_path = config.get("paths", {}).get("input_video", "2103099-uhd_3840_2160_30fps.mp4")

    engine_mgr.jobs[job_id] = {
        "job_id": job_id,
        "video_path": rel_video_path,
        "max_frames": max_frames,
        "engine": engine,
        "status": "queued",
        "progress_percent": 0
    }

    background_tasks.add_task(_run_video_job, job_id, rel_video_path, max_frames, engine)

    return {
        "job_id": job_id,
        "status": "queued",
        "video": rel_video_path,
        "max_frames": max_frames,
        "engine": engine,
        "status_url": f"/api/job/{job_id}/status"
    }


@app.get("/api/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Returns the execution state of a video analysis job."""
    if job_id not in engine_mgr.jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return engine_mgr.jobs[job_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)
