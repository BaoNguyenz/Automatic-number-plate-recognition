"""
TensorRT Model Exporter for YOLOv8 Vehicle & Plate Detectors.
Ensures reproducible engine building with TensorRT 11.3 on NVIDIA GPUs.
"""

from pathlib import Path
from ultralytics import YOLO

def export_models():
    weights_dir = Path("Weight")
    models = [
        weights_dir / "yolov8n.pt",
        weights_dir / "license_plate_detector.pt"
    ]
    
    for model_path in models:
        if not model_path.exists():
            print(f"[-] Warning: {model_path} does not exist.")
            continue
            
        print(f"[+] Loading {model_path} for TensorRT export...")
        model = YOLO(str(model_path))
        
        # Exporting to TensorRT FP32/FP16 without nvidia-modelopt on Windows
        print(f"[+] Exporting {model_path} to TensorRT engine...")
        engine_path = model.export(format="engine", device=0)
        print(f"[✓] Successfully exported to: {engine_path}")

if __name__ == "__main__":
    export_models()
