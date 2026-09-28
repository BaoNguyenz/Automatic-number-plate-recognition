#!/bin/bash
set -e

echo "================================================================================"
echo "🚦 ANPR Sentry Tactical v4.2 - Container Starting..."
echo "================================================================================"

# 1. Ensure required directory structure exists
mkdir -p /app/data /app/media/plates /app/media/videos /app/Weight

# 2. Check GPU & CUDA status
echo "[+] Checking CUDA environment..."
python3 -c "
import torch
cuda_ok = torch.cuda.is_available()
print(f'   CUDA Available: {cuda_ok}')
if cuda_ok:
    print(f'   GPU Device:     {torch.cuda.get_device_name(0)}')
    print(f'   VRAM Total:     {round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1)} MB')
else:
    print('   [!] Running in CPU Fallback mode.')
"

# 3. Check / Export TensorRT engines for Linux if GPU is active
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    VEHICLE_ENGINE="/app/Weight/yolov8n.engine"
    PLATE_ENGINE="/app/Weight/license_plate_detector.engine"

    if [ ! -f "$VEHICLE_ENGINE" ] || [ ! -f "$PLATE_ENGINE" ]; then
        echo "[*] TensorRT engine not found for current Linux GPU environment."
        echo "[*] Attempting TensorRT FP16 export for local hardware..."
        python3 /app/scripts/export_tensorrt.py || echo "[!] TensorRT export skipped/failed; will use PyTorch .pt fallback smoothly."
    else
        echo "[✓] Found existing TensorRT engines in /app/Weight."
    fi
fi

# 4. Start Server or run custom command
if [ "$#" -eq 0 ]; then
    echo "[+] Launching FastAPI Web Dashboard via Uvicorn on 0.0.0.0:8000..."
    exec python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
else
    echo "[+] Executing custom command: $@"
    exec "$@"
fi
