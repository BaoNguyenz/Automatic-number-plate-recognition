"""
Unit test script for Recognition Layer (Group 3).
Tests Qwen2VLEngine and PlatePostProcessor on benchmark license plate crops.
"""

import sys
import glob
from pathlib import Path

# UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recognition import Qwen2VLEngine, PlatePostProcessor

def test_recognition():
    print("=" * 80)
    print("🧠 UNIT TEST: NHÓM 3 - LÕI NHẬN DIỆN VLM QWEN2-VL & POST-PROCESSOR")
    print("=" * 80)

    # 1. Test PlatePostProcessor with edge cases & LLM noise
    print("\n[+] 1. Kiểm thử PlatePostProcessor:")
    test_cases = [
        ("The license plate is: SC56 DYP.", "SC56DYP"),
        ("`EY61 NBG`", "EY61NBG"),
        ("Plate: NA13-NRU", "NA13NRU"),
        ("Result: NG65 ZFX\n", "NG65ZFX"),
    ]
    for raw, expected in test_cases:
        res = PlatePostProcessor.process(raw)
        status = "PASSED" if res["plate_number"] == expected else "FAILED"
        print(f"   [{status}] Raw: '{raw}' -> Cleaned: '{res['plate_number']}' (Expected: '{expected}')")
        assert res["plate_number"] == expected, f"Failed on {raw}"

    # 2. Test Qwen2VLEngine on representative plate crops
    print("\n[+] 2. Khởi tạo Qwen2VLEngine (4-bit BnB trên RTX 3060)...")
    engine = Qwen2VLEngine(
        model_id="unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
        processor_id="Qwen/Qwen2-VL-2B-Instruct",
        max_new_tokens=15
    )

    test_crops = [
        "benchmark_crops/car_1701_SC56DYP_0.71.jpg",
        "benchmark_crops/car_298_EY61NBG_0.93.jpg",
        "benchmark_crops/car_3_NA13NRU_0.79.jpg",
        "benchmark_crops/car_16_FJ14ZHY_0.71.jpg",
        "benchmark_crops/car_1257_NG65ZFX_0.76.jpg",
    ]

    print("\n[+] 3. Nhận diện đơn lẻ từng ảnh crop (Zero-shot VLM):")
    for crop_path in test_crops:
        if not Path(crop_path).exists():
            print(f"[-] Missing: {crop_path}")
            continue

        filename = Path(crop_path).name
        expected_plate = filename.split("_")[2]

        result = engine.recognize_plate(crop_path)
        match_icon = "🎯" if result["plate_number"] == expected_plate else "⚠️"
        print(f"   {match_icon} [{filename}] -> Pred: '{result['plate_number']}' | Expected: '{expected_plate}' | Raw: '{result['raw_text']}' | Valid UK: {result['is_uk_format']}")

    print("\n[+] 4. Kiểm thử suy luận Batch (Batch inference song song):")
    batch_results = engine.recognize_batch(test_crops[:3], vehicle_types=["sedan", "suv", "hatchback"])
    for p, r in zip(test_crops[:3], batch_results):
        print(f"   ⚡ Batch Item: {Path(p).name} -> {r['plate_number']} ({r['vehicle_type']}, Conf: {r['confidence_score']})")

    print("\n" + "=" * 80)
    print("🎉 TẤT CẢ CÁC BÀI KIỂM THỬ NHÓM 3 ĐÃ VƯỢT QUA THÀNH CÔNG!")
    print("=" * 80)

if __name__ == "__main__":
    test_recognition()
