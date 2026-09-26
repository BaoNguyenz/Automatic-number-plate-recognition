"""
Evaluation script to compare Base Zero-Shot Qwen2-VL vs Fine-Tuned QLoRA Qwen2-VL.
Evaluates Exact Match and Character Accuracy on benchmark_crops/.

Usage:
    python scripts/evaluate_finetune.py
"""

import sys
import glob
import time
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recognition.postprocessor import PlatePostProcessor

MODEL_ID = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"
PROCESSOR_ID = "Qwen/Qwen2-VL-2B-Instruct"
LORA_DIR = Path("Weight/qwen2_vl_lora_plate")
PROMPT = (
    "Read the vehicle license plate number shown in this cropped image. "
    "Output ONLY the alphanumeric plate characters with no extra spaces, punctuation, or explanation."
)

def compute_char_acc(pred: str, gt: str) -> float:
    """Computes Levenshtein character-level similarity."""
    import difflib
    matcher = difflib.SequenceMatcher(None, pred, gt)
    return matcher.ratio()

def evaluate_model(model, processor, crops: List[Path], model_label: str):
    print(f"\n[+] Đang đánh giá {model_label} trên {len(crops)} ảnh benchmark...")
    exact_matches = 0
    total_char_acc = 0.0
    start_time = time.time()

    for crop_path in crops:
        parts = crop_path.stem.split("_")
        expected_plate = parts[2].upper()

        pil_img = Image.open(str(crop_path)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": PROMPT}
                ]
            }
        ]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.inference_mode():
            gen_ids = model.generate(**inputs, max_new_tokens=15, do_sample=False)

        gen_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
        ]
        raw_output = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]
        cleaned_pred = PlatePostProcessor.clean_text(raw_output)

        is_match = (cleaned_pred == expected_plate)
        if is_match:
            exact_matches += 1

        c_acc = compute_char_acc(cleaned_pred, expected_plate)
        total_char_acc += c_acc

    total_time = time.time() - start_time
    exact_acc = (exact_matches / len(crops)) * 100
    avg_char_acc = (total_char_acc / len(crops)) * 100
    avg_latency = (total_time / len(crops)) * 1000

    print("=" * 65)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ: {model_label}")
    print("=" * 65)
    print(f"  • Độ chính xác Khớp Tuyệt Đối (Exact Match):  {exact_acc:.1f}% ({exact_matches}/{len(crops)})")
    print(f"  • Độ chính xác Cấp Ký Tự (Character Acc):    {avg_char_acc:.1f}%")
    print(f"  • Tốc độ suy luận trung bình:               {avg_latency:.1f} ms/ảnh")
    print("=" * 65)
    return exact_acc, avg_char_acc

def main():
    crops = sorted(list(Path("benchmark_crops").glob("*.jpg")))
    if not crops:
        print("[-] Không tìm thấy ảnh trong benchmark_crops/")
        return

    print("=" * 65)
    print("🔬 BÀI KIỂM TRA ĐỐI ĐẦU: BASE QWEN2-VL vs FINE-TUNED QLoRA")
    print("=" * 65)

    processor = AutoProcessor.from_pretrained(PROCESSOR_ID, use_fast=False)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16
    )

    # 1. Evaluate Base Model
    base_exact, base_char = evaluate_model(base_model, processor, crops, "Base Qwen2-VL-2B (Zero-Shot)")

    # 2. Evaluate LoRA Model if available
    if LORA_DIR.exists() and (LORA_DIR / "adapter_config.json").exists():
        print(f"\n[+] Tìm thấy LoRA adapter tại '{LORA_DIR}'. Đang gắn vào mô hình...")
        lora_model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
        lora_exact, lora_char = evaluate_model(lora_model, processor, crops, "Fine-Tuned Qwen2-VL-2B (QLoRA)")

        diff_exact = lora_exact - base_exact
        diff_char = lora_char - base_char
        print("\n📈 BẢNG TỔNG KẾT SO SÁNH:")
        print(f"  • Exact Match:  {base_exact:.1f}% ➡️ {lora_exact:.1f}% ({'+' if diff_exact >= 0 else ''}{diff_exact:.1f}%)")
        print(f"  • Char Accuracy: {base_char:.1f}% ➡️ {lora_char:.1f}% ({'+' if diff_char >= 0 else ''}{diff_char:.1f}%)")
    else:
        print(f"\n[i] Ghi chú: Chưa tìm thấy LoRA Adapter tại '{LORA_DIR}'.")
        print("    Hãy chạy: python scripts/finetune_qwen2_vl.py để huấn luyện Adapter!")

if __name__ == "__main__":
    main()
