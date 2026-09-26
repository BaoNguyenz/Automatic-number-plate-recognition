"""
Fine-tuning Qwen2-VL-2B-Instruct using QLoRA for License Plate Recognition.
Trains LoRA adapters (r=16, alpha=32) on 4-bit quantized base model.
Saves lightweight (~50MB) adapter weights to Weight/qwen2_vl_lora_plate/.

Usage:
    conda activate torch
    python scripts/finetune_qwen2_vl.py
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_ID = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"
PROCESSOR_ID = "Qwen/Qwen2-VL-2B-Instruct"
DATA_DIR = ROOT / "data" / "finetune_dataset"
OUTPUT_DIR = ROOT / "Weight" / "qwen2_vl_lora_plate"

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlateVLMDataset(Dataset):
    def __init__(self, json_file: Path, data_dir: Path):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_rel_path = item["image"]
        img_path = self.data_dir / img_rel_path
        pil_img = Image.open(str(img_path)).convert("RGB")

        user_text = item["conversations"][0]["value"].replace("<image>\n", "")
        target_plate = item["conversations"][1]["value"].strip().upper()

        return {
            "image": pil_img,
            "prompt": user_text,
            "target": target_plate
        }


def collate_fn(batch, processor):
    """Formats batch into model input tensors with label masking."""
    all_messages = []
    prompts = []

    for item in batch:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": item["prompt"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["target"]}
                ]
            }
        ]
        all_messages.append(messages)
        prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

    img_inputs, vid_inputs = process_vision_info(all_messages)
    inputs = processor(
        text=prompts,
        images=img_inputs,
        videos=vid_inputs,
        padding=True,
        return_tensors="pt"
    )

    # Prepare labels for causal LM loss computation:
    # Mask prompt/user tokens with -100 so loss is calculated only on the target license plate characters
    labels = inputs.input_ids.clone()
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    labels[labels == pad_token_id] = -100

    # Mask user query tokens (everything up to `<|im_start|>assistant\n`)
    assistant_marker_ids = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    for i in range(labels.shape[0]):
        row = labels[i].tolist()
        # Find assistant start position
        found = False
        for pos in range(len(row) - len(assistant_marker_ids)):
            if row[pos:pos + len(assistant_marker_ids)] == assistant_marker_ids:
                labels[i, :pos + len(assistant_marker_ids)] = -100
                found = True
                break
        if not found:
            # Fallback: if marker not found, compute loss on the last 15 tokens
            labels[i, :-15] = -100

    inputs["labels"] = labels
    return inputs


def train():
    print("=" * 80)
    print("🚀 QWEN2-VL-2B QLoRA FINE-TUNING PIPELINE")
    print("=" * 80)
    print(f"Base Model:       {MODEL_ID}")
    print(f"Target Hardware:  NVIDIA GeForce RTX 3060 (12GB VRAM)")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Hyperparameters:  Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, LR={LEARNING_RATE}")
    print("=" * 80)

    train_json = DATA_DIR / "train.json"
    if not train_json.exists():
        print(f"[-] Error: Training data not found at '{train_json}'. Run 'scripts/prepare_finetune_data.py' first.")
        return

    # 1. Load Processor
    print(f"\n[+] [1/5] Loading AutoProcessor from '{PROCESSOR_ID}'...")
    processor = AutoProcessor.from_pretrained(PROCESSOR_ID, use_fast=False)

    # 2. Load 4-bit Base Model
    print(f"[+] [2/5] Loading 4-bit Qwen2-VL Base Model from '{MODEL_ID}'...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16
    )

    # 3. Setup QLoRA Configuration
    print("[+] [3/5] Attaching LoRA Adapters (r=16, alpha=32)...")
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Prepare DataLoader
    print("[+] [4/5] Preparing Dataset and DataLoader...")
    train_dataset = PlateVLMDataset(train_json, DATA_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor)
    )
    print(f"    Total training samples: {len(train_dataset)} ({len(train_loader)} batches per epoch)")

    # 5. Training Loop
    print("\n[+] [5/5] Starting QLoRA Fine-tuning...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    model.train()
    step_count = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 5 == 0 or step_count == 1:
                    current_loss = loss.item() * GRAD_ACCUM_STEPS
                    current_lr = lr_scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    print(f"  [Epoch {epoch}/{EPOCHS} | Step {step_count}/{total_steps}] Loss: {current_loss:.4f} | LR: {current_lr:.2e} | VRAM: {vram_gb:.2f} GB | Elapsed: {elapsed:.1f}s")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"--> [Epoch {epoch}/{EPOCHS} Complete] Average Loss: {avg_epoch_loss:.4f}\n")

    # 6. Save LoRA Adapter
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Saving fine-tuned LoRA adapter to '{OUTPUT_DIR}'...")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    total_time = time.time() - start_time
    print("=" * 80)
    print(f"🎉 TRAINING COMPLETE in {total_time:.1f} seconds (~{total_time/60:.1f} minutes)!")
    print(f"[✓] LoRA weights saved at: {OUTPUT_DIR}")
    print(f"[✓] Ready for evaluation using: python scripts/evaluate_finetune.py")
    print("=" * 80)

if __name__ == "__main__":
    train()
