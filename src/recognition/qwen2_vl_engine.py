"""
Qwen2-VL-2B-Instruct Vision-Language Model Engine for Zero-Shot & Fine-Tuned License Plate Recognition.
Processes raw RGB crops directly without destructive binary thresholding.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.recognition.postprocessor import PlatePostProcessor


class Qwen2VLEngine:
    """
    4-bit Quantized Qwen2-VL-2B-Instruct Engine.
    Consumes ~1.44 GB VRAM on NVIDIA RTX 3060.
    """

    DEFAULT_PROMPT = (
        "Read the vehicle license plate number shown in this cropped image. "
        "Output ONLY the alphanumeric plate characters with no extra spaces, punctuation, or explanation."
    )

    def __init__(
        self,
        model_id: str = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
        processor_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        lora_dir: Optional[str] = "Weight/qwen2_vl_lora_plate",
        prompt: Optional[str] = None,
        max_new_tokens: int = 15,
        device: str = "cuda:0"
    ):
        self.model_id = model_id
        self.processor_id = processor_id
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.max_new_tokens = max_new_tokens
        self.device = device

        print(f"[+] Loading Qwen2-VL 4-bit model from '{self.model_id}'...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            dtype=torch.float16
        )

        print(f"[+] Loading AutoProcessor from '{self.processor_id}'...")
        self.processor = AutoProcessor.from_pretrained(self.processor_id, use_fast=False)

        # Optional LoRA adapter loading
        if lora_dir and Path(lora_dir).exists():
            adapter_config = Path(lora_dir) / "adapter_config.json"
            if adapter_config.exists():
                print(f"[+] Loading fine-tuned LoRA adapter from '{lora_dir}'...")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, lora_dir)
                print("[✓] LoRA adapter successfully attached to Qwen2-VL!")

        vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"[✓] Qwen2-VL Engine ready! Active GPU VRAM: {vram_gb:.2f} GB")

    def _prepare_pil_image(self, image_input: Union[np.ndarray, Image.Image, str, Path]) -> Image.Image:
        """Converts BGR numpy array, image path, or PIL image into RGB PIL Image."""
        if isinstance(image_input, (str, Path)):
            return Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # OpenCV frames are BGR -> convert to RGB
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
            return Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    @torch.inference_mode()
    def recognize_plate(
        self,
        plate_crop: Union[np.ndarray, Image.Image, str, Path],
        vehicle_type: str = "car"
    ) -> Dict[str, Any]:
        """
        Infers the license plate text from a single raw cropped image.
        """
        pil_img = self._prepare_pil_image(plate_crop)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": self.prompt}
                ]
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Post-process and normalize
        result = PlatePostProcessor.process(raw_output)
        result["vehicle_type"] = vehicle_type
        return result

    @torch.inference_mode()
    def recognize_batch(
        self,
        crops: List[Union[np.ndarray, Image.Image]],
        vehicle_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Batches multiple plate crops for parallel forward inference.
        """
        if not crops:
            return []

        if vehicle_types is None:
            vehicle_types = ["car"] * len(crops)

        pil_images = [self._prepare_pil_image(c) for c in crops]
        prompts = []
        all_messages = []

        for img in pil_images:
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": self.prompt}
                    ]
                }
            ]
            all_messages.append(msg)
            prompts.append(self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

        img_inputs, vid_inputs = process_vision_info(all_messages)
        inputs = self.processor(
            text=prompts,
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        results = []
        for raw, v_type in zip(raw_outputs, vehicle_types):
            res = PlatePostProcessor.process(raw)
            res["vehicle_type"] = v_type
            results.append(res)

        return results
