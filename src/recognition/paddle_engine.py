"""
PaddleOCR Engine for Real-Time License Plate Character Recognition.
Provides a unified interface compatible with Qwen2VLEngine and integrates with PlatePostProcessor.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

# 1. Pre-import torch on Windows to avoid DLL conflicts with paddle/shm.dll
try:
    import torch
except ImportError:
    pass

# 2. Configure Paddle environment flags (disable MKLDNN bug on Windows CPU, bypass host check)
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
from PIL import Image

from src.recognition.postprocessor import PlatePostProcessor


class PaddleOCREngine:
    """
    PaddleOCR-based Engine for ultra-fast, lightweight License Plate Recognition.
    Processes crops in single-digit milliseconds per plate.
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = False,
        use_gpu: bool = False,
        show_log: bool = False
    ):
        """
        Initializes the PaddleOCR reader instance.
        """
        self.lang = lang
        self.use_gpu = use_gpu

        print(f"[+] Initializing PaddleOCR Engine (lang='{lang}', gpu={use_gpu})...")
        from paddleocr import PaddleOCR
        try:
            self.ocr = PaddleOCR(lang=self.lang, enable_mkldnn=False)
            print("[✓] PaddleOCR Engine initialized successfully!")
        except Exception:
            self.ocr = PaddleOCR(lang=self.lang)
            print("[✓] PaddleOCR Engine initialized!")

    def _prepare_cv2_image(self, image_input: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """
        Converts PIL Image, filepath, or numpy array to BGR OpenCV image for PaddleOCR.
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Failed to read image from path: {image_input}")
            return img
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def recognize_plate(
        self,
        plate_crop: Union[np.ndarray, Image.Image, str, Path],
        vehicle_type: str = "car"
    ) -> Dict[str, Any]:
        """
        Infers the license plate text from a single cropped plate image.

        Returns:
            Dict containing:
                - plate_number: cleaned & normalized plate string
                - raw_text: raw joined text from PaddleOCR
                - is_valid_format: bool
                - is_uk_format: bool
                - confidence_score: float confidence
                - vehicle_type: vehicle category
        """
        cv_img = self._prepare_cv2_image(plate_crop)

        detected_lines: List[str] = []
        confidences: List[float] = []

        try:
            pred_res = self.ocr.predict(cv_img)
            if pred_res:
                first = pred_res[0]
                if isinstance(first, dict):
                    # Modern PaddleX 3.x / PaddleOCR 3.x dict format
                    texts = first.get("rec_texts", [])
                    scores = first.get("rec_scores", [])
                    for t, s in zip(texts, scores):
                        detected_lines.append(str(t))
                        confidences.append(float(s))
                elif isinstance(first, list):
                    # Legacy PaddleOCR format: [[box, (text, conf)], ...]
                    for item in first:
                        if item and len(item) >= 2:
                            txt_info = item[1]
                            detected_lines.append(str(txt_info[0]))
                            confidences.append(float(txt_info[1]))
        except Exception as e:
            detected_lines = []
            confidences = []

        if not detected_lines:
            return {
                "plate_number": "",
                "raw_text": "",
                "is_valid_format": False,
                "is_uk_format": False,
                "confidence_score": 0.0,
                "vehicle_type": vehicle_type
            }

        # Join lines (handles plates split into 2 bounding boxes, e.g. 'SC56' + 'DYP')
        raw_text = "".join(detected_lines)
        avg_conf = float(np.mean(confidences)) if confidences else 0.5

        # Process through normalization pipeline
        result = PlatePostProcessor.process(raw_text, default_confidence=avg_conf)
        result["vehicle_type"] = vehicle_type
        return result

    def recognize_batch(
        self,
        crops: List[Union[np.ndarray, Image.Image, str, Path]],
        vehicle_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes a batch of plate crops sequentially.
        """
        if vehicle_types is None:
            vehicle_types = ["car"] * len(crops)

        results = []
        for crop, v_type in zip(crops, vehicle_types):
            res = self.recognize_plate(crop, vehicle_type=v_type)
            results.append(res)
        return results
