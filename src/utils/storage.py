"""
Storage Utility Module for License Plate Image Crops.
Saves crops using highly compressed WebP format organized by date.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
from PIL import Image


class PlateStorageManager:
    """
    Manages saving and loading of license plate crops in WebP format.
    Provides ~70% disk space reduction compared to standard PNG.
    """

    def __init__(self, base_dir: str = "media/plates", webp_quality: int = 85):
        self.base_dir = Path(base_dir)
        self.webp_quality = webp_quality
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_crop(
        self,
        crop: Union[np.ndarray, Image.Image],
        car_id: int,
        plate_number: str,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Saves a plate crop image to disk in WebP format.
        Path format: media/plates/YYYY/MM/DD/car_{car_id}_{plate_number}_{timestamp}.webp
        Returns the relative path string for database storage.
        """
        now = timestamp or datetime.utcnow()
        date_folder = self.base_dir / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d")
        date_folder.mkdir(parents=True, exist_ok=True)

        time_str = now.strftime("%H%M%S_%f")[:10]
        safe_plate = "".join(c for c in plate_number if c.isalnum()) or "UNKNOWN"
        filename = f"car_{car_id}_{safe_plate}_{time_str}.webp"
        target_path = date_folder / filename

        if isinstance(crop, Image.Image):
            crop = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)

        # Save as WebP with specified quality
        success = cv2.imwrite(
            str(target_path),
            crop,
            [cv2.IMWRITE_WEBP_QUALITY, self.webp_quality]
        )

        if not success:
            raise IOError(f"Failed to encode and save crop to '{target_path}'")

        # Return standardized relative path using forward slashes
        return target_path.as_posix()

    def load_crop(self, rel_path: str) -> Optional[np.ndarray]:
        """Loads a stored WebP plate crop from disk."""
        path = Path(rel_path)
        if not path.exists():
            return None
        return cv2.imread(str(path))
