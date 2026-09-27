"""
Post-processing and Text Normalization Module for License Plate Recognition.
Cleans raw VLM text outputs and enforces formatting rules.
"""

import re
from typing import Dict, Any



class PlatePostProcessor:
    """
    Normalizes raw text output from VLM / OCR into standardized license plate strings.
    """

    # Common prefixes LLMs might output despite strict prompts
    PREFIX_PATTERNS = [
        r"^THE\s+(LICENSE\s+)?PLATE\s+(NUMBER\s+)?(IS\s*)?[:\-\s]*",
        r"^LICENSE\s+PLATE[:\-\s]*",
        r"^PLATE[:\-\s]*",
        r"^NUMBER[:\-\s]*",
        r"^VEHICLE\s+PLATE[:\-\s]*",
        r"^RESULT[:\-\s]*",
    ]

    # UK standard plate regex: 2 letters, 2 digits, 3 letters (e.g. SC56DYP, EY61NBG, NA13NRU)
    UK_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")

    # Generic alphanumeric plate pattern (between 4 and 10 characters)
    GENERIC_PLATE_REGEX = re.compile(r"^[A-Z0-9]{4,10}$")

    @classmethod
    def clean_text(cls, raw_text: str) -> str:
        """
        Removes markdown formatting, explanatory prefixes, punctuation, and non-alphanumeric noise.
        """
        if not raw_text:
            return ""

        text = raw_text.strip().upper()

        # Remove markdown bold/code blocks (e.g., `SC56DYP`, **SC56DYP**)
        text = text.replace("`", "").replace("*", "").replace('"', '').replace("'", "")

        # Remove common prefixes
        for pattern in cls.PREFIX_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove spaces, dashes, hyphens, and dots commonly found between plate sections
        text = re.sub(r"[\s\-\.\:\_]+", "", text)

        # Retain only standard uppercase English letters and digits
        cleaned = re.sub(r"[^A-Z0-9]", "", text)

        return cleaned

    @classmethod
    def validate_format(cls, plate_text: str) -> bool:
        """
        Validates whether the text complies with standard license plate patterns.
        """
        if not plate_text:
            return False

        if cls.UK_PLATE_REGEX.match(plate_text):
            return True

        return bool(cls.GENERIC_PLATE_REGEX.match(plate_text))

    @classmethod
    def process(cls, raw_text: str, default_confidence: float = 1.0) -> Dict[str, Any]:
        """
        Processes raw text and computes validation flags and confidence scores.
        """
        cleaned = cls.clean_text(raw_text)
        is_uk = bool(cls.UK_PLATE_REGEX.match(cleaned))
        is_valid = cls.validate_format(cleaned)

        # Confidence heuristic
        conf = default_confidence
        if is_uk:
            conf = min(1.0, conf * 1.05)
        elif not is_valid:
            conf = max(0.2, conf * 0.7)

        return {
            "plate_number": cleaned,
            "raw_text": raw_text.strip(),
            "is_valid_format": is_valid,
            "is_uk_format": is_uk,
            "confidence_score": round(conf, 4)
        }
