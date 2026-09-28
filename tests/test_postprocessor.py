"""Unit tests for PlatePostProcessor (Regex cleaning, format compliance, LLM parsing)."""

import pytest
from src.recognition.postprocessor import PlatePostProcessor


def test_clean_standard_uk_plate():
    text = PlatePostProcessor.clean_text("SC56 DYP")
    assert text == "SC56DYP"
    assert PlatePostProcessor.validate_format(text) is True


def test_clean_dirty_llm_output():
    raw_texts = [
        ("The license plate is: SC56 DYP.", "SC56DYP"),
        ("`EY61 NBG`", "EY61NBG"),
        ("Plate: NA13-NRU", "NA13NRU"),
        ("Result: NG65 ZFX\n", "NG65ZFX"),
        ("**AP05 JEO**", "AP05JEO"),
    ]
    for raw, expected in raw_texts:
        res = PlatePostProcessor.process(raw)
        assert res["plate_number"] == expected
        assert res["is_valid_format"] is True


def test_uk_format_compliance():
    # Valid UK formats: 2 Letters + 2 Digits + 3 Letters
    assert PlatePostProcessor.validate_format("SC56DYP") is True
    assert PlatePostProcessor.process("SC56DYP")["is_uk_format"] is True

    assert PlatePostProcessor.validate_format("AP05JEO") is True
    assert PlatePostProcessor.process("AP05JEO")["is_uk_format"] is True

    # Generic valid plates (alphanumeric 4-10 chars, but not UK format)
    generic = PlatePostProcessor.process("ABC1234")
    assert generic["is_valid_format"] is True
    assert generic["is_uk_format"] is False

    # Invalid formats (too short, too long, or non-alphanumeric)
    assert PlatePostProcessor.validate_format("AB") is False
    assert PlatePostProcessor.validate_format("123") is False
    assert PlatePostProcessor.validate_format("VERYLONGLICENSEPLATENUMBER") is False
    assert PlatePostProcessor.validate_format("SC56!DYP") is False


def test_empty_input():
    res = PlatePostProcessor.process("")
    assert res["plate_number"] == ""
    assert res["is_valid_format"] is False
