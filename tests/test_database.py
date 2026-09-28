"""Unit tests for DatabaseManager (SQLite, SQLAlchemy ORM, Watchlists & Detections)."""

import pytest
from src.database.models import VehicleDetection, Watchlist


def test_init_and_seed(temp_db):
    """Verifies tables are created and default mock watchlists are seeded."""
    with temp_db.get_session() as s:
        watchlists = s.query(Watchlist).all()
        assert len(watchlists) >= 2
        plates = [w.plate_number for w in watchlists]
        assert "SC56DYP" in plates
        assert "EY61NBG" in plates


def test_check_watchlist(temp_db):
    """Verifies watchlist matching against known stolen vehicles."""
    # Stolen car match
    is_match, reason, level = temp_db.check_watchlist("SC56DYP")
    assert is_match is True
    assert level == "CRITICAL"
    assert "mất cắp" in reason

    # Clean car
    is_match_clean, _, _ = temp_db.check_watchlist("AP05JEO")
    assert is_match_clean is False


def test_add_detection_normal(temp_db):
    """Verifies adding a standard vehicle detection."""
    rec = temp_db.add_detection(
        car_id=1,
        plate_number="AP05JEO",
        frame_number=100,
        raw_vlm_text="AP05JEO",
        confidence_score=0.98,
        vehicle_type="car"
    )
    assert rec.id is not None
    assert rec.plate_number == "AP05JEO"
    assert rec.is_watchlist_match is False


def test_add_detection_watchlist_hit(temp_db):
    """Verifies watchlist match flag and reason are automatically populated."""
    rec = temp_db.add_detection(
        car_id=2,
        plate_number="SC56DYP",
        frame_number=150,
        raw_vlm_text="SC56DYP",
        confidence_score=0.99,
        vehicle_type="car"
    )
    assert rec.is_watchlist_match is True
    assert "mất cắp" in rec.watchlist_reason


def test_query_detections(temp_db):
    """Verifies querying detections from session."""
    temp_db.add_detection(car_id=10, plate_number="NA13NRU", frame_number=50)
    temp_db.add_detection(car_id=11, plate_number="LM13VCV", frame_number=60)

    with temp_db.get_session() as s:
        records = s.query(VehicleDetection).order_by(VehicleDetection.frame_number.asc()).all()
        assert len(records) >= 2
        assert records[0].plate_number == "NA13NRU"
        assert records[1].plate_number == "LM13VCV"
