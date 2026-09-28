"""Shared PyTest fixtures for ANPR Sentry."""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.connection import DatabaseManager


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated SQLite database instance for testing."""
    test_db_file = tmp_path / "test_anpr.db"
    db_mgr = DatabaseManager(f"sqlite:///{test_db_file}")
    yield db_mgr


@pytest.fixture
def dummy_frame():
    """Provides a synthetic 1080p frame for vision & overlay testing."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def dummy_crop():
    """Provides a synthetic 60x180 RGB plate crop."""
    return np.zeros((60, 180, 3), dtype=np.uint8)
