from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class VehicleDetection(Base):
    __tablename__ = "vehicle_detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, nullable=False, index=True)
    plate_number = Column(String(20), nullable=False, index=True)
    raw_vlm_text = Column(String(50), nullable=True)
    confidence_score = Column(Float, default=1.0)
    vehicle_type = Column(String(20), default="car")
    vehicle_color = Column(String(30), nullable=True)
    frame_number = Column(Integer, nullable=False)
    plate_crop_path = Column(String(255), nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)
    is_watchlist_match = Column(Boolean, default=False, index=True)
    watchlist_reason = Column(String(255), nullable=True)

    def __repr__(self):
        return f"<VehicleDetection(id={self.id}, car_id={self.car_id}, plate='{self.plate_number}', alert={self.is_watchlist_match})>"


class Watchlist(Base):
    __tablename__ = "watchlists"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plate_number = Column(String(20), unique=True, nullable=False, index=True)
    vehicle_owner = Column(String(100), nullable=True)
    reason = Column(String(255), nullable=False)
    alert_level = Column(String(20), default="WARNING")  # INFO, WARNING, CRITICAL
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Watchlist(plate='{self.plate_number}', level='{self.alert_level}', reason='{self.reason}')>"
