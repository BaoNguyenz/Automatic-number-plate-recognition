import os
import yaml
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base, VehicleDetection, Watchlist


class DatabaseManager:
    def __init__(self, db_url: str = None, config_path: str = "config/config.yaml"):
        if db_url is None:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    db_url = cfg.get("database", {}).get("url", "sqlite:///data/anpr.db")
            else:
                db_url = "sqlite:///data/anpr.db"

        # Đảm bảo thư mục cho SQLite tồn tại nếu dùng SQLite
        if db_url.startswith("sqlite:///"):
            db_path = db_url.replace("sqlite:///", "")
            dir_name = os.path.dirname(db_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

        self.db_url = db_url
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False} if self.db_url.startswith("sqlite") else {}
        )
        self.SessionFactory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False
        )
        self.init_db()

    def init_db(self):
        """Khởi tạo toàn bộ bảng trong cơ sở dữ liệu nếu chưa tồn tại."""
        Base.metadata.create_all(bind=self.engine)
        self.seed_mock_watchlists()

    @contextmanager
    def get_session(self) -> Session:
        """Context manager cung cấp session an toàn, tự động commit và rollback nếu lỗi."""
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def seed_mock_watchlists(self):
        """Tự động nạp 2 biển số giả lập trong video để kiểm thử tính năng Watchlist Alert."""
        mock_data = [
            {
                "plate_number": "SC56DYP",
                "vehicle_owner": "Nguyễn Văn A",
                "reason": "Phương tiện bị báo mất cắp (Mock Test Case)",
                "alert_level": "CRITICAL"
            },
            {
                "plate_number": "EY61NBG",
                "vehicle_owner": "Trần Thị B",
                "reason": "Phương tiện trốn đóng phí giao thông (Mock Test Case)",
                "alert_level": "WARNING"
            }
        ]

        with self.get_session() as session:
            for item in mock_data:
                existing = session.query(Watchlist).filter_by(plate_number=item["plate_number"]).first()
                if not existing:
                    watchlist_entry = Watchlist(**item)
                    session.add(watchlist_entry)

    def check_watchlist(self, plate_number: str) -> tuple[bool, str, str]:
        """Kiểm tra biển số có nằm trong danh sách đen không. Trả về (is_match, reason, alert_level)."""
        clean_plate = plate_number.replace(" ", "").replace("-", "").upper()
        with self.get_session() as session:
            item = session.query(Watchlist).filter(
                (Watchlist.plate_number == clean_plate) | 
                (Watchlist.plate_number == plate_number)
            ).first()
            if item:
                return True, item.reason, item.alert_level
            return False, "", ""

    def add_detection(
        self,
        car_id: int,
        plate_number: str,
        frame_number: int,
        raw_vlm_text: str = None,
        confidence_score: float = 1.0,
        vehicle_type: str = "car",
        vehicle_color: str = None,
        plate_crop_path: str = None
    ) -> VehicleDetection:
        """Lưu một lượt phát hiện xe vào database, tự động kiểm tra Watchlist và kích hoạt cờ cảnh báo."""
        is_match, reason, level = self.check_watchlist(plate_number)

        record = VehicleDetection(
            car_id=car_id,
            plate_number=plate_number,
            raw_vlm_text=raw_vlm_text,
            confidence_score=confidence_score,
            vehicle_type=vehicle_type,
            vehicle_color=vehicle_color,
            frame_number=frame_number,
            plate_crop_path=plate_crop_path,
            is_watchlist_match=is_match,
            watchlist_reason=reason if is_match else None
        )

        with self.get_session() as session:
            session.add(record)
            session.flush()
            session.refresh(record)

            if is_match:
                print("\n" + "!" * 80)
                print(f"🚨 [WATCHLIST ALERT - {level}] PHÁT HIỆN XE TRONG DANH SÁCH THEO DÕI!")
                print(f"   ► Biển số: {plate_number}")
                print(f"   ► Xe ID:   {car_id} (Frame: {frame_number})")
                print(f"   ► Lý do:   {reason}")
                print("!" * 80 + "\n")

            return record


# Global singleton instance
_db_instance = None

def get_db(config_path="config/config.yaml") -> DatabaseManager:
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager(config_path=config_path)
    return _db_instance
