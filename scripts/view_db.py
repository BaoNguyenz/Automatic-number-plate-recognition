"""
Utility script to quickly inspect SQLite database (data/anpr.db) in Terminal.
Usage:
    python scripts/view_db.py
"""

import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Set UTF-8 encoding for Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = Path("data/anpr.db")

def view_database():
    if not DB_PATH.exists():
        print(f"[-] Cơ sở dữ liệu '{DB_PATH}' chưa tồn tại.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # Configure pandas display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.unicode.east_asian_width', True)

    print("=" * 80)
    print("🚗 [1] BẢNG: WATCHLISTS (Danh sách xe theo dõi / Cảnh báo)")
    print("=" * 80)
    df_watch = pd.read_sql_query("SELECT id, plate_number, vehicle_owner, reason, alert_level, created_at FROM watchlists", conn)
    if df_watch.empty:
        print("(Trống)")
    else:
        print(df_watch.to_string(index=False))

    print("\n" + "=" * 80)
    print("📸 [2] BẢNG: VEHICLE_DETECTIONS (Lịch sử xe & biển số đã nhận diện)")
    print("=" * 80)
    df_det = pd.read_sql_query(
        "SELECT id, car_id, plate_number, confidence_score, vehicle_type, vehicle_color, frame_number, is_watchlist_match, watchlist_reason, detected_at "
        "FROM vehicle_detections ORDER BY id DESC LIMIT 20", 
        conn
    )
    if df_det.empty:
        print("(Chưa có bản ghi nhận diện nào)")
    else:
        print(df_det.to_string(index=False))
        
    print("=" * 80)
    conn.close()

if __name__ == "__main__":
    view_database()
