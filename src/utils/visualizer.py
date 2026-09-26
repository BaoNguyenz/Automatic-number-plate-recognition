"""
High-End Video Visualizer and HUD Overlay Module for ANPR System.
Draws stylized corner-accented bounding boxes, license plate badges, and real-time Watchlist alert banners.
"""

from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


class ANPRVisualizer:
    """
    Renders modern, production-grade visual overlays for ANPR traffic analysis.
    Supports dynamic resolution scaling (from 720p to 4K UHD).
    """

    # Color Palette (BGR)
    COLOR_NORMAL = (46, 204, 113)     # Emerald Green
    COLOR_PLATE_BOX = (0, 165, 255)   # Amber Orange
    COLOR_WARNING = (0, 140, 255)     # Deep Amber
    COLOR_CRITICAL = (30, 30, 235)    # Vivid Crimson Red
    COLOR_DARK_BG = (25, 25, 25)      # Dark Matte
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_UK_YELLOW = (0, 215, 255)   # Authentic UK rear plate yellow (BGR)

    def __init__(self, target_resolution: Optional[Tuple[int, int]] = None):
        self.target_resolution = target_resolution

    @classmethod
    def draw_corner_box(
        cls,
        img: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        thickness: int = 4,
        corner_length: int = 40
    ) -> None:
        """
        Draws an aesthetic tech-style corner bounding box with accent lines.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        c_len_x = min(corner_length, w // 3)
        c_len_y = min(corner_length, h // 3)

        # Draw semi-transparent bounding box outline
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, thickness // 2))

        # Top-Left corner
        cv2.line(img, (x1, y1), (x1 + c_len_x, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + c_len_y), color, thickness)

        # Top-Right corner
        cv2.line(img, (x2, y1), (x2 - c_len_x, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + c_len_y), color, thickness)

        # Bottom-Left corner
        cv2.line(img, (x1, y2), (x1 + c_len_x, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - c_len_y), color, thickness)

        # Bottom-Right corner
        cv2.line(img, (x2, y2), (x2 - c_len_x, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - c_len_y), color, thickness)

    def draw_vehicle_overlay(
        self,
        frame: np.ndarray,
        car_id: int,
        car_bbox: Tuple[float, float, float, float],
        plate_bbox: Optional[Tuple[float, float, float, float]] = None,
        plate_number: Optional[str] = None,
        vehicle_type: str = "car",
        is_watchlist: bool = False,
        alert_level: str = "NORMAL",
        watchlist_reason: Optional[str] = None,
        plate_crop: Optional[np.ndarray] = None
    ) -> None:
        """
        Renders complete information badges and banners for a single vehicle.
        """
        H, W = frame.shape[:2]
        scale = max(0.5, H / 1080.0)

        # Determine theme color based on security status
        if is_watchlist:
            color = self.COLOR_CRITICAL if alert_level.upper() == "CRITICAL" else self.COLOR_WARNING
        else:
            color = self.COLOR_NORMAL

        cx1, cy1, cx2, cy2 = [int(v) for v in car_bbox]
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(W, cx2), min(H, cy2)

        # 1. Draw corner bounding box around vehicle
        thickness = max(2, int(4 * scale))
        corner_len = int(35 * scale)
        self.draw_corner_box(frame, (cx1, cy1, cx2, cy2), color, thickness=thickness, corner_length=corner_len)

        # 2. Draw plate bounding box if visible
        if plate_bbox is not None:
            px1, py1, px2, py2 = [int(v) for v in plate_bbox]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(W, px2), min(H, py2)
            cv2.rectangle(frame, (px1, py1), (px2, py2), self.COLOR_PLATE_BOX, max(1, int(2 * scale)))

        # 3. Floating Vehicle Header Tag (above car)
        font_scale = 0.55 * scale
        font_thick = max(1, int(1.5 * scale))
        header_text = f"Car #{car_id} ({vehicle_type.upper()})"

        (t_w, t_h), baseline = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        badge_y2 = max(t_h + 10, cy1 - int(5 * scale))
        badge_y1 = max(0, badge_y2 - t_h - int(10 * scale))
        badge_x1 = cx1
        badge_x2 = min(W, cx1 + t_w + int(16 * scale))

        # Header background
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            frame,
            header_text,
            (badge_x1 + int(8 * scale), badge_y2 - int(5 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            self.COLOR_BLACK if color != self.COLOR_CRITICAL else self.COLOR_WHITE,
            font_thick,
            cv2.LINE_AA
        )

        # 4. Floating Plate & Alert Badge
        if plate_number and plate_number != "UNKNOWN":
            plate_scale = 0.75 * scale
            plate_thick = max(2, int(2.2 * scale))
            plate_text = f"  {plate_number}  "
            (p_w, p_h), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_DUPLEX, plate_scale, plate_thick)

            p_box_y2 = badge_y1 - int(4 * scale)
            p_box_y1 = max(0, p_box_y2 - p_h - int(12 * scale))
            p_box_x1 = cx1
            p_box_x2 = min(W, cx1 + p_w)

            # Draw authentic UK plate badge (Yellow rear plate aesthetic with border)
            cv2.rectangle(frame, (p_box_x1, p_box_y1), (p_box_x2, p_box_y2), self.COLOR_UK_YELLOW, -1)
            cv2.rectangle(frame, (p_box_x1, p_box_y1), (p_box_x2, p_box_y2), self.COLOR_BLACK, max(1, int(2 * scale)))
            cv2.putText(
                frame,
                plate_text,
                (p_box_x1, p_box_y2 - int(6 * scale)),
                cv2.FONT_HERSHEY_DUPLEX,
                plate_scale,
                self.COLOR_BLACK,
                plate_thick,
                cv2.LINE_AA
            )

            # 5. Security Alert Badge (if Watchlist matched)
            if is_watchlist:
                alert_text = f" [!] {alert_level}: {watchlist_reason or 'FLAGGED VEHICLE'} "
                alert_scale = 0.55 * scale
                alert_thick = max(1, int(1.5 * scale))
                (a_w, a_h), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, alert_scale, alert_thick)

                a_box_y2 = p_box_y1 - int(4 * scale)
                a_box_y1 = max(0, a_box_y2 - a_h - int(10 * scale))
                a_box_x1 = cx1
                a_box_x2 = min(W, cx1 + a_w + int(10 * scale))

                # Red pulsating or glowing alert banner
                cv2.rectangle(frame, (a_box_x1, a_box_y1), (a_box_x2, a_box_y2), self.COLOR_CRITICAL, -1)
                cv2.putText(
                    frame,
                    alert_text,
                    (a_box_x1 + int(5 * scale), a_box_y2 - int(5 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    alert_scale,
                    self.COLOR_WHITE,
                    alert_thick,
                    cv2.LINE_AA
                )

    def draw_top_hud(
        self,
        frame: np.ndarray,
        active_alerts: List[Dict[str, Any]],
        fps: float = 30.0,
        frame_idx: int = 0
    ) -> None:
        """
        Draws a Command-Center style Top HUD banner across the video screen.
        """
        H, W = frame.shape[:2]
        scale = max(0.5, H / 1080.0)

        hud_h = int(50 * scale)
        # Semi-transparent dark header bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, hud_h), self.COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # System status title
        title_text = f"ANPR AI SYSTEM  |  QWEN2-VL & TENSORRT  |  FRAME: {frame_idx:04d}  |  FPS: {fps:.1f}"
        font_scale = 0.5 * scale
        cv2.putText(
            frame,
            title_text,
            (int(20 * scale), int(32 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            self.COLOR_WHITE,
            max(1, int(1.2 * scale)),
            cv2.LINE_AA
        )

        # Global Watchlist Alert Flash Ticker
        if len(active_alerts) > 0:
            alert = active_alerts[0]
            alert_msg = f"🚨 ALERT: {alert.get('plate_number')} [{alert.get('alert_level')}] - {alert.get('reason')}"
            (msg_w, _), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            alert_x = max(int(W * 0.45), W - msg_w - int(20 * scale))
            cv2.rectangle(frame, (alert_x - 10, 5), (W - 10, hud_h - 5), self.COLOR_CRITICAL, -1)
            cv2.putText(
                frame,
                alert_msg,
                (alert_x, int(32 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                self.COLOR_WHITE,
                max(1, int(1.5 * scale)),
                cv2.LINE_AA
            )
