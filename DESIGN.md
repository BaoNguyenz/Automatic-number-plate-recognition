---
name: ANPR Sentry Dark
colors:
  primary: "#06b6d4"          # Electric Cyan (Telemetry & Active UI)
  primary-hover: "#0891b2"
  secondary: "#3b82f6"        # Deep Tech Blue
  surface: "#0b0f19"          # Obsidian Background
  surface-card: "#111827"     # Slate Card Surface
  surface-glass: "rgba(17, 24, 39, 0.75)"
  border: "#1f2937"           # Subtle Card Border
  border-accent: "#374151"
  on-surface: "#f3f4f6"       # Crisp White Text
  on-surface-muted: "#9ca3af" # Slate Gray Caption Text
  success: "#10b981"          # Emerald Green (Compliant & Match)
  warning: "#f59e0b"          # Amber (Toll Violator)
  error: "#ef4444"            # Crimson Red (Critical Stolen Vehicle)
  plate-yellow: "#fcd34d"     # UK Standard Rear Plate Yellow
  plate-white: "#ffffff"      # UK Standard Front Plate White
typography:
  fontFamily-sans: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
  fontFamily-mono: "'JetBrains Mono', 'Fira Code', monospace"
  heading-xl:
    fontSize: "28px"
    fontWeight: 700
    lineHeight: "36px"
  heading-md:
    fontSize: "20px"
    fontWeight: 600
    lineHeight: "28px"
  body-md:
    fontSize: "15px"
    fontWeight: 400
    lineHeight: "22px"
  plate-display:
    fontFamily: "'JetBrains Mono', monospace"
    fontSize: "24px"
    fontWeight: 800
    letterSpacing: "3px"
rounded:
  sm: "6px"
  md: "10px"
  lg: "16px"
  pill: "9999px"
shadows:
  glow-cyan: "0 0 20px rgba(6, 182, 212, 0.35)"
  glow-red: "0 0 25px rgba(239, 68, 68, 0.45)"
  card: "0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5)"
---

# Design System: ANPR Sentry AI

## Overview
A hyper-modern, mission-critical Surveillance & License Plate Recognition Command Center.
The aesthetic combines sleek dark obsidian glassmorphism with high-contrast tactical telemetry, reminiscent of next-generation defense and traffic security hubs.

## Color Rationale
- **Surface (`#0b0f19`) & Surface-Card (`#111827`)**: Deep obsidian-slate backdrop that eliminates eye fatigue during continuous surveillance monitoring.
- **Primary Cyan (`#06b6d4`)**: Represents active AI inference, bounding box highlights, and real-time vision pipelines.
- **Success Emerald (`#10b981`)**: Indicates valid UK/EU plate formatting and clear vehicles.
- **Warning Amber (`#f59e0b`)**: Used for moderate security alerts such as toll fee evasion.
- **Error Crimson (`#ef4444`)**: High-priority alert banner for stolen vehicles (CRITICAL watchlist hits), with pulsing glow animations.
- **Plate Display Colors (`#fcd34d` and `#ffffff`)**: Authentic UK rear yellow and front white plate badge representations.

## Typography
- **Headlines & Interface**: Use `Inter` or modern sans-serif for clean legibility.
- **License Plate Numbers & Latencies**: Use `JetBrains Mono` or high-contrast monospace font to ensure zero character ambiguity (`0` vs `O`, `1` vs `I`).
- **Telemetry Tags**: Uppercase, medium weight, 11–13px with tracking for technical status badges.

## Component Patterns

### 1. Dual-Engine Switcher Pill
- Toggle container with subtle background `#1f2937` and rounded pill shape.
- Active state lights up with Cyan glow for PaddleOCR or Purple/Orange glow for Qwen2-VL QLoRA.

### 2. Video & Image Upload Workspace (File-Based Processing)
- Designed specifically for **Video Upload (`.mp4`, `.mov`) and Single-Image Analysis**, requiring no external camera hardware.
- Prominent **Drag & Drop Upload Zone**:
  * Drop traffic video file or select preset video (`2103099-uhd_3840_2160_30fps.mp4`).
  * Instant single-plate image test tab (for rapid sub-second verification).
- **Processing & Playback Canvas**:
  * Displays processing progress bar (e.g. `Processing Frame 450/1800 (25%) | 18 FPS`).
  * Once completed, seamless interactive video player plays the annotated output (`out.mp4`) with corner-accent bounding boxes and security banners.
  * Clicking any detected vehicle in the results table automatically seeks the video to that exact timestamp.


### 3. Hero License Plate Card
- Embossed realistic badge design matching genuine UK vehicular standards (Black embossed lettering on reflective yellow `#fcd34d` or white background with blue UK/EU identifier).
- Real-time telemetry underneath: Engine Type, Latency (ms), and Confidence Score (e.g. `99.8%`).

### 4. Security Alert HUD
- Pulsing red neon border with warning iconography when a vehicle matches the database watchlist.
- Prominent action buttons: `[ Dispatch Patrol ]` (solid crimson) and `[ Acknowledge ]` (ghost border).

### 5. Glassmorphic Data Table
- High-density audit log showing detected vehicles with thumbnail WebP crops.
- Alternating subtle rows with hover elevation and pill status badges.

## Do's and Don'ts
- **DO** use monospace typography for every license plate string.
- **DO** maintain high contrast (at least 7:1) for all critical alert texts.
- **DON'T** use generic bright purple or rainbow gradients that make the dashboard look like a consumer app.
- **DON'T** obscure the live video stream with heavy opaque dialogs; use semi-transparent glass overlays with `backdrop-filter: blur(12px)`.
