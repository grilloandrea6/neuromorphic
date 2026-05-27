#!/usr/bin/env python3
"""Overlay the Loihi/Crazyflie box-demo geometry on a recorded GoPro video.

Run from the repo root:

    source ~/venv/bin/activate
    python video_overlay_loihi_demo.py \
        --video video/GOPR0101.MP4 \
        --log ros2-crazyflie-mocap/ws/loihi_bridge_log_20260520_195350.csv \
        --calibration video/calibration_gopro0101.json \
        --preview-frame 15 \
        --preview-output video/GOPR0101_overlay_preview.jpg

Create a calibration template:

    python video_overlay_loihi_demo.py --write-calibration-template video/calibration_gopro0101.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_VIDEO = Path("video/GOPR0101.MP4")
DEFAULT_LOG = Path("ros2-crazyflie-mocap/ws/loihi_bridge_log_20260520_195350.csv")
DEFAULT_CALIBRATION = Path("video/calibration_gopro0101.json")
DEFAULT_OUTPUT = Path("video/GOPR0101_overlay.mp4")
DEFAULT_PREVIEW_OUTPUT = Path("video/GOPR0101_overlay_preview.jpg")

REFERENCE_RADIUS_M = 0.25
REFERENCE_CENTER_XY_M = np.asarray([0.35, -0.35], dtype=np.float64)
MAX_DISPLAY_BOX_HALF_EXTENT_M = 0.5

COLOR_CIRCLE = (185, 185, 185)
COLOR_BOX = (121, 78, 31)
COLOR_BOX_ACTIVE = (190, 92, 31)
COLOR_STATE = (72, 73, 209)
COLOR_COMMAND = (55, 137, 229)
COLOR_PREDICTED_TRAJ = (71, 201, 255)
COLOR_PREDICTED_TRAJ_START = (255, 120, 45)
COLOR_PREDICTED_TRAJ_END = (35, 220, 255)
COLOR_TEXT = (245, 245, 245)
COLOR_TEXT_BG = (30, 30, 30)
PANEL_MARGIN_PX = 18
PANEL_PAD_X_PX = 18
PANEL_PAD_Y_PX = 15
PANEL_LINE_HEIGHT_PX = 27
PANEL_FONT_SCALE = 0.62
PANEL_FONT_THICKNESS = 2
PANEL_ALPHA = 0.68


def video_time_to_demo_time(video_t: float, start_offset_s: float) -> float:
    """Map video seconds to demo seconds.

    start_offset_s is the video timestamp where the logged LOIHI_DEMO segment
    should be treated as t=0.
    """
    return float(video_t) - float(start_offset_s)


@dataclass(frozen=True)
class Calibration:
    homography: np.ndarray
    crop: tuple[int, int, int, int] | None
    start_offset_s: float = 0.0
    reference_marker_advance_s: float = 0.0


@dataclass(frozen=True)
class DemoData:
    t: np.ndarray
    state_xy: np.ndarray
    command_xy: np.ndarray
    predicted_xy: np.ndarray
    reference_xy: np.ndarray
    ref_next_xy: np.ndarray
    box_limits: np.ndarray
    power_total_w: np.ndarray
    power_age_s: np.ndarray
    control_period_s: float


@dataclass(frozen=True)
class OutputCrop:
    rect: tuple[int, int, int, int] | None
    calibration: Calibration
    size: tuple[int, int]


def finite_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw == "":
        return math.nan
    return float(raw)


def load_demo_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    start = next((idx for idx, row in enumerate(rows) if row.get("mode") == "LOIHI_DEMO"), None)
    if start is None:
        raise ValueError(f"No LOIHI_DEMO segment found in {path}")
    end = len(rows)
    for idx in range(start + 1, len(rows)):
        if rows[idx].get("mode") != "LOIHI_DEMO":
            end = idx
            break
    return rows[start:end]


def array_from_rows(rows: list[dict[str, str]], keys: Iterable[str]) -> np.ndarray:
    return np.asarray([[finite_float(row, key) for key in keys] for row in rows], dtype=np.float64)


def load_demo_data(path: Path) -> DemoData:
    rows = load_demo_rows(path)
    t0 = finite_float(rows[0], "monotonic_s")
    t = np.asarray([finite_float(row, "monotonic_s") - t0 for row in rows], dtype=np.float64)
    finite_t = t[np.isfinite(t)]
    if finite_t.shape[0] >= 2:
        control_period_s = float(np.median(np.diff(finite_t)))
    else:
        control_period_s = 0.01
    state_xy = array_from_rows(rows, ("state_x", "state_y"))
    command_xy = array_from_rows(rows, ("cmd_x", "cmd_y"))
    predicted_columns = [
        (f"loihi_xy_traj_{idx:02d}_x", f"loihi_xy_traj_{idx:02d}_y")
        for idx in range(1, 11)
    ]
    predicted_xy = np.asarray(
        [
            [
                [finite_float(row, x_key), finite_float(row, y_key)]
                for x_key, y_key in predicted_columns
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    reference_xy = array_from_rows(rows, ("ref_x", "ref_y"))
    ref_next_xy = array_from_rows(rows, ("ref_next_x", "ref_next_y"))
    bounds = array_from_rows(rows, ("bound_ex_min", "bound_ex_max", "bound_ey_min", "bound_ey_max"))

    world_x_min = ref_next_xy[:, 0] - bounds[:, 1]
    world_x_max = ref_next_xy[:, 0] - bounds[:, 0]
    world_y_min = ref_next_xy[:, 1] - bounds[:, 3]
    world_y_max = ref_next_xy[:, 1] - bounds[:, 2]
    box_limits = np.column_stack((world_x_min, world_x_max, world_y_min, world_y_max))
    power_total_w = array_from_rows(rows, ("power_total_w",)).reshape(-1)
    power_age_s = array_from_rows(rows, ("power_age_s",)).reshape(-1)

    return DemoData(
        t=t,
        state_xy=state_xy,
        command_xy=command_xy,
        predicted_xy=predicted_xy,
        reference_xy=reference_xy,
        ref_next_xy=ref_next_xy,
        box_limits=box_limits,
        power_total_w=power_total_w,
        power_age_s=power_age_s,
        control_period_s=control_period_s,
    )


def write_calibration_template(path: Path) -> None:
    template = {
        "description": "Map known floor/world XY points in meters to pixels in the original video frame.",
        "crop": None,
        "start_offset_s": 0.0,
        "reference_marker_advance_s": 0.0,
        "anchors": [
            {"world_xy_m": [-0.35, -1.05], "pixel_xy": [0, 0]},
            {"world_xy_m": [1.05, -1.05], "pixel_xy": [0, 0]},
            {"world_xy_m": [1.05, 0.35], "pixel_xy": [0, 0]},
            {"world_xy_m": [-0.35, 0.35], "pixel_xy": [0, 0]},
        ],
        "notes": [
            "Replace pixel_xy values with clicked points from the original 2704x1520 frame.",
            "Use at least four non-collinear anchors. Extra anchors are allowed.",
            "If crop is set, use [x, y, width, height] in original-frame pixels.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2) + "\n")
    print(f"Wrote calibration template to {path}")


def write_calibration_from_points(
    path: Path,
    image_points: np.ndarray,
    crop: tuple[int, int, int, int] | None = None,
    start_offset_s: float = 0.0,
    axis_angle_deg: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, image_points.shape[0], endpoint=False)
    world_unit = np.column_stack((np.cos(theta), np.sin(theta)))
    if flip_x:
        world_unit[:, 0] *= -1.0
    if flip_y:
        world_unit[:, 1] *= -1.0
    if axis_angle_deg:
        angle = math.radians(axis_angle_deg)
        rot = np.asarray(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=np.float64,
        )
        world_unit = world_unit @ rot.T
    world_points = REFERENCE_CENTER_XY_M + REFERENCE_RADIUS_M * world_unit
    payload = {
        "description": "Generated from clicked points on the projected reference circle.",
        "crop": list(crop) if crop is not None else None,
        "start_offset_s": float(start_offset_s),
        "reference_marker_advance_s": 0.0,
        "anchors": [
            {
                "world_xy_m": [round(float(world[0]), 6), round(float(world[1]), 6)],
                "pixel_xy": [round(float(pixel[0]), 3), round(float(pixel[1]), 3)],
            }
            for world, pixel in zip(world_points, image_points)
        ],
        "notes": [
            "Circle calibration assumes the camera view is top-down enough that the reference circle remains a circle.",
            "Default axis convention is world +x to image right and world +y to image up.",
            "If the preview is mirrored or rotated, regenerate with --circle-calibration-flip-x, --circle-calibration-flip-y, or --circle-calibration-axis-angle-deg.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote circle-derived calibration to {path}")


def load_calibration(path: Path) -> Calibration:
    raw = json.loads(path.read_text())
    start_offset_s = float(raw.get("start_offset_s", 0.0))
    reference_marker_advance_s = float(raw.get("reference_marker_advance_s", 0.0))
    anchors = raw.get("anchors", [])
    if len(anchors) < 4:
        raise ValueError(f"{path} must contain at least four anchors")
    world = np.asarray([anchor["world_xy_m"] for anchor in anchors], dtype=np.float32)
    pixel = np.asarray([anchor["pixel_xy"] for anchor in anchors], dtype=np.float32)
    if np.allclose(pixel, 0.0):
        raise ValueError(f"{path} still looks like an unedited template; replace pixel_xy values")

    homography, mask = cv2.findHomography(world, pixel, method=0)
    if homography is None or mask is None or int(mask.sum()) < 4:
        raise ValueError(f"Could not compute a valid homography from {path}")

    crop_raw = raw.get("crop")
    crop = None
    if crop_raw is not None:
        if len(crop_raw) != 4:
            raise ValueError("calibration crop must be [x, y, width, height]")
        crop = tuple(int(v) for v in crop_raw)
        x, y, _, _ = crop
        translate = np.asarray([[1.0, 0.0, -x], [0.0, 1.0, -y], [0.0, 0.0, 1.0]])
        homography = translate @ homography
    return Calibration(
        homography=homography.astype(np.float64),
        crop=crop,
        start_offset_s=start_offset_s,
        reference_marker_advance_s=reference_marker_advance_s,
    )


def make_motion_composite(
    video_path: Path,
    output_path: Path,
    stride: int,
    max_frames: int,
    start_s: float,
    end_s: float | None,
    method: str,
) -> np.ndarray:
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(round(start_s * fps)))
    end_frame = total_frames if end_s is None else min(total_frames, int(round(end_s * fps)))
    stride = max(1, int(stride))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    composite: np.ndarray | None = None
    base: np.ndarray | None = None
    frames_used = 0
    frame_idx = start_frame
    try:
        while frame_idx < end_frame and frames_used < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx - start_frame) % stride != 0:
                frame_idx += 1
                continue
            frame_f = frame.astype(np.float32)
            if composite is None:
                composite = frame_f.copy()
                base = frame_f.copy()
            elif method == "max":
                composite = np.maximum(composite, frame_f)
            elif method == "mean":
                composite += frame_f
            elif method == "diff":
                assert base is not None
                composite = np.maximum(composite, np.abs(frame_f - base))
            else:
                raise ValueError("composite method must be max, mean, or diff")
            frames_used += 1
            frame_idx += 1
    finally:
        cap.release()

    if composite is None or frames_used == 0:
        raise RuntimeError(f"No frames read from {video_path}")
    if method == "mean":
        composite /= float(frames_used)
    if method == "diff":
        assert base is not None
        composite = np.clip(base * 0.55 + composite * 2.8, 0.0, 255.0)
    image = np.clip(composite, 0.0, 255.0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write {output_path}")
    print(f"Wrote motion composite {output_path} from {frames_used} frames")
    return image


def fit_ordered_circle_points(clicked: list[tuple[int, int]], samples: int) -> np.ndarray:
    if len(clicked) < 3:
        raise ValueError("Need at least three clicked points to fit a circle")
    pts = np.asarray(clicked, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    radius_sq = c + cx * cx + cy * cy
    if radius_sq <= 0.0:
        raise ValueError("Clicked points did not produce a valid circle")
    radius = math.sqrt(float(radius_sq))
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    # Image v increases downward, while the default world convention uses +y up.
    return np.column_stack((cx + radius * np.cos(theta), cy - radius * np.sin(theta)))


def interactive_circle_calibration(args: argparse.Namespace) -> None:
    composite_start_s = 0.0 if args.video_start_s is None else float(args.video_start_s)
    composite = make_motion_composite(
        video_path=args.video,
        output_path=args.motion_composite_output,
        stride=args.composite_stride,
        max_frames=args.composite_max_frames,
        start_s=composite_start_s,
        end_s=args.video_end_s,
        method=args.composite_method,
    )

    display_scale = float(args.display_scale)
    if display_scale <= 0.0:
        display_scale = min(1.0, 1400.0 / max(composite.shape[0], composite.shape[1]))
    display = cv2.resize(composite, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
    clicked: list[tuple[int, int]] = []
    window = "click circle points: left=add, right/Backspace=undo, Enter=save, q=quit"

    def redraw() -> None:
        canvas = display.copy()
        for idx, point in enumerate(clicked):
            scaled = tuple(np.rint(np.asarray(point) * display_scale).astype(int))
            cv2.circle(canvas, scaled, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(idx + 1),
                (scaled[0] + 7, scaled[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        if len(clicked) >= 3:
            circle_points_fit = fit_ordered_circle_points(clicked, args.circle_calibration_samples)
            scaled_circle = np.rint(circle_points_fit * display_scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [scaled_circle], True, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{len(clicked)} points. Click circle; Enter saves after 3+ points.",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((int(round(x / display_scale)), int(round(y / display_scale))))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and clicked:
            clicked.pop()
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            break
        if key in (8, 127) and clicked:
            clicked.pop()
            redraw()
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise SystemExit("Circle calibration cancelled")
    cv2.destroyWindow(window)

    circle_points_fit = fit_ordered_circle_points(clicked, args.circle_calibration_samples)
    write_calibration_from_points(
        args.calibration,
        circle_points_fit,
        crop=None,
        start_offset_s=float(args.start_offset_s or 0.0),
        axis_angle_deg=float(args.circle_calibration_axis_angle_deg),
        flip_x=bool(args.circle_calibration_flip_x),
        flip_y=bool(args.circle_calibration_flip_y),
    )


def interp_vector(t: np.ndarray, values: np.ndarray, sample_t: float) -> np.ndarray:
    out = np.empty(values.shape[1], dtype=np.float64)
    for col in range(values.shape[1]):
        valid = np.isfinite(values[:, col])
        if valid.sum() < 2:
            out[col] = math.nan
        else:
            out[col] = np.interp(sample_t, t[valid], values[valid, col])
    return out


def nearest_row_index(t: np.ndarray, sample_t: float) -> int:
    return int(np.argmin(np.abs(t - float(sample_t))))


def predicted_absolute_xy(data: DemoData, row_idx: int) -> np.ndarray:
    predicted_error = data.predicted_xy[row_idx]
    out = np.empty_like(predicted_error)
    for idx in range(predicted_error.shape[0]):
        future_t = data.t[row_idx] + float(idx + 1) * float(data.control_period_s)
        ref_future = interp_vector(data.t, data.reference_xy, future_t)
        out[idx] = ref_future - predicted_error[idx]
    return out


def project_points(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 2)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    projected = np.column_stack((points, ones)) @ homography.T
    projected[:, :2] /= projected[:, 2:3]
    return projected[:, :2]


def draw_polyline(
    frame: np.ndarray,
    points_xy: np.ndarray,
    homography: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    closed: bool = False,
    line_type: int = cv2.LINE_AA,
) -> None:
    finite = np.isfinite(points_xy).all(axis=1)
    points_xy = points_xy[finite]
    if points_xy.shape[0] < 2:
        return
    pts = np.rint(project_points(points_xy, homography)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], closed, color, thickness, line_type)


def dashed_polyline(
    frame: np.ndarray,
    points_xy: np.ndarray,
    homography: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    dash: int = 8,
    gap: int = 8,
) -> None:
    pts = project_points(points_xy, homography)
    for a, b in zip(pts[:-1], pts[1:]):
        segment = b - a
        length = float(np.linalg.norm(segment))
        if length < 1.0:
            continue
        direction = segment / length
        pos = 0.0
        while pos < length:
            start = a + direction * pos
            end = a + direction * min(length, pos + dash)
            cv2.line(frame, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), color, thickness, cv2.LINE_AA)
            pos += dash + gap


def blend_color(start: tuple[int, int, int], end: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    alpha = min(1.0, max(0.0, float(alpha)))
    return tuple(
        int(round((1.0 - alpha) * float(a) + alpha * float(b)))
        for a, b in zip(start, end)
    )


def dashed_gradient_polyline(
    frame: np.ndarray,
    points_xy: np.ndarray,
    homography: np.ndarray,
    start_color: tuple[int, int, int],
    end_color: tuple[int, int, int],
    thickness: int,
    dash: int = 8,
    gap: int = 8,
) -> None:
    finite = np.isfinite(points_xy).all(axis=1)
    points_xy = points_xy[finite]
    if points_xy.shape[0] < 2:
        return
    pts = project_points(points_xy, homography)
    segment_count = max(1, pts.shape[0] - 2)
    for idx, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        color = blend_color(start_color, end_color, idx / segment_count)
        segment = b - a
        length = float(np.linalg.norm(segment))
        if length < 1.0:
            continue
        direction = segment / length
        pos = 0.0
        while pos < length:
            start = a + direction * pos
            end = a + direction * min(length, pos + dash)
            cv2.line(frame, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), color, thickness, cv2.LINE_AA)
            pos += dash + gap


def circle_points(center_xy: np.ndarray, radius_m: float, samples: int = 240) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=True)
    return np.column_stack(
        (
            center_xy[0] + radius_m * np.cos(theta),
            center_xy[1] + radius_m * np.sin(theta),
        )
    )


def box_points(limits: np.ndarray) -> np.ndarray:
    x_min, x_max, y_min, y_max = [float(v) for v in limits]
    return np.asarray(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float64,
    )


def cap_box_limits(limits: np.ndarray, max_half_extent_m: float) -> np.ndarray:
    x_min, x_max, y_min, y_max = [float(v) for v in limits]
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    half_x = min(0.5 * (x_max - x_min), float(max_half_extent_m))
    half_y = min(0.5 * (y_max - y_min), float(max_half_extent_m))
    return np.asarray(
        [
            center_x - half_x,
            center_x + half_x,
            center_y - half_y,
            center_y + half_y,
        ],
        dtype=np.float64,
    )


def draw_marker(
    frame: np.ndarray,
    point_xy: np.ndarray,
    homography: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    if not np.isfinite(point_xy).all():
        return
    point = tuple(np.rint(project_points(point_xy, homography)[0]).astype(int))
    cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, point, radius + 2, (255, 255, 255), 1, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, lines: list[str], anchor: str) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_width = max(
        cv2.getTextSize(line, font, PANEL_FONT_SCALE, PANEL_FONT_THICKNESS)[0][0]
        for line in lines
    )
    width = PANEL_PAD_X_PX * 2 + text_width
    height = PANEL_PAD_Y_PX * 2 + PANEL_LINE_HEIGHT_PX * len(lines)
    if anchor == "top_left":
        x0 = PANEL_MARGIN_PX
        y0 = PANEL_MARGIN_PX
    elif anchor == "bottom_left":
        x0 = PANEL_MARGIN_PX
        y0 = frame.shape[0] - height - PANEL_MARGIN_PX
    else:
        raise ValueError(f"unsupported panel anchor: {anchor}")
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), COLOR_TEXT_BG, -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1.0 - PANEL_ALPHA, 0.0, frame)
    for idx, line in enumerate(lines):
        y = y0 + PANEL_PAD_Y_PX + idx * PANEL_LINE_HEIGHT_PX + 17
        cv2.putText(
            frame,
            line,
            (x0 + PANEL_PAD_X_PX, y),
            font,
            PANEL_FONT_SCALE,
            COLOR_TEXT,
            PANEL_FONT_THICKNESS,
            cv2.LINE_AA,
        )


def draw_label(frame: np.ndarray, lines: list[str]) -> None:
    draw_panel(frame, lines, "top_left")


def interp_scalar(t: np.ndarray, values: np.ndarray, sample_t: float) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return math.nan
    return float(np.interp(sample_t, t[valid], values[valid]))


def draw_power_panel(frame: np.ndarray, data: DemoData, demo_t: float) -> None:
    power_w = interp_scalar(data.t, data.power_total_w, demo_t)
    power_age_s = interp_scalar(data.t, data.power_age_s, demo_t)
    if not math.isfinite(power_w):
        return
    lines = [f"Loihi power {power_w:0.2f} W"]
    if math.isfinite(power_age_s):
        lines.append(f"sample age {1000.0 * power_age_s:0.0f} ms")
    draw_panel(frame, lines, "bottom_left")


def draw_legend(frame: np.ndarray) -> None:
    entries = [
        ("commanded position", COLOR_COMMAND, "line_dot"),
        ("Loihi predicted xy", COLOR_PREDICTED_TRAJ, "gradient_dashed"),
        ("trajectory reference", COLOR_CIRCLE, "line_dot"),
        ("constraints", COLOR_BOX, "box"),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    swatch_w = 44
    text_width = max(
        cv2.getTextSize(label, font, PANEL_FONT_SCALE, PANEL_FONT_THICKNESS)[0][0]
        for label, _, _ in entries
    )
    width = PANEL_PAD_X_PX * 2 + swatch_w + 12 + text_width
    height = PANEL_PAD_Y_PX * 2 + PANEL_LINE_HEIGHT_PX * len(entries)
    x0 = frame.shape[1] - width - PANEL_MARGIN_PX
    y0 = PANEL_MARGIN_PX

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), COLOR_TEXT_BG, -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1.0 - PANEL_ALPHA, 0.0, frame)

    for idx, (label, color, style) in enumerate(entries):
        y = y0 + PANEL_PAD_Y_PX + idx * PANEL_LINE_HEIGHT_PX + 13
        x = x0 + PANEL_PAD_X_PX
        if style == "line_dot":
            cv2.line(frame, (x, y), (x + swatch_w, y), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (x + swatch_w // 2, y), 5, color, -1, cv2.LINE_AA)
        elif style == "dashed_dot":
            for start in range(0, swatch_w, 14):
                cv2.line(frame, (x + start, y), (x + min(start + 8, swatch_w), y), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (x + swatch_w // 2, y), 4, color, -1, cv2.LINE_AA)
        elif style == "gradient_dashed":
            for start in range(0, swatch_w, 10):
                alpha = start / max(1, swatch_w - 1)
                seg_color = blend_color(COLOR_PREDICTED_TRAJ_START, COLOR_PREDICTED_TRAJ_END, alpha)
                cv2.line(frame, (x + start, y), (x + min(start + 6, swatch_w), y), seg_color, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (x + 6, y - 8), (x + swatch_w - 6, y + 8), color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (x + swatch_w + 12, y + 6),
            font,
            PANEL_FONT_SCALE,
            COLOR_TEXT,
            PANEL_FONT_THICKNESS,
            cv2.LINE_AA,
        )


def crop_frame(frame: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return frame
    x, y, w, h = crop
    return frame[y : y + h, x : x + w].copy()


def constraint_box_corners(box_limits: np.ndarray) -> np.ndarray:
    corners = []
    for raw_limits in box_limits:
        x_min, x_max, y_min, y_max = cap_box_limits(
            raw_limits,
            MAX_DISPLAY_BOX_HALF_EXTENT_M,
        )
        if not np.isfinite([x_min, x_max, y_min, y_max]).all():
            continue
        corners.extend(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ]
        )
    if not corners:
        raise ValueError("No finite constraint-box corners found for output crop")
    return np.asarray(corners, dtype=np.float64)


def even_clamped_crop(
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    frame_size: tuple[int, int],
    padding_px: int,
) -> tuple[int, int, int, int]:
    width, height = frame_size
    pad = max(0, int(padding_px))
    x0 = max(0, int(math.floor(float(min_xy[0]))) - pad)
    y0 = max(0, int(math.floor(float(min_xy[1]))) - pad)
    x1 = min(width, int(math.ceil(float(max_xy[0]))) + pad)
    y1 = min(height, int(math.ceil(float(max_xy[1]))) + pad)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Computed output crop is empty")
    w = x1 - x0
    h = y1 - y0
    if w % 2 and x1 < width:
        x1 += 1
    elif w % 2 and x0 > 0:
        x0 -= 1
    if h % 2 and y1 < height:
        y1 += 1
    elif h % 2 and y0 > 0:
        y0 -= 1
    return x0, y0, x1 - x0, y1 - y0


def make_output_crop(
    data: DemoData,
    calibration: Calibration,
    frame_size: tuple[int, int],
    padding_px: int,
    enabled: bool,
) -> OutputCrop:
    if not enabled:
        return OutputCrop(rect=None, calibration=calibration, size=frame_size)
    projected = project_points(constraint_box_corners(data.box_limits), calibration.homography)
    crop = even_clamped_crop(
        np.nanmin(projected, axis=0),
        np.nanmax(projected, axis=0),
        frame_size,
        padding_px,
    )
    x, y, w, h = crop
    translate = np.asarray([[1.0, 0.0, -x], [0.0, 1.0, -y], [0.0, 0.0, 1.0]])
    cropped_calibration = Calibration(
        homography=translate @ calibration.homography,
        crop=None,
        start_offset_s=calibration.start_offset_s,
        reference_marker_advance_s=calibration.reference_marker_advance_s,
    )
    return OutputCrop(rect=crop, calibration=cropped_calibration, size=(w, h))


def draw_overlay(
    frame: np.ndarray,
    data: DemoData,
    calibration: Calibration,
    demo_t: float,
    trail_seconds: float,
) -> None:
    if demo_t < data.t[0] or demo_t > data.t[-1]:
        draw_label(frame, [f"outside demo segment: t={demo_t:0.2f}s"])
        return

    circle = circle_points(REFERENCE_CENTER_XY_M, REFERENCE_RADIUS_M)
    dashed_polyline(frame, circle, calibration.homography, COLOR_CIRCLE, 2)

    box = cap_box_limits(
        interp_vector(data.t, data.box_limits, demo_t),
        MAX_DISPLAY_BOX_HALF_EXTENT_M,
    )
    box_xy = box_points(box)
    draw_polyline(frame, box_xy, calibration.homography, COLOR_BOX, 3, closed=True)

    lx = 0.5 * (box[1] - box[0])
    ly = 0.5 * (box[3] - box[2])
    if lx < REFERENCE_RADIUS_M * 0.9:
        draw_polyline(frame, box_xy[[0, 3]], calibration.homography, COLOR_BOX_ACTIVE, 6)
        draw_polyline(frame, box_xy[[1, 2]], calibration.homography, COLOR_BOX_ACTIVE, 6)
    if ly < REFERENCE_RADIUS_M * 0.9:
        draw_polyline(frame, box_xy[[0, 1]], calibration.homography, COLOR_BOX_ACTIVE, 6)
        draw_polyline(frame, box_xy[[3, 2]], calibration.homography, COLOR_BOX_ACTIVE, 6)

    trail_start_t = max(data.t[0], demo_t - trail_seconds)
    trail_mask = (data.t >= trail_start_t) & (data.t <= demo_t)
    draw_polyline(frame, data.state_xy[trail_mask], calibration.homography, COLOR_STATE, 3)
    state_now = interp_vector(data.t, data.state_xy, demo_t)
    draw_marker(frame, state_now, calibration.homography, COLOR_STATE, 7)

    draw_polyline(frame, data.command_xy[trail_mask], calibration.homography, COLOR_COMMAND, 2)
    command_now = interp_vector(data.t, data.command_xy, demo_t)
    draw_marker(frame, command_now, calibration.homography, COLOR_COMMAND, 5)

    predicted_now = predicted_absolute_xy(data, nearest_row_index(data.t, demo_t))
    if np.isfinite(predicted_now).all():
        dashed_gradient_polyline(
            frame,
            predicted_now,
            calibration.homography,
            COLOR_PREDICTED_TRAJ_START,
            COLOR_PREDICTED_TRAJ_END,
            2,
            dash=7,
            gap=5,
        )
        marker_count = max(1, predicted_now.shape[0] - 1)
        for idx, point_xy in enumerate(predicted_now):
            draw_marker(
                frame,
                point_xy,
                calibration.homography,
                blend_color(COLOR_PREDICTED_TRAJ_START, COLOR_PREDICTED_TRAJ_END, idx / marker_count),
                3,
            )

    ref_now = interp_vector(
        data.t,
        data.reference_xy,
        min(data.t[-1], demo_t + calibration.reference_marker_advance_s),
    )
    draw_marker(frame, ref_now, calibration.homography, COLOR_CIRCLE, 6)

    if lx < REFERENCE_RADIUS_M * 0.9 and ly >= REFERENCE_RADIUS_M * 0.9:
        phase = "x constraints active"
    elif ly < REFERENCE_RADIUS_M * 0.9 and lx >= REFERENCE_RADIUS_M * 0.9:
        phase = "y constraints active"
    elif lx < REFERENCE_RADIUS_M * 0.9 and ly < REFERENCE_RADIUS_M * 0.9:
        phase = "x/y constraints active"
    else:
        phase = "open box"
    draw_label(frame, [f"Loihi demo t = {demo_t:0.1f}s", f"Lx={lx:0.2f}m  Ly={ly:0.2f}m", phase])
    draw_power_panel(frame, data, demo_t)
    draw_legend(frame)


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    return cap


def render_preview(args: argparse.Namespace, data: DemoData, calibration: Calibration) -> None:
    cap = open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_t = float(args.preview_frame)
    cap.set(cv2.CAP_PROP_POS_MSEC, video_t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read preview frame at {video_t:.3f}s")
    frame = crop_frame(frame, calibration.crop)
    output_crop = make_output_crop(
        data,
        calibration,
        frame_size=(frame.shape[1], frame.shape[0]),
        padding_px=int(args.box_crop_padding_px),
        enabled=not bool(args.no_box_crop),
    )
    frame = crop_frame(frame, output_crop.rect)
    draw_overlay(
        frame,
        data,
        output_crop.calibration,
        demo_t=video_time_to_demo_time(video_t, float(args.start_offset_s)),
        trail_seconds=float(args.trail_seconds),
    )
    args.preview_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.preview_output), frame):
        raise RuntimeError(f"Could not write {args.preview_output}")
    print(f"Wrote preview {args.preview_output} ({fps:0.3f} fps source)")


def render_video(args: argparse.Namespace, data: DemoData, calibration: Calibration) -> None:
    cap = open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if calibration.crop is None:
        out_size = (width, height)
    else:
        _, _, w, h = calibration.crop
        out_size = (w, h)
    output_crop = make_output_crop(
        data,
        calibration,
        frame_size=out_size,
        padding_px=int(args.box_crop_padding_px),
        enabled=not bool(args.no_box_crop),
    )
    out_size = output_crop.size

    start_video_t = (
        float(args.start_offset_s)
        if args.video_start_s is None
        else float(args.video_start_s)
    )
    start_video_t = max(0.0, start_video_t)
    default_end = min(
        (total_frames / fps) if total_frames > 0 else data.t[-1] + float(args.start_offset_s),
        data.t[-1] + float(args.start_offset_s),
    )
    end_video_t = float(args.video_end_s) if args.video_end_s is not None else default_end
    if end_video_t <= start_video_t:
        raise ValueError("video end time must be greater than video start time")

    cap.set(cv2.CAP_PROP_POS_MSEC, start_video_t * 1000.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, out_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {args.output}")

    frame_idx = int(round(start_video_t * fps))
    written = 0
    try:
        while True:
            video_t = frame_idx / fps
            if video_t > end_video_t:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame = crop_frame(frame, calibration.crop)
            frame = crop_frame(frame, output_crop.rect)
            draw_overlay(
                frame,
                data,
                output_crop.calibration,
                demo_t=video_time_to_demo_time(video_t, float(args.start_offset_s)),
                trail_seconds=float(args.trail_seconds),
            )
            writer.write(frame)
            written += 1
            frame_idx += 1
            if written == 1 or written % int(max(round(fps * 5), 1)) == 0:
                print(f"Rendered {written} frames", flush=True)
    finally:
        writer.release()
        cap.release()
    print(f"Wrote {args.output} ({written} frames at {fps:0.3f} fps)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-output", type=Path, default=DEFAULT_PREVIEW_OUTPUT)
    parser.add_argument("--write-calibration-template", type=Path, default=None)
    parser.add_argument(
        "--interactive-circle-calibration",
        action="store_true",
        help="Build a motion composite, let you click the projected circle, and write calibration JSON.",
    )
    parser.add_argument(
        "--motion-composite-output",
        type=Path,
        default=Path("video/GOPR0101_motion_composite.jpg"),
    )
    parser.add_argument("--composite-stride", type=int, default=5)
    parser.add_argument("--composite-max-frames", type=int, default=2000)
    parser.add_argument("--composite-method", choices=("max", "mean", "diff"), default="max")
    parser.add_argument("--display-scale", type=float, default=0.0)
    parser.add_argument("--circle-calibration-samples", type=int, default=16)
    parser.add_argument("--circle-calibration-axis-angle-deg", type=float, default=0.0)
    parser.add_argument("--circle-calibration-flip-x", action="store_true")
    parser.add_argument("--circle-calibration-flip-y", action="store_true")
    parser.add_argument("--preview-frame", type=float, default=None, help="Video timestamp in seconds to preview.")
    parser.add_argument(
        "--start-offset-s",
        type=float,
        default=None,
        help=(
            "Video timestamp where the logged demo starts; overrides calibration JSON "
            "start_offset_s. demo_time = video_time - start_offset_s."
        ),
    )
    parser.add_argument(
        "--video-start-s",
        type=float,
        default=None,
        help="Video timestamp to start rendering. Defaults to the calibrated demo start.",
    )
    parser.add_argument("--video-end-s", type=float, default=None)
    parser.add_argument("--trail-seconds", type=float, default=5.0)
    parser.add_argument(
        "--box-crop-padding-px",
        type=int,
        default=100,
        help="Padding around the projected full constraint-box extent in output pixels.",
    )
    parser.add_argument(
        "--no-box-crop",
        action="store_true",
        help="Disable automatic output crop around the projected constraint box.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_calibration_template is not None:
        write_calibration_template(args.write_calibration_template)
        return
    if args.interactive_circle_calibration:
        interactive_circle_calibration(args)
        return

    data = load_demo_data(args.log)
    calibration = load_calibration(args.calibration)
    if args.start_offset_s is None:
        args.start_offset_s = calibration.start_offset_s
    if args.preview_frame is not None:
        render_preview(args, data, calibration)
    else:
        render_video(args, data, calibration)


if __name__ == "__main__":
    main()
