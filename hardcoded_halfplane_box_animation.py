#!/usr/bin/env python3
"""Render a hard-coded quadrotor/halfplane-box animation.

Run:
    source ~/venv/bin/activate
    python hardcoded_halfplane_box_animation.py

The output video is written to:
    results/hardcoded_halfplane_box_animation/quadrotor_halfplane_box.mp4
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


RADIUS = 1.0
FPS = 30
DURATION = 24.0
N_FRAMES = int(FPS * DURATION)
DT = DURATION / (N_FRAMES - 1)
FIG_SIZE = (4.8, 4.8)
FIG_DPI = 80
BITRATE = 700

WIDE = 1.22
NARROW = 0.0
OUTPUT_DIR = Path("results/hardcoded_halfplane_box_animation")
OUTPUT_MP4 = OUTPUT_DIR / "quadrotor_halfplane_box.mp4"


def smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    """Cubic easing for smooth constraint motion."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def hold_profile(t: float, start: float, shrink: float, hold: float, expand: float) -> float:
    """Return 0 for wide, 1 for narrow, with smooth in/out phases."""
    if t < start:
        return 0.0
    if t < start + shrink:
        return float(smoothstep((t - start) / shrink))
    if t < start + shrink + hold:
        return 1.0
    if t < start + shrink + hold + expand:
        return float(1.0 - smoothstep((t - start - shrink - hold) / expand))
    return 0.0


def bounds_at(t: float) -> tuple[float, float, float, float]:
    """Return Lx, Ly and narrowing indicators for the four halfplanes."""
    # First the left/right halfplanes move in, forcing x ~= 0 and vertical motion.
    x_squeeze = hold_profile(t, start=4.0, shrink=2.5, hold=3.5, expand=2.5)

    # Then the top/bottom halfplanes move in, forcing y ~= 0 and horizontal motion.
    y_squeeze = hold_profile(t, start=14.0, shrink=2.5, hold=3.5, expand=2.5)

    lx = WIDE - (WIDE - NARROW) * x_squeeze
    ly = WIDE - (WIDE - NARROW) * y_squeeze
    return lx, ly, x_squeeze, y_squeeze


def halfplane_project(reference_xy: np.ndarray, lx: float, ly: float) -> np.ndarray:
    """Project the reference point into A p <= b for box halfplanes."""
    a_xy = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    b_xy = np.array([lx, lx, ly, ly])

    # For this axis-aligned halfplane set the Euclidean projection is equivalent
    # to saturating each coordinate to the currently feasible interval.
    projected = reference_xy.copy()
    projected[0] = min(max(projected[0], -b_xy[1] / -a_xy[1, 0]), b_xy[0] / a_xy[0, 0])
    projected[1] = min(max(projected[1], -b_xy[3] / -a_xy[3, 1]), b_xy[2] / a_xy[2, 1])
    return projected


def generate_trajectory() -> dict[str, np.ndarray]:
    t = np.linspace(0.0, DURATION, N_FRAMES)
    theta = 2.0 * np.pi * t / 8.0
    reference = np.column_stack((RADIUS * np.cos(theta), RADIUS * np.sin(theta)))

    actual = np.zeros_like(reference)
    bounds = np.zeros((N_FRAMES, 2))
    squeeze = np.zeros((N_FRAMES, 2))
    violation = np.zeros(N_FRAMES)

    a_xy = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )

    for i, ti in enumerate(t):
        lx, ly, sx, sy = bounds_at(float(ti))
        bounds[i] = [lx, ly]
        squeeze[i] = [sx, sy]
        actual[i] = halfplane_project(reference[i], lx, ly)
        b_xy = np.array([lx, lx, ly, ly])
        violation[i] = np.max(a_xy @ actual[i] - b_xy)

    return {
        "t": t,
        "reference": reference,
        "actual": actual,
        "bounds": bounds,
        "squeeze": squeeze,
        "violation": violation,
    }


def render_video(data: dict[str, np.ndarray]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#f7f7f2")
    ax.set_facecolor("#fbfbf7")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Quadrotor constrained by four moving halfplanes")
    ax.grid(True, color="#d9d6c8", linewidth=0.8, alpha=0.7)

    circle = plt.Circle(
        (0.0, 0.0),
        RADIUS,
        fill=False,
        color="#808080",
        linestyle=(0, (5, 5)),
        linewidth=1.8,
        label="fixed circular reference",
    )
    ax.add_patch(circle)

    box_patch = Rectangle(
        (-WIDE, -WIDE),
        2.0 * WIDE,
        2.0 * WIDE,
        fill=False,
        edgecolor="#1f4e79",
        linewidth=2.8,
        label="current halfplane box",
    )
    ax.add_patch(box_patch)

    ref_dot, = ax.plot([], [], "o", color="#9a9a9a", markersize=4.5, label="reference point")
    trail, = ax.plot([], [], color="#d1495b", linewidth=2.2, label="constrained trajectory")
    dot, = ax.plot([], [], "o", color="#d1495b", markersize=9.0, label="quadrotor")
    status = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ffffffe8", "edgecolor": "#d0d0c8"},
    )

    face_lines = {
        "right": ax.axvline(WIDE, color="#1f4e79", linewidth=3.5, alpha=0.0),
        "left": ax.axvline(-WIDE, color="#1f4e79", linewidth=3.5, alpha=0.0),
        "top": ax.axhline(WIDE, color="#1f4e79", linewidth=3.5, alpha=0.0),
        "bottom": ax.axhline(-WIDE, color="#1f4e79", linewidth=3.5, alpha=0.0),
    }

    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

    reference = data["reference"]
    actual = data["actual"]
    bounds = data["bounds"]
    squeeze = data["squeeze"]
    times = data["t"]

    def init():
        ref_dot.set_data([], [])
        dot.set_data([], [])
        trail.set_data([], [])
        return [box_patch, ref_dot, dot, trail, status, *face_lines.values()]

    def update(frame: int):
        lx, ly = bounds[frame]
        sx, sy = squeeze[frame]
        box_patch.set_xy((-lx, -ly))
        box_patch.set_width(2.0 * lx)
        box_patch.set_height(2.0 * ly)

        face_lines["right"].set_xdata([lx, lx])
        face_lines["left"].set_xdata([-lx, -lx])
        face_lines["top"].set_ydata([ly, ly])
        face_lines["bottom"].set_ydata([-ly, -ly])

        x_active = sx > 0.02
        y_active = sy > 0.02
        face_lines["right"].set_alpha(0.65 if x_active else 0.0)
        face_lines["left"].set_alpha(0.65 if x_active else 0.0)
        face_lines["top"].set_alpha(0.65 if y_active else 0.0)
        face_lines["bottom"].set_alpha(0.65 if y_active else 0.0)

        ref_dot.set_data([reference[frame, 0]], [reference[frame, 1]])
        dot.set_data([actual[frame, 0]], [actual[frame, 1]])
        start = max(0, frame - 160)
        trail.set_data(actual[start : frame + 1, 0], actual[start : frame + 1, 1])

        if x_active and not y_active:
            phase = "left/right halfplanes shrinking: vertical harmonic motion"
        elif y_active and not x_active:
            phase = "top/bottom halfplanes shrinking: horizontal harmonic motion"
        else:
            phase = "wide box: circular tracking"

        status.set_text(
            f"t = {times[frame]:4.1f} s\n"
            f"Lx = {lx:0.2f}, Ly = {ly:0.2f}\n"
            f"{phase}"
        )
        return [box_patch, ref_dot, dot, trail, status, *face_lines.values()]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=N_FRAMES,
        init_func=init,
        interval=1000.0 / FPS,
        blit=True,
    )

    writer = animation.FFMpegWriter(
        fps=FPS,
        bitrate=BITRATE,
        codec="libx264",
        extra_args=["-preset", "ultrafast", "-pix_fmt", "yuv420p"],
    )

    def progress_callback(frame: int, total_frames: int) -> None:
        current = frame + 1
        if current == 1 or current == total_frames or current % FPS == 0:
            percent = 100.0 * current / total_frames
            print(f"Rendering frame {current:4d}/{total_frames} ({percent:5.1f}%)", flush=True)

    print(
        f"Rendering {N_FRAMES} frames at {FPS} fps, "
        f"{int(FIG_SIZE[0] * FIG_DPI)}x{int(FIG_SIZE[1] * FIG_DPI)} px, "
        f"bitrate={BITRATE}...",
        flush=True,
    )
    ani.save(OUTPUT_MP4, writer=writer, progress_callback=progress_callback)
    plt.close(fig)


def main() -> None:
    data = generate_trajectory()
    render_video(data)

    print(f"Wrote {OUTPUT_MP4}")
    print(f"Maximum halfplane violation: {np.max(data['violation']):.3e}")
    print("Halfplanes: x <= Lx, -x <= Lx, y <= Ly, -y <= Ly")


if __name__ == "__main__":
    main()
