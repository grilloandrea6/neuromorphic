#!/usr/bin/env python3
"""Create poster-ready still figures for the Loihi halfplane-box demo."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from hardcoded_halfplane_box_animation import RADIUS, bounds_at, generate_trajectory


OUTPUT_DIR = Path("results/poster_figures")
VIDEO_PATH = Path("video/GOPR0101_overlay.mp4")


def save_schematic(output_path: Path) -> None:
    data = generate_trajectory()
    snapshots = [
        (2.0, "Open box", "Circular reference is feasible"),
        (8.0, "Left/right active", "Reference projected onto x-limited corridor"),
        (18.0, "Top/bottom active", "Reference projected onto y-limited corridor"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), dpi=220)
    fig.patch.set_facecolor("#f6f6f2")

    for ax, (time_s, title, subtitle) in zip(axes, snapshots):
        frame = int(np.argmin(np.abs(data["t"] - time_s)))
        lx, ly, x_squeeze, y_squeeze = bounds_at(time_s)

        ax.set_facecolor("#fbfbf7")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.38, 1.38)
        ax.set_ylim(-1.38, 1.38)
        ax.grid(True, color="#d7d3c5", linewidth=0.6, alpha=0.75)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title, fontsize=16, weight="bold", pad=18)
        ax.text(
            0.5,
            1.025,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#343434",
        )

        theta = np.linspace(0.0, 2.0 * np.pi, 360)
        ax.plot(
            RADIUS * np.cos(theta),
            RADIUS * np.sin(theta),
            color="#858585",
            linestyle=(0, (5, 4)),
            linewidth=2.0,
            label="circular reference",
        )

        rect = Rectangle(
            (-lx, -ly),
            2.0 * lx,
            2.0 * ly,
            fill=False,
            edgecolor="#125d98",
            linewidth=3.0,
            label="halfplane box",
        )
        ax.add_patch(rect)

        if x_squeeze > 0.02:
            ax.plot([-lx, -lx], [-ly, ly], color="#f26c2f", linewidth=5.0, solid_capstyle="round")
            ax.plot([lx, lx], [-ly, ly], color="#f26c2f", linewidth=5.0, solid_capstyle="round")
        if y_squeeze > 0.02:
            ax.plot([-lx, lx], [-ly, -ly], color="#f26c2f", linewidth=5.0, solid_capstyle="round")
            ax.plot([-lx, lx], [ly, ly], color="#f26c2f", linewidth=5.0, solid_capstyle="round")

        start = max(0, frame - 170)
        ax.plot(
            data["actual"][start : frame + 1, 0],
            data["actual"][start : frame + 1, 1],
            color="#d1495b",
            linewidth=3.0,
            label="constrained command",
        )
        ax.scatter(
            data["reference"][frame, 0],
            data["reference"][frame, 1],
            s=55,
            color="#858585",
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )
        ax.scatter(
            data["actual"][frame, 0],
            data["actual"][frame, 1],
            s=95,
            color="#d1495b",
            edgecolor="white",
            linewidth=1.3,
            zorder=5,
        )

        ax.text(
            0.03,
            0.04,
            f"t = {time_s:0.0f} s\nLx = {lx:0.2f}, Ly = {ly:0.2f}",
            transform=ax.transAxes,
            fontsize=10,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fffffff0", "edgecolor": "#c8c8c0"},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle("Moving halfplanes turn circular tracking into constrained motion", fontsize=20, weight="bold")
    fig.subplots_adjust(left=0.045, right=0.99, top=0.82, bottom=0.18, wspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def read_video_frame(video_path: Path, time_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame at {time_s:.1f}s from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_video_triptych(output_path: Path) -> None:
    snapshots = [
        (6.0, "Unconstrained circular reference"),
        (15.0, "Left/right box constraints"),
        (32.0, "Top/bottom box constraints"),
    ]
    frames = [read_video_frame(VIDEO_PATH, time_s) for time_s, _ in snapshots]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.35), dpi=220)
    # fig.patch.set_facecolor("#f6f6f2")

    for ax, frame, (time_s, title) in zip(axes, frames, snapshots):
        ax.imshow(frame)
        ax.set_axis_off()
        ax.set_title(title, fontsize=15, weight="bold", pad=10)
        # ax.text(0.5, -0.045, f"overlay video t = {time_s:0.0f}s", transform=ax.transAxes, ha="center", va="top", fontsize=10.5)

    fig.suptitle("Circular trajectory tracking with time-varying constraints", fontsize=20, weight="bold")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.82, bottom=0.08, wspace=0.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    save_schematic(OUTPUT_DIR / "experiment_schematic.png")
    save_video_triptych(OUTPUT_DIR / "flight_overlay_triptych.png")
    print(f"Wrote {OUTPUT_DIR / 'experiment_schematic.png'}")
    print(f"Wrote {OUTPUT_DIR / 'flight_overlay_triptych.png'}")


if __name__ == "__main__":
    main()
