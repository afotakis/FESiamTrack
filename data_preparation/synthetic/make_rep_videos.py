#!/usr/bin/env python3
import os
import re
from pathlib import Path
import numpy as np
import h5py
import hdf5plugin
import cv2
import fire
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)

IMG_H = 384
IMG_W = 512

def _dt_to_str(dt: float) -> str:
    return f"{dt:.4f}"

def _natural_sort_key(path: Path):
    # Extract numeric part of stems like "0400000", "0400000.0"
    s = path.stem
    try:
        # drop leading "0" then parse float or int
        return float(s)
    except Exception:
        # fallback to plain string
        return s

def _find_rep_dir(events_dir: Path, rep_type: str, n_bins: int, dt: float):
    """
    Try multiple possibilities to find the representation directory produced by your generators.
    """
    if rep_type == "event_count":
        cand = events_dir / "count_images"
        return cand if cand.exists() else None

    dt_str = _dt_to_str(dt)
    base = events_dir / dt_str

    candidates = []
    if rep_type in ("time_surface", "time_surface_gpu"):
        # CPU & GPU naming
        candidates += [
            base / f"time_surfaces_v2_{n_bins}",
            base / f"time_surfaces_v2_gpu_{n_bins}",
        ]
        # CPU hard-coded '0.0200' fallback
        candidates += [
            events_dir / "0.0200" / f"time_surfaces_v2_{n_bins}",
            events_dir / "0.0200" / f"time_surfaces_v2_gpu_{n_bins}",
        ]
    elif rep_type == "voxel_grid":
        candidates += [base / f"voxel_grids_{n_bins}"]
    elif rep_type == "event_stack":
        candidates += [base / f"event_stacks_{n_bins}"]
    else:
        return None

    for c in candidates:
        if c.exists():
            return c
    return None

def _load_frame(path: Path, rep_type: str):
    """
    Return an HxWx3 uint8 BGR frame for the video.
    """
    if rep_type == "event_count":
        arr = np.load(str(path))  # (H,W,2) uint8 counts
        # Per-frame normalize to [0,1]
        pos = arr[..., 1].astype(np.float32)
        neg = arr[..., 0].astype(np.float32)
        def norm(a):
            m = a.max()
            return a / m if m > 0 else a
        pos = norm(pos)
        neg = norm(neg)
        rgb = np.stack([pos, neg, np.zeros_like(pos)], axis=-1)
        frame = (255.0 * rgb).astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # HDF5 path
    with h5py.File(str(path), "r") as h5f:
        if "time_surface" in h5f:
            ts = h5f["time_surface"][()]  # (H,W,2*n_bins) float32 in [0,1]
            # Even channels: p=0 (neg); Odd channels: p=1 (pos)
            pos = ts[..., 1::2].max(axis=-1) if ts.shape[-1] >= 2 else np.zeros(ts.shape[:2], np.float32)
            neg = ts[..., 0::2].max(axis=-1)
            # Per-frame normalization
            pos = pos / max(pos.max(), 1e-8)
            neg = neg / max(neg.max(), 1e-8)
            rgb = np.stack([pos, neg, np.zeros_like(pos)], axis=-1)
            frame = (255.0 * rgb).astype(np.uint8)
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if "voxel_grid" in h5f:
            vg = h5f["voxel_grid"][()].astype(np.float32)  # (H,W,C)
            pos = np.maximum(vg, 0).max(axis=-1)
            neg = np.maximum(-vg, 0).max(axis=-1)
            # Normalize
            pos = pos / max(pos.max(), 1e-8)
            neg = neg / max(neg.max(), 1e-8)
            rgb = np.stack([pos, neg, np.zeros_like(pos)], axis=-1)
            frame = (255.0 * rgb).astype(np.uint8)
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if "event_stack" in h5f:
            es = h5f["event_stack"][()].astype(np.float32)  # (H,W,B) signed sums
            pos = np.maximum(es, 0).max(axis=-1)
            neg = np.maximum(-es, 0).max(axis=-1)
            # Normalize
            pos = pos / max(pos.max(), 1e-8)
            neg = neg / max(neg.max(), 1e-8)
            rgb = np.stack([pos, neg, np.zeros_like(pos)], axis=-1)
            frame = (255.0 * rgb).astype(np.uint8)
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    raise RuntimeError(f"Unrecognized dataset keys in {path}")

def _list_frame_files(rep_dir: Path, rep_type: str):
    if rep_dir is None or not rep_dir.exists():
        return []
    if rep_type == "event_count":
        files = sorted(rep_dir.glob("*.npy"), key=_natural_sort_key)
    else:
        files = sorted(rep_dir.glob("*.h5"), key=_natural_sort_key)
    return files

def _default_fps(dt: float, fallback: int = 30) -> int:
    # Try to match real sampling: fps ≈ 1/dt
    if dt and dt > 0:
        fps = int(round(1.0 / dt))
        return max(1, min(fps, 240))  # keep reasonable
    return fallback

def _load_time_surface_array(path: Path):
    with h5py.File(str(path), "r") as h5f:
        if "time_surface" not in h5f:
            raise RuntimeError(f"No 'time_surface' in {path}")
        ts = h5f["time_surface"][()]  # (H, W, 2*B), float32 in [0,1]
    return ts

def _save_time_surface_manifolds_for_frame(
    h5_path: Path,
    out_dir: Path,
    n_bins: int,
    stride: int = 4,
    elev: int = 45,
    azim: int = 60,
    dpi: int = 120,
):
    """
    Save one PNG per bin showing 2 subplots:
      left: bin's POS surface (p=1)
      right: bin's NEG surface (p=0)
    """
    ts = _load_time_surface_array(h5_path)  # (H, W, 2B)
    H, W, C = ts.shape
    assert C == 2 * n_bins, f"Expected 2*{n_bins} channels, got {C}"

    ys = np.arange(0, H, max(1, stride))
    xs = np.arange(0, W, max(1, stride))
    X, Y = np.meshgrid(xs, ys)

    out_dir.mkdir(parents=True, exist_ok=True)
    for b in range(n_bins):
        pos = ts[..., 2 * b + 1]
        neg = ts[..., 2 * b + 0]
        Zp = pos[np.ix_(ys, xs)]
        Zn = neg[np.ix_(ys, xs)]

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        s1 = ax1.plot_surface(X, Y, Zp, linewidth=0, antialiased=False)
        ax1.set_title(f"bin {b} (pos = p=1)")
        ax1.set_zlim(0, 1)
        ax1.view_init(elev=elev, azim=azim)
        ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("S")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        s2 = ax2.plot_surface(X, Y, Zn, linewidth=0, antialiased=False)
        ax2.set_title(f"bin {b} (neg = p=0)")
        ax2.set_zlim(0, 1)
        ax2.view_init(elev=elev, azim=azim)
        ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("S")

        fig.tight_layout()
        out_file = out_dir / f"{h5_path.stem}_bin{b}.png"
        fig.savefig(out_file, dpi=dpi)
        plt.close(fig)



def make_videos(
    extras_root: str,
    split: str = "train",
    rep_type: str = "time_surface",   # ["time_surface", "time_surface_gpu", "voxel_grid", "event_stack", "event_count"]
    n_bins: int = 5,
    dt: float = 0.02,
    first_n: int = 5,
    fps: int = None,
    overwrite: bool = False,
    save_surfaces: bool = True,
    surface_frame_index: int = 0,  # which frame to snapshot (default: first)
    surface_stride: int = 4,       # subsampling for speed
    surface_elev: int = 45,
    surface_azim: int = 60,
    surface_dpi: int = 120,
):
    """
    Create MP4 videos for the first N sequences of a split from precomputed representations.

    extras_root: path to the *output* dataset root (e.g., multiflow_reloaded_extra)
    split: 'train' or 'test'
    rep_type: one of ["time_surface", "time_surface_gpu", "voxel_grid", "event_stack", "event_count"]
    n_bins: number of bins used during preprocessing
    dt: window step used during preprocessing (seconds)
    first_n: number of sequences to process
    fps: frames per second for the video (default ~ 1/dt)
    overwrite: if False, skip sequences whose video already exists
    """
    extras_root = Path(extras_root)
    split_dir = extras_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    sequences = sorted([
        p for p in split_dir.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and (p / "events").is_dir()
    ])
    sequences = sequences[:first_n]

    if fps is None:
        fps = _default_fps(dt)

    print(f"[make_videos] split={split} rep={rep_type} n_bins={n_bins} dt={dt} fps={fps}")
    print(f"[make_videos] Processing {len(sequences)} sequences from: {split_dir}")

    for seq_dir in tqdm(sequences, desc=f"Videos ({split})"):
        events_dir = seq_dir / "events"
        rep_dir = _find_rep_dir(events_dir, rep_type, n_bins, dt)
        if rep_dir is None:
            print(f"  ! No representation directory for {seq_dir.name} (rep={rep_type}). Skipping.")
            continue

        # Output video path
        videos_out = seq_dir / "videos"
        videos_out.mkdir(parents=True, exist_ok=True)
        safe_dt = _dt_to_str(dt) if rep_type != "event_count" else "NA"
        out_name = f"{rep_type}_bins{n_bins}_dt{safe_dt}.mp4"
        out_path = videos_out / out_name

        if out_path.exists() and not overwrite:
            print(f"  - Exists, skipping: {out_path}")
            continue

        frame_files = _list_frame_files(rep_dir, rep_type)
        if len(frame_files) == 0:
            print(f"  ! No frames in {rep_dir}. Skipping {seq_dir.name}.")
            continue

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (IMG_W, IMG_H))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter. Try installing codecs or switch to .avi with MJPG.")

        # Write frames
        for f in frame_files:
            try:
                frame_bgr = _load_frame(f, rep_type)
                # Ensure correct size (in case)
                if frame_bgr.shape[0] != IMG_H or frame_bgr.shape[1] != IMG_W:
                    frame_bgr = cv2.resize(frame_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
                writer.write(frame_bgr)
            except Exception as e:
                print(f"    ! Failed on frame {f}: {e}")

        writer.release()
        print(f"  ✓ Wrote: {out_path}")
                # --- NEW: 3D manifold snapshots (time_surface only) ---
        if save_surfaces and rep_type.startswith("time_surface"):
            if len(frame_files) == 0:
                print(f"  ! No frames in {rep_dir}. Skipping surfaces for {seq_dir.name}.")
            else:
                idx = min(max(0, surface_frame_index), len(frame_files) - 1)
                surf_dir = seq_dir / "videos" / "surfaces"
                try:
                    _save_time_surface_manifolds_for_frame(
                        frame_files[idx],
                        surf_dir,
                        n_bins=n_bins,
                        stride=surface_stride,
                        elev=surface_elev,
                        azim=surface_azim,
                        dpi=surface_dpi,
                    )
                    print(f"  ✓ Saved 3D surfaces to: {surf_dir}")
                except Exception as e:
                    print(f"  ! Surface save failed for {seq_dir.name}: {e}")

if __name__ == "__main__":
    fire.Fire(make_videos)
