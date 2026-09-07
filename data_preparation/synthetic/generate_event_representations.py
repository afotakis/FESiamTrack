import multiprocessing
import os
from pathlib import Path
import torch
import hdf5plugin
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import cv2
import fire
import h5py
import numpy as np
from tqdm import tqdm
import math
from utils.representations import VoxelGrid, events_to_voxel_grid
from utils.utils import blosc_opts

IMG_H = 384
IMG_W = 512
VOXEL_GRID_CONSTRUCTOR = VoxelGrid((5, 384, 512), True)


def generate_event_count_images_single(
    input_seq_dir, output_dir, visualize=False, dt=0.01, **kwargs
):
    """
    For each ts in [0.4:0.01:0.9], generate the event count image for a Multiflow sequence
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / split / input_seq_dir.stem
    output_dir = output_seq_dir / "events" / "count_images"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6

    with h5py.File(str(input_seq_dir / "events" / "events.h5"), "r") as h5f:
        time = np.asarray(h5f["t"])

        for t1 in np.arange(400000, 900000 + dt_us, dt_us):
            output_path = output_dir / f"0{t1}.npy"
            if output_path.exists():
                continue

            t0 = t1 - dt_us
            first_idx = np.searchsorted(time, t0, side="left")
            last_idx_p1 = np.searchsorted(time, t1, side="right")
            out = {
                "x": np.asarray(h5f["x"][first_idx:last_idx_p1]),
                "y": np.asarray(h5f["y"][first_idx:last_idx_p1]),
                "p": np.asarray(h5f["p"][first_idx:last_idx_p1]),
                "t": time[first_idx:last_idx_p1],
            }
            n_events = out["x"].shape[0]
            img_counts = np.zeros((IMG_H, IMG_W, 2), dtype=np.uint8)
            for i in range(n_events):
                img_counts[out["y"][i], out["x"][i], out["p"][i]] += 1

            # Write to disk
            np.save(str(output_path), img_counts)

            # Visualize
            if visualize:
                img_vis = np.interp(img_counts, (0, img_counts.max()), (0, 255)).astype(
                    np.uint8
                )
                img_vis = np.concatenate([img_vis, np.zeros((IMG_H, IMG_W, 1))], axis=2)
                cv2.imshow("Count Image", img_vis)
                cv2.waitKey(1)


def generate_sbt_single(
    input_seq_dir, output_dir, visualize=False, n_bins=5, dt=0.01, **kwargs
):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / split / input_seq_dir.stem
    output_dir = output_seq_dir / "events" / f"{dt:.4f}" / f"event_stacks_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6
    dt_us_bin = dt_us / n_bins

    with h5py.File(str(input_seq_dir / "events" / "events.h5"), "r") as h5f:
        x, y, p, time = (
            np.asarray(h5f["x"]),
            np.asarray(h5f["y"]),
            np.asarray(h5f["p"]),
            np.asarray(h5f["t"]),
        )

        # dt of labels
        for t1 in np.arange(400000, 900000 + dt_us, dt_us):
            output_path = output_dir / f"0{t1}.h5"
            if output_path.exists():
                continue

            time_surface = np.zeros((IMG_H, IMG_W, n_bins), dtype=np.int64)
            t0 = t1 - dt_us

            # iterate over bins
            for i_bin in range(n_bins):
                t0_bin = t0 + i_bin * dt_us_bin
                t1_bin = t0_bin + dt_us_bin
                idx0 = np.searchsorted(time, t0_bin, side="left")
                idx1 = np.searchsorted(time, t1_bin, side="right")
                x_bin = x[idx0:idx1]
                y_bin = y[idx0:idx1]
                p_bin = p[idx0:idx1] * 2 - 1

                n_events = len(x_bin)
                for i in range(n_events):
                    time_surface[y_bin[i], x_bin[i], i_bin] += p_bin[i]

            # Write to disk
            with h5py.File(output_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "event_stack",
                    data=time_surface,
                    shape=time_surface.shape,
                    dtype=np.float32,
                    **blosc_opts(complevel=1, shuffle="byte"),
                )
            # Visualize
            if visualize:
                for i in range(n_bins):
                    cv2.imshow(
                        f"Time Surface Bin {i}",
                        (time_surface[:, :, i] * 255).astype(np.uint8),
                    )
                    cv2.waitKey(0)

import torch
import numpy as np
import h5py
from pathlib import Path

def generate_time_surface_single_gpu(
    input_seq_dir, output_dir, visualize=False, n_bins=5, dt=0.01, **kwargs
):
    """
    GPU version (dtype-safe). Per window [t0,t1], per bin & polarity, keep the last
    event's normalized time (t - t0)/dt at (y,x).
    Writes: events/{dt:.4f}/time_surfaces_v2_{n_bins}/0{t1}.h5  (float32)
    """
    input_seq_dir = Path(input_seq_dir)
    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / split / input_seq_dir.stem
    out_root = output_seq_dir / "events" / f"{dt:.4f}" / f"time_surfaces_v2_gpu_{n_bins}"
    print(f"Output dir: {out_root}")
    out_root.mkdir(exist_ok=True, parents=True)

    dt_us = float(dt * 1e6)
    dt_us_bin = dt_us / float(n_bins)
    H, W = IMG_H, IMG_W

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(str(input_seq_dir / "events" / "events.h5"), "r") as h5f:
        # ---- Load & sort once (CPU) ----
        t_np = np.asarray(h5f["t"])
        order = np.argsort(t_np)
        # Cast to PyTorch-supported dtypes (IMPORTANT)
        t_np = t_np[order].astype(np.float32, copy=False)
        x_np = np.asarray(h5f["x"])[order].astype(np.int32,  copy=False)   # was uint16
        y_np = np.asarray(h5f["y"])[order].astype(np.int32,  copy=False)   # was uint16
        p_np = np.asarray(h5f["p"])[order].astype(np.int32,  copy=False)   # 0/1

        def slice_window(t0_us, t1_us):
            i0 = np.searchsorted(t_np, t0_us, side="left")
            i1 = np.searchsorted(t_np, t1_us, side="right")
            return i0, i1

        for t1 in np.arange(400000, 900000 + dt_us, dt_us):
            t0 = t1 - dt_us
            out_path = out_root / f"0{int(t1)}.h5"
            if out_path.exists():
                continue

            i0, i1 = slice_window(t0, t1)
            if i1 <= i0:
                empty = np.zeros((H, W, 2 * n_bins), dtype=np.float32)
                with h5py.File(out_path, "w") as h5f_out:
                    h5f_out.create_dataset(
                        "time_surface", data=empty, shape=empty.shape,
                        dtype=np.float32, **blosc_opts(complevel=1, shuffle="byte"),
                    )
                continue

            # ---- To GPU tensors (now that dtypes are supported) ----
            x = torch.from_numpy(x_np[i0:i1]).to(device=device, dtype=torch.long)
            y = torch.from_numpy(y_np[i0:i1]).to(device=device, dtype=torch.long)
            p = torch.from_numpy(p_np[i0:i1]).to(device=device, dtype=torch.long)  # 0/1
            t = torch.from_numpy(t_np[i0:i1]).to(device=device, dtype=torch.float32)

            # Bin per event & validity mask
            tb = torch.floor((t - t0) / dt_us_bin).to(torch.long)
            mask = (t > t0) & (t <= t1) & (tb >= 0) & (tb < n_bins)
            if mask.any():
                x = x[mask]; y = y[mask]; p = p[mask]; t = t[mask]; tb = tb[mask]
                # (optional) guard bounds
                inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                if not torch.all(inb):
                    x = x[inb]; y = y[inb]; p = p[inb]; t = t[inb]; tb = tb[inb]
                # Channel: 2*bin + polarity, value: normalized time in [0,1]
                ch   = tb * 2 + p
                vals = ((t - t0) / dt_us).clamp_(0.0, 1.0)

                # Flatten indices for scatter-reduce
                idx_flat = ch * (H * W) + y * W + x

                grid = torch.zeros((2 * n_bins * H * W), device=device, dtype=torch.float32)
                if hasattr(grid, "scatter_reduce_"):  # PyTorch >= 1.12
                    grid.scatter_reduce_(0, idx_flat, vals, reduce="amax", include_self=True)
                else:
                    # Fallback if scatter_reduce_ is unavailable
                    srt = torch.argsort(idx_flat)
                    idx_sorted = idx_flat[srt]
                    vals_sorted = vals[srt]
                    keep = torch.ones_like(idx_sorted, dtype=torch.bool)
                    keep[:-1] = idx_sorted[1:] != idx_sorted[:-1]
                    grid[idx_sorted[keep]] = vals_sorted[keep]

                grid = grid.view(2 * n_bins, H, W).permute(1, 2, 0).contiguous()
                time_surface = grid.detach().cpu().numpy()
            else:
                time_surface = np.zeros((H, W, 2 * n_bins), dtype=np.float32)

            with h5py.File(out_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "time_surface", data=time_surface, shape=time_surface.shape,
                    dtype=np.float32, **blosc_opts(complevel=1, shuffle="byte"),
                )

            if visualize:
                for i in range(n_bins):
                    img = (time_surface[:, :, 2 * i] * 255).astype(np.uint8)
                    cv2.imshow(f"Time Surface Bin {i}", img)
                    cv2.waitKey(1)


def generate_time_surface_single(
    input_seq_dir, output_dir, visualize=True, n_bins=5, dt=0.01, **kwargs
):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / split / input_seq_dir.stem
    #output_dir = output_seq_dir / "events" / "0.0200" / f"time_surfaces_v2_{n_bins}"
    output_dir = output_seq_dir / "events" / f"{dt:.4f}" / f"time_surfaces_v2_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)
    dt_us = dt * 1e6
    dt_us_bin = dt_us / n_bins

    with h5py.File(str(input_seq_dir / "events" / "events.h5"), "r") as h5f:
        time = np.asarray(h5f["t"])
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(h5f["x"])[idxs_sorted],
            np.asarray(h5f["y"])[idxs_sorted],
            np.asarray(h5f["p"])[idxs_sorted],
            np.asarray(h5f["t"])[idxs_sorted],
        )

        # dt of labels
        for t1 in np.arange(400000, 900000 + dt_us, dt_us):
            output_path = output_dir / f"0{t1}.h5"
            if output_path.exists():
                continue

            time_surface = np.zeros((IMG_H, IMG_W, n_bins * 2), dtype=np.uint64)
            t0 = t1 - dt_us

            # iterate over bins
            for i_bin in range(n_bins):
                t0_bin = t0 + i_bin * dt_us_bin
                t1_bin = t0_bin + dt_us_bin
                mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
                x_bin, y_bin, p_bin, t_bin = (
                    x[mask_t],
                    y[mask_t],
                    p[mask_t],
                    time[mask_t],
                )
                n_events = len(x_bin)
                for i in range(n_events):
                    time_surface[y_bin[i], x_bin[i], 2 * i_bin + int(p_bin[i])] = (
                        t_bin[i] - t0
                    )
            time_surface = np.divide(time_surface, dt_us)

            # Write to disk
            with h5py.File(output_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "time_surface",
                    data=time_surface,
                    shape=time_surface.shape,
                    dtype=np.float32,
                    **blosc_opts(complevel=1, shuffle="byte"),
                )
            # Visualize
            if visualize:
                for i in range(n_bins):
                    cv2.imshow(
                        f"Time Surface Bin {i}",
                        (time_surface[:, :, i] * 255).astype(np.uint8),
                    )
                    cv2.waitKey(0)


def generate_voxel_grid_single(input_seq_dir, output_dir, n_bins=5, dt=0.01, **kwargs):
    """
    For each ts in [0.4:0.02:0.9], generate the event count image
    :param seq_dir:
    :return:
    """
    input_seq_dir = Path(input_seq_dir)
    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / split / input_seq_dir.stem
    output_dir = output_seq_dir / "events" / f"{dt:.4f}" / f"voxel_grids_{n_bins}"
    output_dir.mkdir(exist_ok=True, parents=True)

    dt_us = dt * 1e6

    with h5py.File(str(input_seq_dir / "events" / "events.h5"), "r") as h5f:
        time = np.asarray(h5f["t"])
        idxs_sorted = np.argsort(time)
        x, y, p, time = (
            np.asarray(h5f["x"])[idxs_sorted],
            np.asarray(h5f["y"])[idxs_sorted],
            np.asarray(h5f["p"])[idxs_sorted],
            np.asarray(h5f["t"])[idxs_sorted],
        )

        # dt of labels
        for t1 in np.arange(400000, 900000 + dt_us, dt_us):
            output_path = output_dir / f"0{int(t1)}.h5"
            if output_path.exists():
                continue

            t0 = t1 - dt_us
            mask_t = np.logical_and(time > t0, time <= t1)
            x_bin, y_bin, p_bin, t_bin = x[mask_t], y[mask_t], p[mask_t], time[mask_t]
            curr_voxel_grid = events_to_voxel_grid(
                VOXEL_GRID_CONSTRUCTOR, p_bin, t_bin, x_bin, y_bin
            )
            curr_voxel_grid = curr_voxel_grid.numpy()
            curr_voxel_grid = np.transpose(curr_voxel_grid, (1, 2, 0))

            # Write to disk
            with h5py.File(output_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "voxel_grid",
                    data=curr_voxel_grid,
                    shape=curr_voxel_grid.shape,
                    dtype=np.float32,
                    **blosc_opts(complevel=1, shuffle="byte"),
                )

# ----------------------------- ATSLTD (arXiv:2002.05583) -----------------------------
from dataclasses import dataclass
import numpy as np
import math
import h5py
import cv2
from pathlib import Path
from utils.utils import blosc_opts


@dataclass(frozen=True)
class Event:
    u: int      # x (horizontal)
    v: int      # y (vertical)
    p: int      # polarity: 0/1 (Off/On)
    t: float    # timestamp (monotonic increasing)

def init_atsltd_frame(h: int, w: int, device=device) -> torch.Tensor:
    # Keep integer-valued state but store as float32 for fast in-place mul/round.
    # Values remain exact integers in [0,255] after every update (same as uint8 frame in numpy version).
    return torch.zeros((h, w, 2), device=device, dtype=torch.float32)

def atsltd_update_inplace(F: torch.Tensor, e_k: Event, t_prev: float) -> float:
    """
    Same Eq. (2) as before, just in Torch:
      F <- round(F * (t_prev / t_k))
      F[v_k, u_k, p_k] <- 255
    """
    t_k = float(e_k.t)

    if t_prev is None or t_k <= 0.0:
        decay = 1.0
    else:
        decay = t_prev / t_k

    # torch.round uses bankers rounding (ties-to-even), matching np.rint for our case.
    F.mul_(decay).round_()
    F[e_k.v, e_k.u, e_k.p] = 255.0

    return t_k

def patch_entropy_uint8(patch: np.ndarray) -> float:
    """
    patch: (r, r) uint8 with values in [0,255]
    Implements Eq. (4): -sum_z p(z) log p(z)
    """
    hist = np.bincount(patch.reshape(-1), minlength=256).astype(np.float64)
    prob = hist / hist.sum()

    nz = prob > 0
    return float(-(prob[nz] * np.log(prob[nz])).sum())

def nzge_atsltd(F: torch.Tensor, r: int = 4) -> float:
    """
    Same Eq. (3)-(4) as before, but computed on GPU.
    F: (H,W,2) float32 with integer values 0..255
    """
    H, W, C = F.shape
    assert C == 2
    assert H % r == 0 and W % r == 0
    p = H // r
    q = W // r
    P = p * q
    L = r * r

    # Convert to uint8 for histogram bins 0..255 (exact, since F values are integers)
    Fu8 = F.to(torch.uint8)                         # (H,W,2)
    Fu8 = Fu8.permute(2, 0, 1).contiguous()         # (2,H,W)

    # Non-overlapping r×r patches -> (2, p, q, r, r) -> (2, P, L)
    patches = Fu8.view(2, p, r, q, r).permute(0, 1, 3, 2, 4).contiguous()
    patches = patches.view(2, P, L)                 # uint8

    def entropy_from_patches(pch: torch.Tensor) -> torch.Tensor:
        # pch: (P, L) uint8
        idx = pch.to(torch.long)                    # (P, L)
        hist = torch.zeros((P, 256), device=idx.device, dtype=torch.int32)
        ones = torch.ones_like(idx, dtype=torch.int32)
        hist.scatter_add_(1, idx, ones)             # per-patch histogram

        prob = hist.to(torch.float64) / float(L)    # (P,256)
        logp = torch.zeros_like(prob)
        mask = prob > 0
        logp[mask] = torch.log(prob[mask])
        ent = -(prob * logp).sum(dim=1)             # (P,)
        return ent

    ent0 = entropy_from_patches(patches[0])
    ent1 = entropy_from_patches(patches[1])
    ent = 0.5 * (ent0 + ent1)                       # (P,)

    nonzero = ent > 0
    if not torch.any(nonzero):
        return 0.0

    return float(ent[nonzero].mean().item())

def nzge_confidence_interval(C_bar: float, S_C: float, n_s: int, t_crit: float) -> tuple[float, float]:
    """
    Implements Eq. (8) given:
      C_bar = sample mean, S_C = sample std, n_s = number of samples,
      t_crit = |t_{omega/2}| (e.g., 1.984 for n_s=100, omega=0.05 in the paper)
    """
    margin = t_crit * (S_C / math.sqrt(n_s))
    return (C_bar - margin, C_bar + margin)

def should_finalize_frame(F: np.ndarray, alpha: float, beta: float, r: int = 4) -> bool:
    val = nzge_atsltd(F, r=r)
    return (alpha <= val <= beta)

def _make_unique_path(path: Path) -> Path:
    """Avoid collisions if two frames finalize at the same timestamp."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    k = 1
    while True:
        cand = path.with_name(f"{stem}_{k}{suffix}")
        if not cand.exists():
            return cand
        k += 1

def _calibrate_ci_from_stream(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    H: int,
    W: int,
    r: int,
    n_samples: int,
    sample_every_events: int,
    t_crit: float,
) -> tuple[float, float, int]:
    F = init_atsltd_frame(H, W, device=device)
    t_prev = None
    C = []

    N = len(t)
    i = 0
    while i < N and len(C) < n_samples:
        e = Event(u=int(x[i]), v=int(y[i]), p=int(p[i]), t=float(t[i]))
        t_prev = atsltd_update_inplace(F, e, t_prev)

        if (i + 1) % sample_every_events == 0:
            C.append(nzge_atsltd(F, r=r))

        i += 1

    if len(C) == 0:
        return 0.0, 0.0, i

    C = np.asarray(C, dtype=np.float64)
    C_bar = float(C.mean())
    S_C = float(C.std(ddof=1)) if len(C) > 1 else 0.0
    alpha, beta = nzge_confidence_interval(C_bar, S_C, max(len(C), 1), t_crit=t_crit)
    return float(alpha), float(beta), i

def generate_atsltd_single(
    input_seq_dir,
    output_dir,
    visualize=False,
    # ATSLTD params
    r=4,
    alpha=None,
    beta=None,
    n_samples=100,
    t_crit=1.984,
    sample_every_events=5000,
    check_every_events=5000,
    # to match your Multiflow processing interval (microseconds)
    start_us=400000,
    end_us=900000,
    **kwargs,
):
    """
    Generate ATSLTD frames asynchronously (paper Eq. (2) + NZGE trigger).
    Reads:  input_seq_dir/events/events.h5 with datasets {x,y,p,t}
    Writes: output_dir/split/seq/events/atsltd_r{r}/0{t_end}.h5 (uint8, HxWx2)
           Each file stores dataset "atsltd" and attrs {alpha,beta,nzge,t_end_us}.
    """
    input_seq_dir = Path(input_seq_dir)
    output_dir = Path(output_dir)

    split = input_seq_dir.parents[0].stem
    output_seq_dir = output_dir / split / input_seq_dir.stem
    out_root = output_seq_dir / "events" / f"atsltd_r{int(r)}"
    out_root.mkdir(exist_ok=True, parents=True)

    H, W = IMG_H, IMG_W
    assert H % r == 0 and W % r == 0, f"IMG_H and IMG_W must be divisible by r={r}"

    with h5py.File(str(input_seq_dir / "events" / "events.h5"), "r") as h5f:
        # load + sort by time (like your other functions)
        t_all = np.asarray(h5f["t"])
        order = np.argsort(t_all)
        t_all = t_all[order].astype(np.float64, copy=False)
        x_all = np.asarray(h5f["x"])[order].astype(np.int32, copy=False)
        y_all = np.asarray(h5f["y"])[order].astype(np.int32, copy=False)
        p_all = np.asarray(h5f["p"])[order].astype(np.int32, copy=False)

        # restrict to desired time span (microseconds)
        i0 = np.searchsorted(t_all, float(start_us), side="left")
        i1 = np.searchsorted(t_all, float(end_us), side="right")
        if i1 <= i0:
            return

        t = t_all[i0:i1]
        x = x_all[i0:i1]
        y = y_all[i0:i1]
        p = p_all[i0:i1]

        # bounds guard
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (p >= 0) & (p <= 1)
        if not np.all(inb):
            x = x[inb]; y = y[inb]; p = p[inb]; t = t[inb]
            if len(t) == 0:
                return

        # ---------------- CI calibration (if alpha/beta not provided) ----------------
        if alpha is None or beta is None:
            alpha, beta, next_idx = _calibrate_ci_from_stream(
                x=x, y=y, p=p, t=t, H=H, W=W, r=int(r),
                n_samples=int(n_samples),
                sample_every_events=int(sample_every_events),
                t_crit=float(t_crit),
            )
        else:
            alpha = float(alpha)
            beta = float(beta)
            next_idx = 0

        # ---------------- Main ATSLTD generation ----------------
        F = init_atsltd_frame(H, W)
        
        t_prev = None
        frame_count = 0

        # for visualization throttling
        vis_every = max(int(check_every_events), 1)

        for i in range(next_idx, len(t)):
            e = Event(u=int(x[i]), v=int(y[i]), p=int(p[i]), t=float(t[i]))
            t_prev = atsltd_update_inplace(F, e, t_prev)

            # Check NZGE only every N events (computing NZGE is expensive)
            if (i + 1) % int(check_every_events) != 0:
                continue

            nz = nzge_atsltd(F, r=int(r))
            if alpha <= nz <= beta:
                t_end = int(round(t_prev)) if t_prev is not None else int(round(t[i]))
                out_path = _make_unique_path(out_root / f"0{t_end}.h5")

                F_out = F.to(torch.uint8).cpu().numpy()  # (H, W, 2) uint8 on CPU

                with h5py.File(out_path, "w") as h5f_out:
                    h5f_out.create_dataset(
                        "atsltd",
                        data=F_out,
                        shape=F_out.shape,
                        dtype=np.uint8,
                        **blosc_opts(complevel=1, shuffle="byte"),
                    )
                    # store metadata
                    h5f_out.attrs["alpha"] = float(alpha)
                    h5f_out.attrs["beta"] = float(beta)
                    h5f_out.attrs["nzge"] = float(nz)
                    h5f_out.attrs["t_end_us"] = int(t_end)
                    h5f_out.attrs["r"] = int(r)
                    h5f_out.attrs["frame_index"] = int(frame_count)

                frame_count += 1

                if visualize:
                    vis_dir = Path("vis")
                    vis_dir.mkdir(exist_ok=True, parents=True)

                    Fu8 = F.to(torch.uint8)  # (H,W,2) on GPU
                    vis = torch.cat([Fu8[:, :, 0], Fu8[:, :, 1]], dim=1).cpu().numpy()  # (H, 2W) uint8

                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                    cv2.putText(
                        vis_bgr, f"t={t_end} us", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                    )

                    img_path = _make_unique_path(vis_dir / f"0{t_end}.png")
                    cv2.imwrite(str(img_path), vis_bgr)

                # start next Fi+1
                F = init_atsltd_frame(H, W)
                t_prev = None

# --------------------------- end ATSLTD -----------------------------------


def generate(
    input_dir,
    output_dir,
    representation_type,
    dts=(0.01, 0.02),
    n_bins=5,
    visualize=False,
    **kwargs,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if representation_type == "time_surface":
        generation_function = generate_time_surface_single
    elif representation_type == "time_surface_gpu":
        generation_function = generate_time_surface_single_gpu
    elif representation_type == "voxel_grid":
        generation_function = generate_voxel_grid_single
    elif representation_type == "event_stack":
        generation_function = generate_sbt_single
    elif representation_type == "event_count":
        generation_function = generate_event_count_images_single
    elif representation_type == "atsltd":
        generation_function = generate_atsltd_single
    else:
        raise NotImplementedError(f"No generation function for {representation_type}")

    for split in ["train", "test"]:
        split_dir = input_dir / split
        n_seqs = len(os.listdir(str(split_dir)))
        print(f"Generate representations for {split}")

        for input_seq_dir in tqdm(split_dir.iterdir(), total=n_seqs):
            if representation_type == "atsltd":
                    generation_function(
                        input_seq_dir, output_dir, visualize=visualize, **kwargs)
            else:
                for dt in dts:
                    generation_function(
                        input_seq_dir, output_dir, visualize=visualize, n_bins=n_bins, dt=dt, **kwargs
                    )


if __name__ == "__main__":
    fire.Fire(generate)
