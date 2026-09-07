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

from utils.representations import VoxelGrid, events_to_voxel_grid
from utils.utils import blosc_opts

# ========= NZGE helpers (paper params) =========
NZGE_P, NZGE_Q, NZGE_R = 45, 60, 4
_H_NZ, _W_NZ = NZGE_P * NZGE_R, NZGE_Q * NZGE_R  # 180x240

def _entropy_uint8_block(arr_uint8: np.ndarray) -> float:
    counts = np.bincount(arr_uint8.reshape(-1), minlength=256).astype(np.float64)
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts / (tot + 1e-12)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])))

def _atsltd_nzge(frame_uint8: np.ndarray) -> float:
    """
    frame_uint8: (H,W,2), uint8
    NZGE with (p,q,r)=(45,60,4) per paper.
    """
    ch0 = cv2.resize(frame_uint8[..., 0], (_W_NZ, _H_NZ), interpolation=cv2.INTER_AREA)
    ch1 = cv2.resize(frame_uint8[..., 1], (_W_NZ, _H_NZ), interpolation=cv2.INTER_AREA)
    ch0 = ch0.reshape(NZGE_P, NZGE_R, NZGE_Q, NZGE_R)
    ch1 = ch1.reshape(NZGE_P, NZGE_R, NZGE_Q, NZGE_R)
    entropies = []
    for i in range(NZGE_P):
        for j in range(NZGE_Q):
            e0 = _entropy_uint8_block(ch0[i, :, j, :])
            e1 = _entropy_uint8_block(ch1[i, :, j, :])
            e = 0.5 * (e0 + e1)
            if e > 0.0:
                entropies.append(e)
    return float(np.mean(entropies)) if entropies else 0.0


# ========= Calibration: compute a,b (alpha,beta) and all stats =========
def calibrate_atsltd_ci(
    input_dir,
    output_dir=None,
    splits=("train", "test"),
    dt=0.02,                 # window length for sampling (s) — just for sampling, not for async gen
    n_samples=100,           # target number of NZGE samples
    t_start_s=0.4,           # sampling interval start (s)
    t_end_s=0.9,             # sampling interval end (s)
    min_nonzero_frac=0.005,  # quick gates to prefer clear/contour frames
    max_nonzero_frac=0.25,
    min_edge_ratio=0.01,
    max_edge_ratio=0.15,
    seed=0,
):
    """
    Compute NZGE confidence interval [alpha, beta] as in Eqs. (5)-(8) of the paper.
    Outputs: mean (Cbar), std (S_C), tcrit (t_{0.975}), alpha, beta, n_s, and raw samples.

    The sampling builds ATSLTD frames over random fixed windows [t1-dt, t1] for
    stability and speed, using a GPU vectorized last-timestamp map.

    Saves JSON next to the dataset (or in output_dir if provided).
    """
    import json, torch
    rng = np.random.default_rng(seed)

    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ATSLTD_MAX_VAL = 255.0
    dt_us = int(dt * 1e6)
    t0_us = int(t_start_s * 1e6)
    t1_us = int(t_end_s * 1e6)

    # collect sequences with events.h5
    seqs = []
    for sp in splits:
        cand = input_dir / sp
        if not cand.exists():
            continue
        for d in cand.iterdir():
            if (d / "events" / "events.h5").exists():
                seqs.append(d)
    seqs = sorted(seqs)
    if not seqs:
        print("[calib] No sequences found.")
        return

    print(f"[calib] Found {len(seqs)} sequences. Target samples: {n_samples}")

    def _edge_ratio(gray_uint8: np.ndarray) -> float:
        v = np.median(gray_uint8)
        lo = int(max(0, 0.66 * v))
        hi = int(min(255, 1.33 * v))
        edges = cv2.Canny(gray_uint8, lo, hi)
        return float((edges > 0).mean())

    nzges = []
    tried = 0
    max_tries = n_samples * 50

    while len(nzges) < n_samples and tried < max_tries:
        tried += 1
        seq = seqs[rng.integers(0, len(seqs))]
        with h5py.File(str(seq / "events" / "events.h5"), "r") as h5f:
            t_np = np.asarray(h5f["t"])
            if t_np.size == 0:
                continue
            order = np.argsort(t_np)
            x_np = np.asarray(h5f["x"])[order].astype(np.int64)
            y_np = np.asarray(h5f["y"])[order].astype(np.int64)
            p_np = np.asarray(h5f["p"])[order].astype(np.int64)
            t_np = t_np[order].astype(np.int64)

        # random anchor in [0.4,0.9]s
        t1_anchor = int(rng.integers(t0_us, t1_us + 1))
        t0 = t1_anchor - dt_us

        # select events in window
        t_all = torch.from_numpy(t_np).to(device)
        i0 = int(torch.searchsorted(t_all, torch.tensor(t0, device=device), right=False))
        i1 = int(torch.searchsorted(t_all, torch.tensor(t1_anchor, device=device), right=True))
        if i1 <= i0:
            continue

        x = torch.from_numpy(x_np[i0:i1]).to(device)
        y = torch.from_numpy(y_np[i0:i1]).to(device)
        pp = torch.from_numpy(p_np[i0:i1]).to(device)
        tt = torch.from_numpy(t_np[i0:i1]).to(device)

        # last-timestamp map (vectorized)
        lin = (y * IMG_W + x) * 2 + pp
        last_flat = torch.zeros(IMG_H * IMG_W * 2, dtype=torch.float32, device=device)
        last_flat.scatter_reduce_(0, lin, tt.to(torch.float32), reduce="amax", include_self=True)
        last = last_flat.view(IMG_H, IMG_W, 2)

        denom = max(t1_anchor - t0, 1)
        frame = (last - float(t0)).clamp_min(0.0) * (ATSLTD_MAX_VAL / float(denom))
        F_uint8 = frame.clamp(0, 255).to(torch.uint8).detach().cpu().numpy()

        # gates to prefer “clear & sharp contours”
        combined = np.maximum(F_uint8[..., 0], F_uint8[..., 1])
        nonzero_frac = float((combined > 0).mean())
        if not (min_nonzero_frac <= nonzero_frac <= max_nonzero_frac):
            continue

        gray = cv2.cvtColor(np.stack([F_uint8[..., 1], F_uint8[..., 0], np.zeros_like(F_uint8[..., 0])], axis=-1),
                            cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (256, 192), interpolation=cv2.INTER_AREA)
        er = _edge_ratio(gray_small)
        if not (min_edge_ratio <= er <= max_edge_ratio):
            continue

        nzges.append(_atsltd_nzge(F_uint8))

    nzges = np.array(nzges, dtype=np.float64)
    n_s = int(nzges.size)
    if n_s == 0:
        print("[calib] No samples passed the gates; relax thresholds.")
        return

    Cbar = float(nzges.mean())
    S_C = float(nzges.std(ddof=1)) if n_s > 1 else 0.0

    # t-critical (two-sided 95%)
    try:
        from scipy.stats import t as student_t
        tcrit = float(student_t.ppf(0.975, df=max(n_s - 1, 1)))
    except Exception:
        # normal approx fallback
        tcrit = 1.96

    half_w = tcrit * S_C / max(np.sqrt(n_s), 1.0)
    alpha = Cbar - half_w
    beta = Cbar + half_w

    # save & print
    out = {
        "dataset": "multiflow",
        "grid": {"p": NZGE_P, "q": NZGE_Q, "r": NZGE_R},
        "image_size": {"H": IMG_H, "W": IMG_W},
        "sampling": {
            "dt": dt,
            "t_window_s": [t_start_s, t_end_s],
            "n_samples": n_s,
            "gates": {
                "min_nonzero_frac": min_nonzero_frac,
                "max_nonzero_frac": max_nonzero_frac,
                "min_edge_ratio": min_edge_ratio,
                "max_edge_ratio": max_edge_ratio,
            },
        },
        "stats": {
            "Cbar": Cbar,
            "S_C": S_C,
            "tcrit_0_975": tcrit,
            "alpha": alpha,   # a
            "beta": beta,     # b
        },
        "samples": nzges.tolist(),
    }

    save_path = Path(output_dir) / "atsltd_nzge_calibration.json"
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== ATSLTD NZGE calibration ===")
    print(f"n_s      = {n_s}")
    print(f"Cbar     = {Cbar:.6f}")
    print(f"S_C      = {S_C:.6f}")
    print(f"tcrit    = {tcrit:.6f}")
    print(f"alpha(a) = {alpha:.6f}")
    print(f"beta(b)  = {beta:.6f}")
    print(f"saved to {save_path}")
