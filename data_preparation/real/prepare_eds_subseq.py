import os
from pathlib import Path
from shutil import copy

import fire
import h5py
import hdf5plugin  # noqa: F401  (needed for blosc filters)
import numpy as np
from tqdm import tqdm

from utils.utils import blosc_opts

# Where eval_real expects the EDS subsequences (override via output_root)
EDS_SUBSEQ_ROOT = Path("/path/to/eds_subseq")

# EDS resolution
IMG_H = 480
IMG_W = 640


def generate_subseq(root_dir, seq_name, start_idx, end_idx, dt, output_root=None):
    """
    Prepare an EDS subsequence with 10-bin time_surfaces_v2:

        <output_root>/<base_name>_<start>_<end>/
            images_corrected/
                frame_0000000000.png  (original start_idx)
                ...
            images_timestamps.txt    (cropped [start_idx:end_idx])
            stamped_groundtruth.txt
            events/
                <dt>/
                    time_surfaces_v2_10/
                        0000000.h5
                        0005000.h5
                        ...
                        <(n_events-1)*dt_us>.h5
            events_corrected.h5      (cropped events)

    Arguments:
        root_dir    : path to EDS raw root
        seq_name    : raw sequence name, e.g. "01_peanuts_light"
        start_idx   : starting frame index (inclusive)
        end_idx     : ending frame index (exclusive)
        dt          : time delta in seconds (e.g. 0.005)
        output_root : destination for prepared subsequences (defaults to EDS_SUBSEQ_ROOT)
    """

    root_dir = Path(root_dir)
    dest_root = Path(output_root) if output_root is not None else EDS_SUBSEQ_ROOT
    input_dir = root_dir / seq_name

    # Strip numeric prefix "01_" → "peanuts_light"
    if len(seq_name) > 3 and seq_name[:2].isdigit() and seq_name[2] == "_":
        base_name = seq_name[3:]
    else:
        base_name = seq_name

    # Output directory that eval_real uses
    output_dir = dest_root / f"{base_name}_{start_idx}_{end_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EDS] Input dir : {input_dir}")
    print(f"[EDS] Output dir: {output_dir}")

    # ------------------------------------------------------------------
    # 1) Copy pose data
    # ------------------------------------------------------------------
    copy(
        str(input_dir / "stamped_groundtruth.txt"),
        str(output_dir / "stamped_groundtruth.txt"),
    )

    # ------------------------------------------------------------------
    # 2) Filter and save image timestamps (they are in microseconds)
    # ------------------------------------------------------------------
    image_timestamps_full = np.genfromtxt(str(input_dir / "images_timestamps.txt"))
    image_timestamps = image_timestamps_full[start_idx:end_idx]
    np.savetxt(str(output_dir / "images_timestamps.txt"), image_timestamps, fmt="%i")

    # Convenience vars
    first_ts = int(image_timestamps[0])
    last_ts = int(image_timestamps[-1])

    # ------------------------------------------------------------------
    # 3) Copy images to images_corrected/ and reindex from 0
    # ------------------------------------------------------------------
    output_image_dir = output_dir / "images_corrected"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(start_idx, end_idx), desc="Copying images..."):
        src = input_dir / "images_corrected" / f"frame_{str(idx).zfill(10)}.png"
        dst = output_image_dir / f"frame_{str(idx - start_idx).zfill(10)}.png"
        copy(str(src), str(dst))

    # ------------------------------------------------------------------
    # 4) Generate time_surfaces_v2_10 aligned with EDSSubseq indexing
    # ------------------------------------------------------------------
    n_bins = 5
    dt_us = int(round(dt * 1e6))  # microseconds
    dt_bin_us = dt_us / n_bins

    # SAME formula as EDSSubseq: n_events = ceil((t_end - t_init)/dt)
    delta_us = last_ts - first_ts
    n_events = int(np.ceil(delta_us / dt_us))

    # We must generate indices i = 0 .. n_events-1
    # EDSSubseq.events() will request files with indices
    # current_idx = 1..n_events-1  → filename = current_idx * dt_us
    output_ts_dir = (
        output_dir / "events" / f"{dt:.4f}" / f"time_surfaces_v2_{n_bins}_"
    )
    output_ts_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(input_dir / "events_corrected.h5"), "r") as h5f:
        time = np.asarray(h5f["t"])  # microseconds
        x_all = np.asarray(h5f["x"])
        y_all = np.asarray(h5f["y"])
        p_all = np.asarray(h5f["p"])

        for i in tqdm(range(n_events), desc="Generating time surfaces..."):
            # This window covers [t0, t0 + dt_us)
            t0 = first_ts + i * dt_us
            t1 = t0 + dt_us

            out_path = output_ts_dir / f"{str(int(i * dt_us)).zfill(7)}.h5"
            if out_path.exists():
                continue

            # H x W x (2*n_bins) (polarity split)
            time_surface = np.zeros((IMG_H, IMG_W, 2 * n_bins), dtype=np.float32)

            # For each bin, write last timestamp-normalized value
            for i_bin in range(n_bins):
                t0_bin = t0 + i_bin * dt_bin_us
                t1_bin = t0_bin + dt_bin_us

                first_idx = np.searchsorted(time, t0_bin, side="left")
                last_idx_p1 = np.searchsorted(time, t1_bin, side="right")

                x = np.rint(x_all[first_idx:last_idx_p1]).astype(int)
                y = np.rint(y_all[first_idx:last_idx_p1]).astype(int)
                p = p_all[first_idx:last_idx_p1].astype(int)
                t_ev = time[first_idx:last_idx_p1]

                n_ev = x.shape[0]
                for k in range(n_ev):
                    if 0 <= x[k] < IMG_W and 0 <= y[k] < IMG_H:
                        ch = 2 * i_bin + int(p[k])
                        # normalize to [0,1] using the *window* start t0
                        time_surface[y[k], x[k], ch] = (t_ev[k] - t0) / dt_us

            with h5py.File(out_path, "w") as h5f_out:
                h5f_out.create_dataset(
                    "time_surface",
                    data=time_surface,
                    shape=time_surface.shape,
                    dtype=np.float32,
                    **blosc_opts(complevel=1, shuffle="byte"),
                )

        # ------------------------------------------------------------------
        # 5) Store cropped events for this subsequence
        # ------------------------------------------------------------------
        event_idx = np.searchsorted(time, np.asarray([first_ts, last_ts]), side="left")

        events_out_path = output_dir / "events_corrected.h5"
        with h5py.File(events_out_path, "w") as h5f_out:
            h5f_out.create_dataset(
                "x", data=x_all[event_idx[0] : event_idx[1]].astype(np.uint16)
            )
            h5f_out.create_dataset(
                "y", data=y_all[event_idx[0] : event_idx[1]].astype(np.uint16)
            )
            h5f_out.create_dataset("p", data=p_all[event_idx[0] : event_idx[1]])
            h5f_out.create_dataset("t", data=time[event_idx[0] : event_idx[1]])


if __name__ == "__main__":
    fire.Fire(generate_subseq)
