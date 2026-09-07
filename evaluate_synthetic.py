"""Evaluate tracker on Multiflow synthetic dataset.

Structure intentionally mirrors evaluate_real.py, with only necessary
changes for the Multiflow layout and GT format.
"""

import logging
import os
from pathlib import Path

import hydra
import imageio
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from prettytable import PrettyTable
from tqdm import tqdm
import sys
import cv2
import h5py

from utils.dataset import (
    CornerConfig,
    SequenceDataset,
)
from utils.timers import CudaTimer, cuda_timers
from utils.track_utils import (
    TrackObserver,
    get_gt_corners,
)
from utils.utils import read_input
from utils.visualization import (
    generate_track_colors,
    render_pred_tracks,
)

# ----------------- Logging -----------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    logger.addHandler(_h)

results_table = PrettyTable()
results_table.field_names = ["Sequence", "Inference Time (s)"]

# ----------------- Paths -------------------

MULTIFLOW_DATA_DIR = Path("/path/to/multiflow")
MULTIFLOW_EXTRA_DIR = Path("/path/to/multiflow_reloaded_extra")
GT_FILENAME = "shitomasi_custom_v5.gt.txt"

# Same corner config as evaluate_real
corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)


# ----------------- Helpers -----------------


def _parse_num_stem(path: Path) -> int:
    """
    Return integer from a file stem that may be like '0400000' or '0400000.0'.
    """
    stem = path.stem
    try:
        return int(stem)
    except ValueError:
        return int(float(stem))


# ----------------- Multiflow Dataset -----------------


class MultiflowSequence(SequenceDataset):
    """
    Synthetic Multiflow evaluation dataset.

    Matches training + real-eval convention:
      x = [events_representation (C channels), grayscale_ref (1 channel)]
    where grayscale_ref is taken from the first frame (0400000.png),
    and keypoints are overridden from GT.
    """

    def __init__(
        self,
        sequence_dir_extra: Path,
        data_dir_root: Path,
        patch_size: int,
        representation: str,
        corner_config: CornerConfig,
    ):
        super().__init__()

        self.sequence_dir_extra = sequence_dir_extra
        self.sequence_name = sequence_dir_extra.name  # e.g. 001dc..._done
        self.data_sequence_dir = data_dir_root / "test" / self.sequence_name

        self.patch_size = patch_size
        self.representation = representation
        self.corner_config = corner_config

        # Time convention (Multiflow fixed window)
        self.t_init = 400000 / 1e6
        self.t_now = self.t_init

        # First frame (reference)
        img_first_path = self.data_sequence_dir / "images" / "0400000.png"
        self.frame_first = cv2.imread(str(img_first_path), cv2.IMREAD_GRAYSCALE)
        if self.frame_first is None:
            raise FileNotFoundError(
                f"Could not read first frame: {img_first_path}"
            )

        # Choose events directory (prefer 0.0100, else 0.0200) for this representation
        ev_root = self.sequence_dir_extra / "events"
        self.ev_rep_dir = None
        self.dt = None

        for dt_str in ["0.0100", "0.0200"]:
            cand = ev_root / dt_str / self.representation
            if cand.exists():
                self.ev_rep_dir = cand
                self.dt = float(dt_str)
                break

        if self.ev_rep_dir is None:
            raise FileNotFoundError(
                f"No events found for representation '{self.representation}' "
                f"under {ev_root} (looked for 0.0100/ and 0.0200)."
            )

        # Collect event paths in [400000, 900000]
        all_event_paths = sorted(self.ev_rep_dir.glob("*.h5"))
        self.event_paths = [
            p
            for p in all_event_paths
            if 400000 <= _parse_num_stem(p) <= 900000
        ]
        if len(self.event_paths) < 2:
            raise RuntimeError(
                f"Not enough event slices for {self.sequence_name} "
                f"under {self.ev_rep_dir}"
            )

        # n_events follows evaluate_real semantics:
        # events() will yield (n_events - 1) steps.
        self.n_events = len(self.event_paths)

        # For visualization: predefine number of frames in [0.4, 0.9] step 0.05
        self.n_frames = 11

        # Initialize keypoints (Harris) + x_ref; will be overridden by GT later.
        self.initialize()

    # --- SequenceDataset abstract impls ---

    def initialize_reference_patches(self):
        """
        Build reference patches from the first grayscale frame.

        This matches training & real eval:
          - reference_encoder expects 1 channel
          - target_encoder sees C event channels
        """

        # Event channels per patch (used later to know C_events)
        if "grayscale" in self.representation:
            self.channels_in_per_patch = 1
        else:
            # e.g. time_surfaces_v2_5 or time_surfaces_v2_gpu_5
            self.channels_in_per_patch = int(self.representation[-1])
            if "v2" in self.representation:
                self.channels_in_per_patch *= 2

        # (1,1,H,W) float32
        ref_arr = self.frame_first.astype(np.float32) / 255.0
        ref_tensor = (
            torch.from_numpy(ref_arr).unsqueeze(0).unsqueeze(0)
        )  # (1,1,H,W)

        # One patch per keypoint → (n_tracks, 1, P, P)
        self.x_ref = self.get_patches(ref_tensor)

    def events(self):
        """
        Yield (t, x) where:
          - t is current time in seconds
          - x has shape (n_tracks, C_events + 1, P, P)
            [0:C_events]   = event representation
            [C_events:...] = fixed grayscale ref patch
        """
        # Ensure we start from the first *after* the initial timestamp.
        # event_paths[0] is typically 0400000.h5 → used implicitly as ref range.
        # We iterate from the second file onward.
        for p in self.event_paths[1:]:
            t_us = _parse_num_stem(p)
            self.t_now = t_us / 1e6

            ev_arr = read_input(str(p), self.representation)
            # (n_tracks, C_events, P, P)
            x_cur = self.get_patches_new(ev_arr)

            if self.x_ref.device != x_cur.device:
                self.x_ref = self.x_ref.to(x_cur.device)

            # Concatenate events + 1ch ref
            x = torch.cat([x_cur, self.x_ref], dim=1)

            yield self.t_now, x

    def frames(self):
        """
        Frames for visualization: 0400000.png ... 0900000.png every 0.05s
        """
        for i in range(self.n_frames):
            ts_us = 400000 + i * 50000
            t = ts_us / 1e6
            img_path = (
                self.data_sequence_dir / "images" / f"{ts_us:07d}.png"
            )
            frame = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                break
            yield t, frame

    # get_next unused; SequenceDataset.accumulate_y_hat is used by evaluate()


# ----------------- Core evaluation -----------------


def evaluate(model, sequence_dataset, dt_track_vis, sequence_name, visualize):
    """
    Run tracking on one Multiflow synthetic sequence and store predictions.
    Mirrors evaluate_real.evaluate.
    """
    tracks_pred = TrackObserver(
        t_init=sequence_dataset.t_init,
        u_centers_init=sequence_dataset.u_centers,
    )

    model.reset(sequence_dataset.n_tracks)
    event_generator = sequence_dataset.events()

    cuda_timer = CudaTimer(model.device, sequence_dataset.sequence_name)

    with torch.no_grad():
        # Predict tracks over event slices
        for t, x in tqdm(
            event_generator,
            total=sequence_dataset.n_events - 1,
            desc=f"[{sequence_name}] Predicting tracks...",
        ):
            with cuda_timer:
                x = x.to(model.device)
                y_hat = model(x)
                sequence_dataset.accumulate_y_hat(y_hat)

            tracks_pred.add_observation(
                t, sequence_dataset.u_centers.cpu().numpy()
            )

        # Optional visualization
        if visualize:
            gif_img_arr = []
            tracks_pred_interp = tracks_pred.get_interpolators()
            track_colors = generate_track_colors(sequence_dataset.n_tracks)

            for t, img_now in tqdm(
                sequence_dataset.frames(),
                total=sequence_dataset.n_frames,
                desc=f"[{sequence_name}] Rendering predicted tracks...",
            ):
                fig_arr = render_pred_tracks(
                    tracks_pred_interp,
                    t,
                    img_now,
                    track_colors,
                    dt_track=dt_track_vis,
                )
                gif_img_arr.append(fig_arr)

            imageio.mimsave(
                f"{sequence_name}_tracks_pred.gif", gif_img_arr
            )

    # Save predicted tracks: (track_id, t, x, y)
    # ensure per-sequence dir
    out_dir = Path(os.environ.get("RESULTS_ROOT", ".")) / sequence_name / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{sequence_name}.txt"
    np.savetxt(
        out_path,
        tracks_pred.track_data,
        fmt=["%i", "%.9f", "%i", "%i"],
        delimiter=" ",
    )
    logger.info(f"Saved predictions to: {out_path}")

    metrics = {
        "latency": float(sum(cuda_timers.get(sequence_dataset.sequence_name, [])))
    }
    return metrics


# ----------------- Hydra entrypoint -----------------


@hydra.main(config_path="configs", config_name="eval_real_defaults")
def track(cfg):
    pl.seed_everything(1234)
    OmegaConf.set_struct(cfg, True)

    # Ensure model.representation follows cfg.representation
    with open_dict(cfg):
        cfg.model.representation = cfg.representation

    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Use Hydra output dir (same behavior as evaluate_real)
    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        run_dir = Path.cwd()

    logger.info(f"PRED_DIR: {run_dir}")
    print(f"PRED_DIR={run_dir}")

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    state = torch.load(cfg.weights_path, map_location="cuda:0")
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Discover Multiflow test sequences
    test_root = MULTIFLOW_EXTRA_DIR / "test"
    if not test_root.exists():
        raise FileNotFoundError(
            f"Multiflow extra test dir not found: {test_root}"
        )

    seq_dirs = sorted(
        d
        for d in test_root.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != ".cache"
    )

    MAX_TOTAL_TRACKS = 3000
    total_tracks = 0

    for seq_dir in seq_dirs:
        if total_tracks >= MAX_TOTAL_TRACKS:
            logger.info(f"Reached {MAX_TOTAL_TRACKS} total tracks. Stopping evaluation.")
            break

        seq_name = seq_dir.name
        logger.info(f"=== SEQUENCE: {seq_name} ===")

        # Skip sequences without GT
        gt_path = seq_dir / "tracks" / GT_FILENAME
        if not gt_path.exists():
            logger.warning(f"[SKIP] No GT file for sequence {seq_name}: {gt_path}")
            continue
        try:
            gt_start_corners = get_gt_corners(str(gt_path))
        except Exception as e:
            logger.warning(f"[SKIP] Failed to read GT for {seq_name}: {e}")
            continue

        # Build dataset
        dataset = MultiflowSequence(
            sequence_dir_extra=seq_dir,
            data_dir_root=MULTIFLOW_DATA_DIR,
            patch_size=cfg.patch_size,
            representation=cfg.representation,
            corner_config=corner_config,
        )

        dataset.override_keypoints(gt_start_corners)

        # --- Subsample if needed to respect 3k global cap ---
        remaining = MAX_TOTAL_TRACKS - total_tracks
        if dataset.n_tracks > remaining:
            torch.manual_seed(1234)
            perm = torch.randperm(dataset.n_tracks)[:remaining]
            dataset.u_centers = dataset.u_centers[perm]
            dataset.u_centers_init = dataset.u_centers_init[perm]
            dataset.n_tracks = remaining
            if dataset.x_ref is not None:
                dataset.x_ref = dataset.x_ref[perm]
            logger.info(f"Subsampled {seq_name} tracks to {remaining} (global cap {MAX_TOTAL_TRACKS})")

        # Skip empty sequences
        if dataset.n_tracks == 0:
            continue

        total_tracks += dataset.n_tracks

        # Evaluate
        metrics = evaluate(
            model=model,
            sequence_dataset=dataset,
            dt_track_vis=cfg.dt_track_vis,
            sequence_name=seq_name,
            visualize=cfg.visualize,
        )

        logger.info(f"Latency: {metrics['latency']} s")
        results_table.add_row([seq_name, metrics["latency"]])

    logger.info(f"\nTotal evaluated tracks: {total_tracks}")
    logger.info("\n" + results_table.get_string())



if __name__ == "__main__":
    track()
