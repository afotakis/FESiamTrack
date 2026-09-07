"""
Synthetic benchmark:
Compare our tracker results against ground-truth tracks on Multiflow-style data.

Usage:
    RESULTS_ROOT=/path/to/benchmark_tree \
    METHODS=attention,baseline \
    python -m scripts.benchmark
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

# -------------------------------------------------------------------------
# Project root on path
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.track_utils import compute_tracking_errors, read_txt_results

plt.rcParams["font.family"] = "serif"

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Root directory of the benchmark tree produced by make_benchmark_tree.py:
#   RESULTS_ROOT/
#       seq_name/
#           gt/seq_name.gt.txt
#           <method>/seq_name.txt
RESULTS_ROOT = Path(
    os.environ.get("RESULTS_ROOT", ".")
).resolve()

if not RESULTS_ROOT.exists():
    raise FileNotFoundError(
        f"RESULTS_ROOT does not exist: {RESULTS_ROOT}. "
        f"Set RESULTS_ROOT to your synthetic benchmark root."
    )

# Methods to evaluate (comma-separated env var), e.g. METHODS=attention,ours
methods = os.environ.get("METHODS", "attention").split(",")
methods = [m.strip() for m in methods if m.strip()]
if not methods:
    raise ValueError("No methods specified. Set METHODS env var to at least one method name.")

# Discover sequences automatically from RESULTS_ROOT
EVAL_SEQUENCES = sorted(
    [
        d.name
        for d in RESULTS_ROOT.iterdir()
        if d.is_dir() and d.name != "benchmark_results"
    ]
)

if not EVAL_SEQUENCES:
    raise RuntimeError(
        f"No sequence folders found under {RESULTS_ROOT}. "
        f"Expected structure: RESULTS_ROOT/seq_name/gt/seq_name.gt.txt"
    )

# Error thresholds for feature-age / expected-age curves
error_threshold_range = np.arange(1, 32, 1)

# Output directory
out_dir = RESULTS_ROOT / "benchmark_results"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Benchmarking synthetic results in: {RESULTS_ROOT}")
print(f"Output will be written to:          {out_dir}")
print(f"Evaluating methods:                 {methods}")
print(f"Sequences:                          {EVAL_SEQUENCES}")

# -------------------------------------------------------------------------
# Tables to summarize metrics
# -------------------------------------------------------------------------
table_keys = [
    "age_5_mu",
    "age_5_std",
    "te_5_mu",
    "te_5_std",
    "age_mu",
    "age_std",
    "inliers_mu",
    "inliers_std",
    "expected_age",
]

tables = {}
for k in table_keys:
    t = PrettyTable()
    t.title = k
    t.field_names = ["Sequence Name"] + methods
    tables[k] = t

# -------------------------------------------------------------------------
# Evaluation loop
# -------------------------------------------------------------------------
for sequence_name in tqdm(EVAL_SEQUENCES, desc="Benchmarking sequences"):

    gt_path = RESULTS_ROOT / sequence_name / "gt" / f"{sequence_name}.gt.txt"
    if not gt_path.exists():
        print(f"[WARN] Missing GT for {sequence_name}: {gt_path}, skipping.")
        continue

    track_data_gt = read_txt_results(str(gt_path))

    # Initialize per-sequence rows
    rows = {k: [sequence_name] for k in tables.keys()}

    for method in methods:
        pred_path = (
            RESULTS_ROOT
            / sequence_name
            / method
            / f"{sequence_name}.txt"
        )
        if not pred_path.exists():
            print(f"[WARN] Missing predictions for {sequence_name}, method={method}: {pred_path}, filling zeros.")
            # Fill zeros for this method
            for k in rows.keys():
                rows[k].append(0.0)
            continue

        track_data_pred = read_txt_results(str(pred_path))

        # Time alignment safeguard:
        # If first timestamps differ, shift predictions to start at GT time.
        if track_data_pred.shape[0] > 0 and track_data_gt.shape[0] > 0:
            if track_data_pred[0, 1] != track_data_gt[0, 1]:
                dt_shift = track_data_gt[0, 1] - track_data_pred[0, 1]
                track_data_pred[:, 1] += dt_shift

        # -----------------------------------------------------------------
        # Threshold sweep → feature age & inlier statistics
        # -----------------------------------------------------------------
        inlier_ratio_arr = []
        fa_rel_nz_arr = []

        for thresh in error_threshold_range:
            fa_rel, _ = compute_tracking_errors(
                track_data_pred,
                track_data_gt,
                error_threshold=thresh,
                asynchronous=False,
            )

            inlier_ratio = np.sum(fa_rel > 0) / len(fa_rel) if len(fa_rel) > 0 else 0.0
            if inlier_ratio > 0:
                fa_rel_nz = fa_rel[np.nonzero(fa_rel)[0]]
            else:
                fa_rel_nz = np.array([0.0], dtype=np.float32)

            inlier_ratio_arr.append(inlier_ratio)
            fa_rel_nz_arr.append(float(np.mean(fa_rel_nz)))

        inlier_ratio_arr = np.asarray(inlier_ratio_arr, dtype=np.float32)
        fa_rel_nz_arr = np.asarray(fa_rel_nz_arr, dtype=np.float32)

        mean_inlier_ratio = float(np.mean(inlier_ratio_arr))
        std_inlier_ratio = float(np.std(inlier_ratio_arr))
        mean_fa_rel_nz = float(np.mean(fa_rel_nz_arr))
        std_fa_rel_nz = float(np.std(fa_rel_nz_arr))

        # Expected feature age over thresholds
        expected_age = float(np.mean(inlier_ratio_arr * fa_rel_nz_arr))

        rows["age_mu"].append(mean_fa_rel_nz)
        rows["age_std"].append(std_fa_rel_nz)
        rows["inliers_mu"].append(mean_inlier_ratio)
        rows["inliers_std"].append(std_inlier_ratio)
        rows["expected_age"].append(expected_age)

        # -----------------------------------------------------------------
        # Metrics at fixed threshold = 5 px (as in the paper)
        # -----------------------------------------------------------------
        fa_rel_5, te_5 = compute_tracking_errors(
            track_data_pred,
            track_data_gt,
            error_threshold=5,
            asynchronous=False,
        )

        inlier_ratio_5 = np.sum(fa_rel_5 > 0) / len(fa_rel_5) if len(fa_rel_5) > 0 else 0.0
        if inlier_ratio_5 > 0:
            fa_rel_5_nz = fa_rel_5[np.nonzero(fa_rel_5)[0]]
        else:
            fa_rel_5_nz = np.array([0.0], dtype=np.float32)
            te_5 = np.array([0.0], dtype=np.float32)

        mean_fa_rel_5 = float(np.mean(fa_rel_5_nz))
        std_fa_rel_5 = float(np.std(fa_rel_5_nz))
        mean_te_5 = float(np.mean(te_5))
        std_te_5 = float(np.std(te_5))

        rows["age_5_mu"].append(mean_fa_rel_5)
        rows["age_5_std"].append(std_fa_rel_5)
        rows["te_5_mu"].append(mean_te_5)
        rows["te_5_std"].append(std_te_5)

    # Add rows for this sequence to all tables
    for k in tables.keys():
        tables[k].add_row(rows[k])

# -------------------------------------------------------------------------
# Save & print
# -------------------------------------------------------------------------
csv_path = out_dir / "benchmarking_results_synthetic.csv"
with open(csv_path, "w") as f:
    for k in table_keys:
        f.write(f"{k}\n")
        f.write(tables[k].get_csv_string())
        f.write("\n")

for k in table_keys:
    print("\n" + "=" * 80)
    print(tables[k].get_string())
