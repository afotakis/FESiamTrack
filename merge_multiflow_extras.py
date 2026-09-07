#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import shutil
import csv
from collections import defaultdict

EXPECTED_TRACK_FILE = "shitomasi.gt.txt"  # adjust if needed
DEFAULT_RATES = ["0.0100", "0.0200"]      # adapt if you also use 0.0050 etc.

def discover_sequence_dirs(root: Path) -> List[Path]:
    """
    Robustly discover sequence directories anywhere under `root`.
    A sequence dir is defined as any directory that directly contains 'events/' or 'tracks/'.
    This will find things like: <root>/train/<seq>_done/events/...
    """
    if not root.is_dir():
        return []
    seq_dirs = set()
    for marker in ("events", "tracks"):
        for p in root.rglob(marker):
            if p.is_dir():
                seq_dirs.add(p.parent.resolve())
    return sorted(seq_dirs)

def list_reps_and_rates(seq_dir: Path) -> Dict[str, Set[str]]:
    """
    Return mapping: rate -> set(representation_names) found under seq/events/<rate>/<rep_name>/
    """
    out = defaultdict(set)
    events_dir = seq_dir / "events"
    if not events_dir.is_dir():
        return out
    for rate_dir in events_dir.iterdir():
        if not rate_dir.is_dir():
            continue
        rate = rate_dir.name  # e.g., "0.0200"
        for rep_dir in rate_dir.iterdir():
            if rep_dir.is_dir():
                out[rate].add(rep_dir.name)
    return out

def summarize_root(root: Path) -> Dict[str, Dict]:
    """
    Build a summary for printing and planning:
      {
        'root': <str>,
        'sequences': set(<relative_seq_path_str>),
        'per_seq': {
           <relative_seq_path_str>: {
              'rates': { rate: {rep1, rep2, ...} },
              'has_tracks': bool
           }
        }
      }
    NOTE: We store sequence keys as RELATIVE paths (relative to root), so we can copy with that structure preserved.
    """
    summary = {
        "root": str(root),
        "sequences": set(),
        "per_seq": {}
    }
    seq_dirs = discover_sequence_dirs(root)
    for s_abs in seq_dirs:
        try:
            seq_rel = s_abs.relative_to(root)
        except ValueError:
            # Fallback if paths are on different mounts (shouldn't happen here)
            seq_rel = Path(os.path.relpath(s_abs, root))
        rates_map = list_reps_and_rates(s_abs)
        has_tracks = (s_abs / "tracks").is_dir() and any((s_abs / "tracks").iterdir())
        key = str(seq_rel)
        summary["sequences"].add(key)
        summary["per_seq"][key] = {
            "rates": {r: set(sorted(reps)) for r, reps in rates_map.items()},
            "has_tracks": has_tracks
        }
    return summary

def pretty_print_summary(label: str, summary: Dict):
    print(f"\n=== Summary: {label} ===")
    print(f"Root: {summary.get('root')}")
    seqs = sorted(summary["sequences"])
    print(f"Found {len(seqs)} sequence folders")
    for seq_rel in seqs:
        info = summary["per_seq"][seq_rel]
        rates = info["rates"]
        trk = "yes" if info["has_tracks"] else "no"
        print(f"  - {seq_rel} (tracks: {trk})")
        for rate in sorted(rates.keys()):
            reps = ", ".join(sorted(rates[rate])) if rates[rate] else "-"
            print(f"      rate {rate}: {reps}")
    print("=" * 40)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_if_needed(src: Path, dst: Path, conflicts: List[Tuple[Path, Path, str]], dry_run: bool):
    """
    Copy src -> dst if dst missing.
    If dst exists:
      - same size -> skip
      - different size -> record conflict and skip (no overwrite)
    """
    if not dst.exists():
        if dry_run:
            print(f"[DRY] COPY  : {src} -> {dst}")
        else:
            ensure_dir(dst.parent)
            shutil.copy2(src, dst)
            print(f"[COPY] {src} -> {dst}")
        return

    try:
        s1 = os.path.getsize(src)
        s2 = os.path.getsize(dst)
    except OSError:
        s1 = s2 = -1

    if s1 == s2:
        return
    else:
        conflicts.append((src, dst, "different_size"))
        print(f"[CONFLICT] Different size, skip overwrite:\n  src={src}\n  dst={dst}")

def merge_root(
    src: Path,
    dst: Path,
    allowed_rates: Set[str],
    allowed_reps: Set[str],
    include_tracks: bool,
    conflicts: List[Tuple[Path, Path, str]],
    dry_run: bool
):
    """
    Merge content from src -> dst, preserving full relative paths (e.g., train/<seq>_done/...).
    """
    src_summ = summarize_root(src)
    for seq_rel in sorted(src_summ["sequences"]):
        src_seq_abs = (src / seq_rel).resolve()

        # tracks/
        if include_tracks and (src_seq_abs / "tracks").is_dir():
            for f in (src_seq_abs / "tracks").rglob("*"):
                if f.is_file():
                    rel = f.relative_to(src)  # keeps e.g. train/<seq>/tracks/...
                    target = dst / rel
                    copy_if_needed(f, target, conflicts, dry_run)

        # events/<rate>/<rep>/*.h5
        rep_map = src_summ["per_seq"][seq_rel]["rates"]
        for rate, reps in rep_map.items():
            if allowed_rates and rate not in allowed_rates:
                continue
            for rep in reps:
                if allowed_reps and rep not in allowed_reps:
                    continue
                src_rep_dir = src_seq_abs / "events" / rate / rep
                if not src_rep_dir.is_dir():
                    continue
                for f in src_rep_dir.glob("*.h5"):
                    rel = f.relative_to(src)
                    target = dst / rel
                    copy_if_needed(f, target, conflicts, dry_run)

def write_conflicts_csv(conflicts: List[Tuple[Path, Path, str]], out_csv: Path, dry_run: bool):
    if not conflicts:
        return
    if dry_run:
        print(f"[DRY] Would write conflicts CSV to: {out_csv}")
        return
    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["src", "dst", "reason"])
        for s, d, r in conflicts:
            w.writerow([str(s), str(d), r])
    print(f"[INFO] Conflicts written to: {out_csv}")

def parse_args():
    ap = argparse.ArgumentParser(
        description="Merge two MultiFlow extras roots (events + tracks) into one destination safely."
    )
    ap.add_argument("--src", action="append", required=True,
                    help="Source root (can pass twice; order matters).")
    ap.add_argument("--dst", required=True,
                    help="Destination root (often the existing multiflow_reloaded_extra).")
    ap.add_argument("--rates", nargs="*", default=DEFAULT_RATES,
                    help=f"Rates to include (default: {DEFAULT_RATES}). Use --rates ALL to include everything found.")
    ap.add_argument("--reps", nargs="*", default=["time_surfaces_v2_5", "time_surfaces_v2_gpu_5", "voxel_grids_5"],
                    help="Representations to include. Use --reps ALL to include everything found.")
    ap.add_argument("--no-tracks", action="store_true", help="Do NOT merge tracks.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without copying.")
    ap.add_argument("--conflicts-csv", default="merge_conflicts.csv", help="Where to write conflicts CSV.")
    return ap.parse_args()

def main():
    args = parse_args()
    srcs = [Path(p).resolve() for p in args.src]
    dst = Path(args.dst).resolve()
    include_tracks = not args.no_tracks

    # Summaries
    summaries = []
    all_found_rates = set()
    all_found_reps = set()

    for i, s in enumerate(srcs, 1):
        summ = summarize_root(s)
        summaries.append(summ)
        pretty_print_summary(f"SRC#{i}", summ)
        for _, info in summ["per_seq"].items():
            for rate, reps in info["rates"].items():
                all_found_rates.add(rate)
                all_found_reps |= reps

    # Destination summary (optional)
    if dst.exists():
        summ_dst = summarize_root(dst)
        pretty_print_summary("DEST (before merge)", summ_dst)
    else:
        print(f"\n[INFO] Destination does not exist yet: {dst}")

    # Resolve allowed sets
    if len(args.rates) == 1 and args.rates[0].upper() == "ALL":
        allowed_rates = all_found_rates
    else:
        allowed_rates = set(args.rates)

    if len(args.reps) == 1 and args.reps[0].upper() == "ALL":
        allowed_reps = all_found_reps
    else:
        allowed_reps = set(args.reps)

    print("\n=== Merge Plan ===")
    print(f"Destination: {dst}")
    print(f"Include tracks: {include_tracks}")
    print(f"Rates: {sorted(allowed_rates) if allowed_rates else '(none)'}")
    print(f"Representations: {sorted(allowed_reps) if allowed_reps else '(none)'}")
    print(f"Dry run: {args.dry_run}")
    print("="*40)

    conflicts: List[Tuple[Path, Path, str]] = []

    # Merge sources in listed order
    ensure_dir(dst)
    for s in srcs:
        merge_root(
            src=s,
            dst=dst,
            allowed_rates=allowed_rates,
            allowed_reps=allowed_reps,
            include_tracks=include_tracks,
            conflicts=conflicts,
            dry_run=args.dry_run
        )

    # Conflicts report
    write_conflicts_csv(conflicts, dst / args.conflicts_csv, args.dry_run)

    # Final summary
    if not args.dry_run:
        summ_dst_after = summarize_root(dst)
        pretty_print_summary("DEST (after merge)", summ_dst_after)
    else:
        print("\n[DRY] Skipping final summary (no files changed).")

if __name__ == "__main__":
    sys.exit(main())
