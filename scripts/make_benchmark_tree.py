#!/usr/bin/env python3
import argparse, os
from pathlib import Path

def main(gt_dir, pred_dir, results_root, method_name="ours_attn", copy=False):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(gt_dir.glob("*.gt.txt"))
    missing = []

    for gt in gt_files:
        # e.g. "boards_01.gt.txt" -> seq_name = "boards_01"
        seq_name = gt.stem.replace(".gt", "")
        seq_dir = results_root / seq_name
        gt_out = seq_dir / "gt" / f"{seq_name}.gt.txt"
        pred_out = seq_dir / method_name / f"{seq_name}.txt"

        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        (seq_dir / method_name).mkdir(parents=True, exist_ok=True)

        # link/copy GT
        if gt_out.exists():
            gt_out.unlink()
        if copy:
            gt_out.write_bytes(gt.read_bytes())
        else:
            os.symlink(gt.resolve(), gt_out)

        # find prediction (search recursively; must be "<seq_name>.txt")
        cand = list(pred_dir.rglob(f"{seq_name}.txt"))
        if not cand:
            missing.append(seq_name)
            continue

        # link/copy prediction
        if pred_out.exists():
            pred_out.unlink()
        if copy:
            pred_out.write_bytes(Path(cand[0]).read_bytes())
        else:
            os.symlink(Path(cand[0]).resolve(), pred_out)

    print(f"Built tree at: {results_root}")
    if missing:
        print("WARNING: missing predictions for:", ", ".join(missing))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--method_name", default="ours_attn")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlink")
    args = ap.parse_args()
    main(**vars(args))
    # Resolve from env if provided (so we can pass PRED_DIR/RESULTS_ROOT via notebook env)
    pred_dir = os.environ.get("PRED_DIR", args.pred_dir)
    results_root = os.environ.get("RESULTS_ROOT", args.results_root)
    if pred_dir is None:
        raise SystemExit("PRED_DIR is not set. Provide --pred_dir or export PRED_DIR in the environment.")
    if results_root is None:
        raise SystemExit("RESULTS_ROOT is not set. Provide --results_root or export RESULTS_ROOT in the environment.")
    main(gt_dir=args.gt_dir, pred_dir=pred_dir, results_root=results_root,
         method_name=args.method_name, copy=args.copy)