"""List every .dat file and which split it belongs to (train/val/test), matching training code."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from radar_cnn.config import load_yaml
from radar_cnn.labels import parse_filename
from radar_cnn.splits import (
    assert_disjoint_subject_splits,
    discover_dat_files,
    subject_train_val_test,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Dump subject-wise train/val/test assignment for all .dat under --data_root. "
            "Uses the same discover_dat_files + subject_train_val_test as train scripts."
        )
    )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML with training.split_fractions (e.g. configs/lstm_multiclass.yaml).",
    )
    ap.add_argument(
        "--split_fractions",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=None,
        help="Override fractions e.g. 0.8 0.1 0.1 (must sum to 1). Ignored if --config sets them.",
    )
    ap.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to write path,split,subject_id (one row per file).",
    )
    args = ap.parse_args()

    fractions: tuple[float, float, float]
    if args.config:
        cfg = load_yaml(args.config)
        train_cfg = cfg.get("training", {})
        fractions = tuple(train_cfg.get("split_fractions", [0.8, 0.1, 0.1]))
    elif args.split_fractions is not None:
        fractions = tuple(args.split_fractions)
    else:
        fractions = (0.8, 0.1, 0.1)

    root = Path(args.data_root)
    all_files = discover_dat_files(root)
    if not all_files:
        raise SystemExit(f"No .dat files under {root}")

    train_p, val_p, test_p = subject_train_val_test(
        all_files,
        parse_filename,
        seed=args.seed,
        fractions=fractions,
    )
    assert_disjoint_subject_splits(train_p, val_p, test_p, parse_filename)

    def sid_set(paths: list[Path]) -> list[int]:
        return sorted({parse_filename(p)[0] for p in paths})

    print(f"data_root: {root.resolve()}")
    print(f"seed: {args.seed}  split_fractions: {fractions}")
    print(f"total .dat files: {len(all_files)}")
    print(
        f"train: {len(train_p)} files, n_subjects={len(sid_set(train_p))}  "
        f"subject_ids: {sid_set(train_p)}"
    )
    print(
        f"val:   {len(val_p)} files, n_subjects={len(sid_set(val_p))}  "
        f"subject_ids: {sid_set(val_p)}"
    )
    print(
        f"test:  {len(test_p)} files, n_subjects={len(sid_set(test_p))}  "
        f"subject_ids: {sid_set(test_p)}"
    )

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for p in train_p:
            rows.append((str(p.resolve()), "train", parse_filename(p)[0]))
        for p in val_p:
            rows.append((str(p.resolve()), "val", parse_filename(p)[0]))
        for p in test_p:
            rows.append((str(p.resolve()), "test", parse_filename(p)[0]))
        rows.sort(key=lambda x: (x[1], x[2], x[0]))
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "split", "subject_id"])
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
