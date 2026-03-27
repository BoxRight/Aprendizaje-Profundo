"""Plot train/val loss and val macro-F1 from metrics.csv written by train.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def plot_metrics_csv(csv_path: Path, out_path: Path | None = None, dpi: int = 150) -> Path:
    if not csv_path.is_file():
        raise SystemExit(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("epoch", "train_loss", "val_loss", "val_macro_f1"):
        if col not in df.columns:
            raise SystemExit(f"CSV missing column {col!r}: {list(df.columns)}")

    out_path = out_path or (csv_path.parent / "metrics.png")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(df["epoch"], df["train_loss"], label="train_loss", color="C0")
    ax1.plot(df["epoch"], df["val_loss"], label="val_loss", color="C1")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(csv_path.name)

    ax2.plot(df["epoch"], df["val_macro_f1"], label="val_macro_f1", color="C2")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("macro-F1")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default="runs/exp1/metrics.csv",
        help="Path to metrics.csv",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: same dir as csv, name metrics.png)",
    )
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = plot_metrics_csv(
        csv_path=csv_path,
        out_path=Path(args.out) if args.out else None,
        dpi=args.dpi,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
