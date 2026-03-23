"""Evaluate checkpoint: global metrics + per-subject CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from radar_cnn.dataset import RadarSpectrogramDataset
from radar_cnn.labels import parse_filename
from radar_cnn.model import SmallRadarCNN


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_pred = []
    all_y = []
    all_sid = []
    for batch in loader:
        x, y, sid = batch
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_pred.append(pred)
        all_y.append(y.numpy())
        all_sid.append(np.array(sid))
    return (
        np.concatenate(all_pred),
        np.concatenate(all_y),
        np.concatenate(all_sid),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--output_csv", type=str, default=None)
    ap.add_argument("--cache_dir", type=str, default=None)
    args = ap.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    from radar_cnn.spectrogram import SpectrogramConfig

    spec_cfg = SpectrogramConfig(**ckpt["spec_cfg"])
    mean = float(ckpt["mean"])
    std = float(ckpt["std"])
    seed = int(ckpt.get("seed", 42))
    frac = tuple(ckpt.get("split_fractions", (0.8, 0.1, 0.1)))
    cnn_base = int(ckpt.get("cnn_base", 48))
    use_head_bn = bool(ckpt.get("use_head_bn", False))

    from radar_cnn.splits import discover_dat_files, subject_train_val_test

    all_files = discover_dat_files(args.data_root)
    train_p, val_p, test_p = subject_train_val_test(
        all_files,
        parse_filename,
        seed=seed,
        fractions=frac,
    )
    if args.split == "train":
        paths = train_p
    elif args.split == "val":
        paths = val_p
    else:
        paths = test_p

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    ds = RadarSpectrogramDataset(
        paths,
        spec_cfg,
        train_mean=mean,
        train_std=std,
        cache_dir=cache_dir,
        split_name=args.split,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallRadarCNN(num_classes=6, base=cnn_base, use_head_bn=use_head_bn).to(device)
    model.load_state_dict(ckpt["model"])

    pred, y_true, sids = run_eval(model, loader, device)

    acc = accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    print(f"Split={args.split}  n={len(y_true)}  accuracy={acc:.4f}  macro_f1={macro_f1:.4f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_true, pred))
    print("\n", classification_report(y_true, pred, zero_division=0))

    rows = []
    for sid in np.unique(sids):
        m = sids == sid
        acc_s = accuracy_score(y_true[m], pred[m])
        f1_s = f1_score(y_true[m], pred[m], average="macro", zero_division=0)
        rows.append(
            {
                "subject_id": int(sid),
                "n_files": int(m.sum()),
                "accuracy": acc_s,
                "macro_f1": f1_s,
            }
        )
    df = pd.DataFrame(rows).sort_values("subject_id")
    out_csv = args.output_csv or f"eval_{args.split}_per_subject.csv"
    df.to_csv(out_csv, index=False)
    print(f"Per-subject metrics saved to {out_csv}")
    print("Subject macro-F1 mean:", df["macro_f1"].mean(), "std:", df["macro_f1"].std())


if __name__ == "__main__":
    main()
