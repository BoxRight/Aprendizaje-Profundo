"""Train small CNN with weighted CE and subject-wise split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from radar_cnn.augment import augment_spectrogram
from radar_cnn.config import load_yaml, spectrogram_from_dict
from radar_cnn.dataset import RadarSpectrogramDataset, compute_train_statistics
from radar_cnn.labels import activity_to_label, parse_filename
from radar_cnn.model import SmallRadarCNN, count_parameters
from radar_cnn.splits import (
    assert_disjoint_subject_splits,
    discover_dat_files,
    split_class_counts,
    subject_train_val_test,
)

# Offset so label-shuffle RNG stream is distinct from split seed uses.
_SHUFFLE_LABELS_SEED_OFFSET = 913_337


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights_from_train(paths: list[Path], num_classes: int = 6) -> torch.Tensor:
    labels = [activity_to_label(parse_filename(p)[1]) for p in paths]
    cnt = Counter(labels)
    weights = []
    for c in range(num_classes):
        n = cnt.get(c, 0)
        w = 1.0 / max(n, 1)
        weights.append(w)
    w_t = torch.tensor(weights, dtype=torch.float32)
    w_t = w_t / w_t.mean()
    return w_t


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_augment: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        if use_augment:
            x = augment_spectrogram(x)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_pred = []
    all_y = []
    for batch in loader:
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
        pred = logits.argmax(dim=1)
        all_pred.append(pred.cpu().numpy())
        all_y.append(y.cpu().numpy())
    return total_loss / max(n, 1), np.concatenate(all_y), np.concatenate(all_pred)


def _metric_improved(
    best_metric: str,
    val_loss: float,
    macro_f1: float,
    best_val_loss: float,
    best_f1: float,
) -> bool:
    if best_metric == "val_macro_f1":
        return macro_f1 > best_f1
    return val_loss < best_val_loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="runs/radar_cnn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--stats_max_files", type=int, default=None, help="Cap files for mean/std pass")
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Extra stderr logs during train_stats pass (slow on large data)",
    )
    ap.add_argument(
        "--shuffle_labels",
        action="store_true",
        help="Permute training labels among samples (same class histogram); val/test stay true. Pipeline sanity check.",
    )
    ap.add_argument(
        "--best_metric",
        type=str,
        default="val_macro_f1",
        choices=["val_macro_f1", "val_loss"],
        help="Metric for best.pt and early-stopping patience (default: val_macro_f1).",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Stop if best_metric does not improve for this many epochs (0 = disabled).",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_yaml(args.config)
    spec_cfg = spectrogram_from_dict(cfg.get("spectrogram", {}))
    train_cfg = cfg.get("training", {})

    data_root = Path(args.data_root)
    all_files = discover_dat_files(data_root)
    if not all_files:
        raise SystemExit(f"No .dat files under {data_root}")

    train_paths, val_paths, test_paths = subject_train_val_test(
        all_files,
        parse_filename,
        seed=args.seed,
        fractions=tuple(train_cfg.get("split_fractions", [0.8, 0.1, 0.1])),
    )

    assert_disjoint_subject_splits(train_paths, val_paths, test_paths, parse_filename)

    print("Split sizes:", len(train_paths), len(val_paths), len(test_paths))
    print("Train class counts:", split_class_counts(train_paths, parse_filename, activity_to_label))
    print("Val class counts:", split_class_counts(val_paths, parse_filename, activity_to_label))

    if args.shuffle_labels:
        print(
            "Shuffle-label mode: training labels permuted among samples (val/test unchanged). "
            "Expect chance-level val macro-F1 if pipeline is sound.",
            flush=True,
        )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("  (Install CUDA PyTorch + NVIDIA driver for GPU.)", flush=True)

    mean, std = compute_train_statistics(
        train_paths,
        spec_cfg,
        cache_dir=cache_dir,
        max_files=args.stats_max_files,
        verbose=args.verbose,
    )
    print(f"Dataset normalization: mean={mean:.6f}, std={std:.6f}")

    label_override: np.ndarray | None = None
    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed + _SHUFFLE_LABELS_SEED_OFFSET)
        true_y = np.array(
            [activity_to_label(parse_filename(p)[1]) for p in train_paths],
            dtype=np.int64,
        )
        perm = rng.permutation(len(train_paths))
        label_override = true_y[perm]

    train_ds = RadarSpectrogramDataset(
        train_paths,
        spec_cfg,
        train_mean=mean,
        train_std=std,
        cache_dir=cache_dir,
        split_name="train",
        label_override=label_override,
    )
    val_ds = RadarSpectrogramDataset(
        val_paths,
        spec_cfg,
        train_mean=mean,
        train_std=std,
        cache_dir=cache_dir,
        split_name="val",
    )

    weights = class_weights_from_train(train_paths).to(device)

    use_head_bn = bool(train_cfg.get("use_head_bn", False))
    model = SmallRadarCNN(
        num_classes=6,
        base=int(train_cfg.get("cnn_base", 48)),
        use_head_bn=use_head_bn,
    ).to(device)
    print("Parameters:", count_parameters(model))
    if use_head_bn:
        print("use_head_bn: True (BatchNorm1d on GAP features before classifier)", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    metrics_csv = out_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_macro_f1"],
        )
    print(f"Metrics CSV: {metrics_csv}", flush=True)
    _es = "off" if args.patience <= 0 else "on"
    print(
        f"best_metric={args.best_metric}  patience={args.patience}  (early stopping {_es})",
        flush=True,
    )

    best_f1 = float("-inf")
    best_val_loss = float("inf")
    no_improve = 0
    last_epoch = 0
    early_stopped = False

    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_augment=True,
        )
        val_loss, y_true, y_pred = eval_epoch(model, val_loader, criterion, device)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        last_epoch = epoch + 1
        print(
            f"Epoch {epoch+1}/{args.epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_macro_f1={macro_f1:.4f}"
        )
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch + 1, tr_loss, val_loss, macro_f1])

        improved = _metric_improved(
            args.best_metric, val_loss, macro_f1, best_val_loss, best_f1
        )
        if improved:
            if args.best_metric == "val_macro_f1":
                best_f1 = macro_f1
            else:
                best_val_loss = val_loss
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "spec_cfg": spec_cfg.__dict__,
                    "mean": mean,
                    "std": std,
                    "config_path": str(args.config),
                    "epoch": epoch,
                    "seed": args.seed,
                    "split_fractions": tuple(
                        train_cfg.get("split_fractions", [0.8, 0.1, 0.1])
                    ),
                    "cnn_base": int(train_cfg.get("cnn_base", 48)),
                    "use_head_bn": use_head_bn,
                    "best_metric": args.best_metric,
                    "shuffle_labels": args.shuffle_labels,
                },
                best_path,
            )
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch+1} (no {args.best_metric} improvement for {args.patience} epochs).",
                flush=True,
            )
            early_stopped = True
            break

    print("Saved:", best_path)
    with open(out_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": mean,
                "std": std,
                "train_files": len(train_paths),
                "val_files": len(val_paths),
                "test_files": len(test_paths),
                "shuffle_labels": args.shuffle_labels,
                "best_metric": args.best_metric,
                "patience": args.patience,
                "epochs_ran": last_epoch,
                "early_stopped": early_stopped,
                "use_head_bn": use_head_bn,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
