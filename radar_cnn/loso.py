"""Leave-one-subject-out: train on all but one subject, test on held-out subject; rotate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from radar_cnn.config import load_yaml, spectrogram_from_dict
from radar_cnn.dataset import RadarSpectrogramDataset, compute_train_statistics
from radar_cnn.labels import activity_to_label, parse_filename
from radar_cnn.model import SmallRadarCNN
from radar_cnn.splits import discover_dat_files, group_files_by_subject
from radar_cnn.train import class_weights_from_train, eval_epoch, set_seed, train_one_epoch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_json", type=str, default="loso_results.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--max_folds", type=int, default=None, help="Limit number of subjects (debug)")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_yaml(args.config)
    spec_cfg = spectrogram_from_dict(cfg.get("spectrogram", {}))
    train_cfg = cfg.get("training", {})

    all_files = discover_dat_files(args.data_root)
    by_sub = group_files_by_subject(all_files, parse_filename)
    subjects = sorted(by_sub.keys())
    if args.max_folds:
        subjects = subjects[: args.max_folds]

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = int(train_cfg.get("cnn_base", 48))

    results = []
    for holdout in tqdm(subjects, desc="LOSO subjects"):
        train_paths = [p for p in all_files if parse_filename(p)[0] != holdout]
        test_paths = [p for p in all_files if parse_filename(p)[0] == holdout]
        if not test_paths:
            continue

        mean, std = compute_train_statistics(train_paths, spec_cfg, cache_dir=cache_dir)
        train_ds = RadarSpectrogramDataset(
            train_paths,
            spec_cfg,
            train_mean=mean,
            train_std=std,
            cache_dir=cache_dir,
            split_name=f"loso_train_{holdout}",
        )
        test_ds = RadarSpectrogramDataset(
            test_paths,
            spec_cfg,
            train_mean=mean,
            train_std=std,
            cache_dir=cache_dir,
            split_name=f"loso_test_{holdout}",
        )

        weights = class_weights_from_train(train_paths).to(device)
        use_head_bn = bool(train_cfg.get("use_head_bn", False))
        model = SmallRadarCNN(num_classes=6, base=base, use_head_bn=use_head_bn).to(device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        crit = nn.CrossEntropyLoss(weight=weights)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        for _ in range(args.epochs):
            train_one_epoch(model, train_loader, opt, crit, device, use_augment=True)

        _, y_true, y_pred = eval_epoch(model, test_loader, crit, device)
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        results.append(
            {
                "holdout_subject": int(holdout),
                "n_test": len(test_paths),
                "accuracy": float(acc),
                "macro_f1": float(mf1),
            }
        )

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    print(f"LOSO mean accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"LOSO mean macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
